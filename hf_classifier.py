import streamlit as st
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EvalPrediction
)
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2 # Import Chi-Squared
import os
import json
import traceback
import re
import config
from typing import List, Dict, Any, Tuple, Optional

# --- Data Preparation ---
def prepare_hierarchical_training_data(
    df: pd.DataFrame,
    text_col: str,
    hierarchy_cols: Dict[str, Optional[str]]
) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    """Prepares training data for HF by creating prefixed hierarchical labels."""
    # (Implementation remains the same)
    if df is None or not text_col: st.error("HF Prep: Training DataFrame or text column is missing."); return None, None
    if text_col not in df.columns: st.error(f"HF Prep: Selected text column '{text_col}' not found."); return None, None
    valid_hierarchy_cols = {level: col for level, col in hierarchy_cols.items() if col and col != "(None)"}
    if not valid_hierarchy_cols: st.error("HF Prep: No hierarchy columns selected."); return None, None
    missing_cols = [col for col in valid_hierarchy_cols.values() if col not in df.columns]
    if missing_cols: st.error(f"HF Prep: Hierarchy columns not found: {', '.join(missing_cols)}"); return None, None

    st.info("HF Prep: Preparing training data with hierarchical prefixes...")
    all_texts, all_prefixed_labels = [], []
    error_count = 0
    with st.spinner("Processing training rows for HF..."):
        for index, row in df.iterrows():
            try:
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                all_texts.append(text)
                row_labels = set()
                for level, col_name in valid_hierarchy_cols.items():
                    if col_name in row and pd.notna(row[col_name]):
                        cell_value = str(row[col_name]).strip()
                        if cell_value:
                            values = [v.strip() for v in cell_value.replace(';',',').split(',') if v.strip()]
                            for value in values: row_labels.add(f"{level}: {value}")
                all_prefixed_labels.append(list(row_labels))
            except Exception as e:
                 error_count += 1
                 if error_count <= 10: st.warning(f"HF Prep: Skipping row {index} due to error: {e}")
                 all_texts.append(""); all_prefixed_labels.append([])
    if error_count > 0: st.warning(f"HF Prep: Finished with errors in {error_count} rows.")
    if not any(all_prefixed_labels): st.error("HF Prep: NO labels generated."); return None, None
    st.success(f"HF Prep: Data preparation complete ({len(all_texts)} texts).")
    return all_texts, all_prefixed_labels

# --- Training Callback ---
class StProgressCallback(TrainerCallback):
    """Custom HF Trainer Callback for Streamlit progress."""
    # (Implementation remains the same)
    def __init__(self, progress_bar, progress_text, status_text, total_steps):
        self.progress_bar = progress_bar
        self.progress_text = progress_text
        self.status_text = status_text
        self.total_steps = total_steps
        self.start_step_progress = 0.6

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        if self.total_steps > 0:
            step_progress = (current_step / self.total_steps) * (0.95 - self.start_step_progress)
            total_progress = self.start_step_progress + step_progress
            self.progress_bar.progress(min(total_progress, 0.95))
            epoch_num = state.epoch if state.epoch is not None else 0
            self.progress_text.text(f"Training: Step {current_step}/{self.total_steps} | Epoch {epoch_num:.2f}")
        else:
            self.progress_text.text(f"Training step {current_step}...")

# --- Compute Metrics ---
def compute_metrics(p: EvalPrediction):
    """Computes evaluation metrics for multi-label classification."""
    # (Implementation remains the same)
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    threshold = 0.5
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels.astype(int)
    y_pred = y_pred.astype(int)
    subset_accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    micro_precision, micro_recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    return { 'subset_accuracy': subset_accuracy, 'micro_f1': micro_f1, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
             'micro_precision': micro_precision, 'micro_recall': micro_recall }

# --- Model Training ---
def train_hf_model(
    all_train_texts: List[str],
    all_train_labels_list: List[List[str]],
    model_choice: str,
    num_epochs: int,
    validation_split_ratio: float = config.DEFAULT_VALIDATION_SPLIT
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, int]], pd.DataFrame]:
    """Trains the HF classification model with validation split."""
    # (Implementation remains the same as previous version with validation)
    st.info(f"Starting HF model training with {validation_split_ratio*100:.1f}% validation split...")
    model, tokenizer, label_map, rules_df = None, None, None, pd.DataFrame()

    # 1. Process Labels
    st.write("HF Train: Processing labels...")
    all_labels_set = set(label for sublist in all_train_labels_list for label in sublist if label)
    if not all_labels_set: st.error("HF Train: No valid labels found."); return None, None, None, pd.DataFrame()
    unique_labels = sorted(list(all_labels_set))
    label_map = {label: i for i, label in enumerate(unique_labels)}; num_labels = len(unique_labels)
    st.write(f"HF Train: Found {num_labels} unique prefixed labels.")
    def encode_labels(labels: List[str]) -> np.ndarray:
        encoded = np.zeros(num_labels, dtype=np.float32)
        for label in labels:
            if label in label_map: encoded[label_map[label]] = 1.0
        return encoded
    all_encoded_labels = [encode_labels(labels) for labels in all_train_labels_list]

    # 2. Split Data
    st.write("HF Train: Splitting data...")
    try:
        train_texts, val_texts, train_labels_encoded, val_labels_encoded = train_test_split(
            all_train_texts, all_encoded_labels, test_size=validation_split_ratio, random_state=42)
        st.success(f"Data split: {len(train_texts)} train, {len(val_texts)} validation.")
        if not train_texts or not val_texts: raise ValueError("Empty train/val set after split.")
    except ValueError as e:
         if "test_size" in str(e): st.error(f"HF Train: Invalid validation split ratio ({validation_split_ratio}).")
         else: st.error(f"HF Train: Error splitting data: {e}")
         return None, None, None, pd.DataFrame()
    except Exception as e: st.error(f"HF Train: Error splitting data: {e}"); return None, None, None, pd.DataFrame()

    progress_bar = st.progress(0); progress_text = st.empty(); status_text = st.empty()

    try:
        # 3. Load Model/Tokenizer
        status_text.text("HF Train: Loading tokenizer..."); progress_bar.progress(0.1)
        with st.spinner(f'Loading tokenizer {model_choice}...'): tokenizer = AutoTokenizer.from_pretrained(model_choice)
        status_text.text("HF Train: Loading model..."); progress_bar.progress(0.2)
        with st.spinner(f'Loading model {model_choice}...'):
            model = AutoModelForSequenceClassification.from_pretrained(model_choice, num_labels=num_labels, problem_type="multi_label_classification", ignore_mismatched_sizes=True)
            if model.config.num_labels != num_labels: model.config.num_labels = num_labels

        # 4. Tokenize Sets
        status_text.text("HF Train: Tokenizing..."); progress_bar.progress(0.4)
        with st.spinner('Tokenizing...'):
            train_texts_clean = [str(t) if pd.notna(t) else "" for t in train_texts]
            val_texts_clean = [str(t) if pd.notna(t) else "" for t in val_texts]
            train_encodings = tokenizer(train_texts_clean, truncation=True, padding=True, max_length=512)
            val_encodings = tokenizer(val_texts_clean, truncation=True, padding=True, max_length=512)

        # 5. Create Datasets
        status_text.text("HF Train: Creating datasets..."); progress_bar.progress(0.5)
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, e, l): self.encodings, self.labels = e, l
            def __getitem__(self, i): return {k: torch.tensor(v[i]) for k, v in self.encodings.items()} | {'labels': torch.tensor(self.labels[i], dtype=torch.float)}
            def __len__(self): return len(self.encodings['input_ids']) if 'input_ids' in self.encodings else 0
        train_dataset = TextDataset(train_encodings, train_labels_encoded)
        eval_dataset = TextDataset(val_encodings, val_labels_encoded)
        if len(train_dataset) == 0 or len(eval_dataset) == 0: raise ValueError("Dataset empty after encoding.")

        # 6. Training Arguments
        status_text.text("HF Train: Setting up training..."); progress_bar.progress(0.55)
        is_large = any(m in model_choice for m in ["large", "bert-base", "roberta-base"])
        bs = max(1, (4 if is_large else 8) // (2 if torch.cuda.is_available() else 1))
        gas = max(1, 16 // bs)
        ws = int(os.environ.get("WORLD_SIZE", 1))
        updates_per_epoch = max(1, len(train_dataset) // (bs * gas * ws)) + (1 if len(train_dataset) % (bs * gas * ws) != 0 else 0)
        total_steps = max(1, updates_per_epoch * num_epochs)
        training_args = TrainingArguments(
            output_dir='./results_hf_training', num_train_epochs=num_epochs,
            per_device_train_batch_size=bs, per_device_eval_batch_size=bs * 2,
            gradient_accumulation_steps=gas, warmup_ratio=0.1, weight_decay=0.01,
            logging_dir='./logs_hf_training', logging_strategy="epoch", eval_strategy="epoch",
            save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="micro_f1", greater_is_better=True, report_to="none",
            fp16=torch.cuda.is_available(), dataloader_pin_memory=torch.cuda.is_available(),
            dataloader_num_workers=min(4, os.cpu_count() // 2) if torch.cuda.is_available() else 0 )
        progress_callback = StProgressCallback(progress_bar, progress_text, status_text, total_steps)

        # 7. Initialize Trainer
        trainer = Trainer( model=model, args=training_args, train_dataset=train_dataset,
                           eval_dataset=eval_dataset, compute_metrics=compute_metrics, callbacks=[progress_callback] )
        progress_bar.progress(0.6)

        # 8. Train
        status_text.text("HF Train: Starting training...")
        train_result = trainer.train()
        metrics = train_result.metrics; st.write("Training completed. Metrics:"); st.json(metrics)

        # 9. Extract Rules (using original full data)
        status_text.text("HF Train: Extracting rules..."); progress_bar.progress(0.95)
        rules_df = extract_hf_rules(all_train_texts, all_encoded_labels, label_map) # Calls updated func

        progress_bar.progress(1.0); status_text.text("HF Training complete!"); progress_text.text("")
        best_model = trainer.model; best_model.cpu()
        return best_model, tokenizer, label_map, rules_df

    except Exception as e:
        st.error(f"HF Train Error: {e}"); st.error(traceback.format_exc())
        progress_text.text("HF Training failed.")
        return None, None, None, pd.DataFrame()


# --- Model Saving/Loading ---
# (Keep save_hf_model_artifacts and load_hf_model_artifacts as they were in the previous response,
# including the handling of the 'Custom Keywords' column)
def save_hf_model_artifacts(model: Any, tokenizer: Any, label_map: Dict[str, int], rules_df: pd.DataFrame):
    # (Implementation remains the same)
    path = config.SAVED_HF_MODEL_PATH
    if model is None or tokenizer is None or label_map is None or rules_df is None: st.error("HF Save: Components missing."); return False
    try:
        st.info(f"HF Save: Saving artifacts to {path}...")
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path); tokenizer.save_pretrained(path)
        with open(os.path.join(path, "label_map.json"), 'w') as f: json.dump(label_map, f, indent=4)
        cols_to_save = ['Label', 'Keywords', 'Confidence Threshold', 'Custom Keywords']
        rules_df_to_save = rules_df.copy()
        for col in cols_to_save:
            if col not in rules_df_to_save.columns:
                if col == 'Custom Keywords': rules_df_to_save[col] = ''
                elif col == 'Keywords': rules_df_to_save[col] = 'N/A'
                elif col == 'Confidence Threshold': rules_df_to_save[col] = 0.5
                else: rules_df_to_save[col] = None
        rules_df_to_save = rules_df_to_save[cols_to_save]
        rules_df_to_save.to_csv(os.path.join(path, "rules.csv"), index=False)
        st.success(f"✅ HF Model artifacts saved to {path}")
        return True
    except Exception as e: st.error(f"HF Save Error: {e}"); st.error(traceback.format_exc()); return False

def load_hf_model_artifacts() -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, int]], Optional[pd.DataFrame]]:
    # (Implementation remains the same)
    path = config.SAVED_HF_MODEL_PATH
    st.info(f"HF Load: Attempting load from {path}...")
    if not os.path.isdir(path): st.error(f"HF Load Error: Directory not found: {path}"); return None, None, None, None
    model, tokenizer, label_map, rules_df = None, None, None, None
    try:
        label_map_path = os.path.join(path, "label_map.json")
        rules_path = os.path.join(path, "rules.csv")
        config_path = os.path.join(path, "config.json")
        tok_config_path = os.path.join(path, "tokenizer_config.json")

        if not all(os.path.exists(p) for p in [config_path, tok_config_path, label_map_path]): st.error("HF Load Error: Essential file(s) missing."); return None, None, None, None
        with open(label_map_path, 'r') as f: label_map = json.load(f)

        if not os.path.exists(rules_path):
            st.warning("HF Load: Rules file not found. Creating defaults.")
            rules_df = pd.DataFrame({'Label': list(label_map.keys()), 'Keywords': 'N/A', 'Confidence Threshold': 0.5, 'Custom Keywords': ''})
        else:
             rules_df = pd.read_csv(rules_path)
             if 'Label' not in rules_df.columns: raise ValueError("Rules file missing 'Label'.")
             if 'Confidence Threshold' not in rules_df.columns: rules_df['Confidence Threshold'] = 0.5
             if 'Keywords' not in rules_df.columns: rules_df['Keywords'] = "N/A"
             if 'Custom Keywords' not in rules_df.columns: rules_df['Custom Keywords'] = ''
             rules_df['Confidence Threshold'] = pd.to_numeric(rules_df['Confidence Threshold'], errors='coerce').fillna(0.5).clip(0.05, 0.95)
             rules_df['Custom Keywords'] = rules_df['Custom Keywords'].fillna('')
             rules_df['Keywords'] = rules_df['Keywords'].fillna('N/A')
             if set(rules_df['Label']) != set(label_map.keys()): st.warning("HF Load Warning: Rules labels mismatch label map.")

        with st.spinner("HF Load: Loading model..."): model = AutoModelForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True)
        if model.config.num_labels != len(label_map): model.config.num_labels = len(label_map)
        with st.spinner("HF Load: Loading tokenizer..."): tokenizer = AutoTokenizer.from_pretrained(path)
        st.success(f"✅ HF Model artifacts loaded from {path}")
        return model, tokenizer, label_map, rules_df
    except Exception as e: st.error(f"HF Load Error: {e}"); st.error(traceback.format_exc()); return None, None, None, None


# --- *** UPDATED Rule Extraction using Chi-Squared *** ---
def extract_hf_rules(
    full_train_texts: List[str],
    full_train_labels_encoded: List[np.ndarray],
    label_map: Dict[str, int]
    ) -> pd.DataFrame:
    """Extracts keyword associations using Chi-Squared AND adds 'Custom Keywords' column."""
    st.info("HF Rules: Extracting keyword associations using Chi-Squared...")
    required_columns = ['Label', 'Keywords', 'Confidence Threshold', 'Custom Keywords']
    if not full_train_texts or full_train_labels_encoded is None or not label_map:
         st.warning("HF Rules: Cannot extract - missing data/map.")
         return pd.DataFrame(columns=required_columns)

    reverse_label_map = {v: k for k, v in label_map.items()}
    rules = []
    num_labels = len(label_map)
    train_labels_array = np.array(full_train_labels_encoded)
    if train_labels_array.shape[0] != len(full_train_texts) or train_labels_array.shape[1] != num_labels:
         st.error("HF Rules: Shape mismatch texts vs labels."); return pd.DataFrame(columns=required_columns)

    # --- Vectorization ---
    try:
        with st.spinner("HF Rules: Vectorizing text..."):
            # Using CountVectorizer suitable for chi2
            vectorizer = CountVectorizer(max_features=1500, stop_words='english', binary=False, min_df=3) # Use counts, slightly more features
            X = vectorizer.fit_transform(full_train_texts)
            feature_names = vectorizer.get_feature_names_out()
            if X.shape[0] == 0 or X.shape[1] == 0: # Check if matrix is empty
                 st.warning("HF Rules: Vectorization resulted in an empty matrix.")
                 return pd.DataFrame({'Label': list(label_map.keys()), 'Keywords': 'N/A (Empty Matrix)', 'Confidence Threshold': 0.5, 'Custom Keywords': ''})
    except ValueError as e:
         st.warning(f"HF Rules: Vectorization failed: {e}")
         return pd.DataFrame({'Label': list(label_map.keys()), 'Keywords': 'N/A (Vectorization Error)', 'Confidence Threshold': 0.5, 'Custom Keywords': ''})
    except Exception as e:
        st.warning(f"HF Rules: Vectorization error: {e}"); return pd.DataFrame(columns=required_columns)

    # --- Chi-Squared Calculation ---
    st.info("HF Rules: Calculating Chi-Squared scores...")
    default_entry = {'Keywords': "N/A", 'Confidence Threshold': 0.5, 'Custom Keywords': ''}
    with st.spinner("Analyzing feature relevance..."):
        for label_idx in range(num_labels):
            label_name = reverse_label_map.get(label_idx, f"Unknown_{label_idx}")
            y = train_labels_array[:, label_idx] # Target vector (0s and 1s)

            # Skip if label has no variance (all same value)
            if np.std(y) < 1e-9:
                keywords = "N/A (No variance in label)"
            else:
                try:
                    # Calculate chi2 scores
                    chi2_scores, _ = chi2(X, y) # X should be non-negative counts

                    # Handle potential NaNs returned by chi2 (e.g., if feature is zero everywhere)
                    feature_scores = [(name, score) for name, score in zip(feature_names, chi2_scores) if not np.isnan(score)]

                    # Get top N features (e.g., 7) based on chi2 score
                    top_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)[:7]

                    # Format keywords string - just list the words, scores aren't very intuitive here
                    keywords = ', '.join([word for word, score in top_features])
                    if not keywords: keywords = "N/A (No features found)"

                except Exception as e_chi2:
                    st.warning(f"HF Rules: Error calculating Chi2 for '{label_name}': {e_chi2}")
                    keywords = "N/A (Calculation Error)"

            rules.append({'Label': label_name, **default_entry, 'Keywords': keywords})

    st.success("HF Rules: Extraction finished.")
    return pd.DataFrame(rules)


# --- MODIFIED HF Classification (to use custom keywords) ---
def classify_texts_with_hf(texts: List[str], model: Any, tokenizer: Any, label_map: Dict[str, int], rules_df: Optional[pd.DataFrame]) -> List[List[str]]:
    """Classifies texts using the HF model and applies custom keyword overrides."""
    # (Implementation remains the same as previous response with keyword override)
    if not texts: return []
    if model is None or tokenizer is None or label_map is None:
        st.error("HF Classify: Model/tokenizer/map missing.")
        return [["Classification Error: Missing components"] for _ in texts]

    thresholds, custom_keywords_map = {}, {}
    default_threshold = config.DEFAULT_HF_THRESHOLD

    if rules_df is None or rules_df.empty:
         st.info(f"HF Classify: No rules. Using default threshold ({default_threshold}).")
         thresholds = {label: default_threshold for label in label_map.keys()}
    elif 'Label' not in rules_df.columns or 'Confidence Threshold' not in rules_df.columns:
         st.warning("HF Classify: Rules missing columns. Using default threshold.")
         thresholds = {label: default_threshold for label in label_map.keys()}
    else:
        st.info(f"HF Classify: Using thresholds/custom keywords from rules.")
        try:
            rules_df['Confidence Threshold'] = pd.to_numeric(rules_df['Confidence Threshold'], errors='coerce').fillna(default_threshold).clip(0.05, 0.95)
            thresholds = dict(zip(rules_df['Label'], rules_df['Confidence Threshold']))
            for label in label_map.keys():
                if label not in thresholds: thresholds[label] = default_threshold
            if 'Custom Keywords' in rules_df.columns:
                 rules_df['Custom Keywords'] = rules_df['Custom Keywords'].fillna('')
                 for _, row in rules_df.iterrows():
                     kw_list = [kw.strip().lower() for kw in str(row['Custom Keywords']).split(',') if kw.strip()]
                     if kw_list: custom_keywords_map[row['Label']] = kw_list
            else: st.info("HF Classify: 'Custom Keywords' column not found.")
        except Exception as e: st.error(f"HF Classify: Error processing rules: {e}. Using defaults."); thresholds = {l: default_threshold for l in label_map.keys()}; custom_keywords_map = {}

    st.info("HF Classify: Starting classification...")
    reverse_label_map = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)
    results = []
    batch_size = 16
    progress_bar = st.progress(0); progress_text = st.empty()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); st.write(f"HF Classification on: {device}")
    model.to(device); model.eval()
    texts_clean = [str(text) if pd.notna(text) else "" for text in texts]; total_texts = len(texts_clean)

    for i in range(0, total_texts, batch_size):
        batch_texts = texts_clean[i:min(i + batch_size, total_texts)]
        if not batch_texts: continue
        current_batch_size = len(batch_texts)
        progress_text.text(f"HF Classifying: {i + 1}-{i + current_batch_size}/{total_texts}...")
        try:
            inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad(): outputs = model(**inputs); probs = torch.sigmoid(outputs.logits).cpu().numpy()

            batch_final_labels = []
            for j in range(current_batch_size):
                probs_row = probs[j]; text_content_lower = batch_texts[j].lower()
                initial_labels = set()
                for label_idx in range(num_labels):
                    if label_idx in reverse_label_map:
                         label_name = reverse_label_map[label_idx]
                         threshold = thresholds.get(label_name, default_threshold)
                         if probs_row[label_idx] >= threshold: initial_labels.add(label_name)

                final_labels = initial_labels.copy()
                if custom_keywords_map:
                    for label_name, keywords in custom_keywords_map.items():
                        if label_name in final_labels: continue
                        keyword_found = False
                        for kw in keywords:
                             if kw:
                                 try:
                                     if re.search(r'\b' + re.escape(kw) + r'\b', text_content_lower): keyword_found = True; break
                                 except re.error:
                                      if kw in text_content_lower: keyword_found = True; break
                        if keyword_found: final_labels.add(label_name)

                if not final_labels and len(probs_row) > 0:
                    highest_prob_idx = np.argmax(probs_row)
                    if highest_prob_idx in reverse_label_map:
                        fallback_label = reverse_label_map[highest_prob_idx]
                        if probs_row[highest_prob_idx] > 0.1: final_labels.add(fallback_label)
                batch_final_labels.append(list(final_labels))
            results.extend(batch_final_labels)

        except Exception as e: st.error(f"HF Classify Error (Batch {i}): {e}"); st.error(traceback.format_exc()); results.extend([["Classification Error"] for _ in range(current_batch_size)])
        progress = min(1.0, (i + current_batch_size) / total_texts) if total_texts > 0 else 1.0; progress_bar.progress(progress)

    progress_text.text("HF Classification complete!"); st.success("HF Classification finished.")
    st.info("View results in the Results tab")
    return results