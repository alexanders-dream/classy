# hf_classifier.py
"""Functions for the Hugging Face Transformers classification workflow."""

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
from sklearn.feature_selection import chi2
import os
import json
import traceback
import re
from pathlib import Path
import config
from typing import List, Dict, Any, Tuple, Optional

# --- Data Preparation ---
# (Keep prepare_hierarchical_training_data as is)
def prepare_hierarchical_training_data(
    df: pd.DataFrame,
    text_col: str,
    hierarchy_cols: Dict[str, Optional[str]]
) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    """Prepares training data for HF by creating prefixed hierarchical labels."""
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
# (Keep StProgressCallback as is)
class StProgressCallback(TrainerCallback):
    def __init__(self, progress_bar, progress_text, status_text, total_steps):
        self.progress_bar = progress_bar; self.progress_text = progress_text; self.status_text = status_text; self.total_steps = total_steps; self.start_step_progress = 0.6
    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        if self.total_steps > 0:
            step_prog = (current_step / self.total_steps) * (0.95 - self.start_step_progress); total_prog = self.start_step_progress + step_prog
            self.progress_bar.progress(min(total_prog, 0.95)); epoch = state.epoch if state.epoch is not None else 0
            self.progress_text.text(f"Training: Step {current_step}/{self.total_steps} | Epoch {epoch:.2f}")
        else: self.progress_text.text(f"Training step {current_step}...")

# --- Compute Metrics ---
# (Keep compute_metrics as is)
def compute_metrics(p: EvalPrediction):
    logits=p.predictions[0] if isinstance(p.predictions,tuple) else p.predictions; labels=p.label_ids; sig=torch.nn.Sigmoid(); probs=sig(torch.Tensor(logits)); thr=0.5; y_p=np.zeros(probs.shape); y_p[np.where(probs>=thr)]=1; y_t=labels.astype(int); y_p=y_p.astype(int); sa=accuracy_score(y_t,y_p); mif1=f1_score(y_t,y_p,average='micro',zero_division=0); maf1=f1_score(y_t,y_p,average='macro',zero_division=0); wf1=f1_score(y_t,y_p,average='weighted',zero_division=0); mip,mir,_,_=precision_recall_fscore_support(y_t,y_p,average='micro',zero_division=0); return {'subset_accuracy':sa,'micro_f1':mif1,'macro_f1':maf1,'weighted_f1':wf1,'micro_precision':mip,'micro_recall':mir}

# --- Model Training (Corrected) ---
def train_hf_model(
    all_train_texts: List[str],
    all_train_labels_list: List[List[str]],
    model_choice: str,
    num_epochs: int,
    validation_split_ratio: float = config.DEFAULT_VALIDATION_SPLIT
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, int]], pd.DataFrame]:
    """Trains the HF classification model with validation split."""

    st.info(f"Starting HF model training with {validation_split_ratio*100:.1f}% validation split...")
    model, tokenizer, label_map, rules_df = None, None, None, pd.DataFrame()

    # 1. Process Labels
    st.write("HF Train: Processing labels...")
    all_labels_set = set(l for sub in all_train_labels_list for l in sub if l)
    if not all_labels_set:
        st.error("HF Train: No valid labels found."); return None, None, None, pd.DataFrame()
    unique_labels = sorted(list(all_labels_set))
    label_map = {l: i for i, l in enumerate(unique_labels)}; n_labels = len(unique_labels)
    st.write(f"HF Train: {n_labels} unique labels found.")
    def encode_labels(lbls: List[str]) -> np.ndarray:
        e = np.zeros(n_labels, dtype=np.float32)
        for l in lbls:
            if l in label_map: e[label_map[l]] = 1.0 # Use indexed assignment
        return e
    all_encoded_labels = [encode_labels(lbls) for lbls in all_train_labels_list] # Correct variable name usage

    # 2. Split Data
    st.write("HF Train: Splitting data...")
    try:
        train_texts, val_texts, train_labels_encoded, val_labels_encoded = train_test_split(
            all_train_texts, all_encoded_labels, test_size=validation_split_ratio, random_state=42) # Use correct variable
        st.success(f"Data split: {len(train_texts)} train, {len(val_texts)} validation.")
        if not train_texts or not val_texts: raise ValueError("Empty train/val set after split.")
    except ValueError as e:
        if "test_size" in str(e): st.error(f"HF Train: Invalid validation split ratio ({validation_split_ratio}).")
        else: st.error(f"HF Train: Error splitting data: {e}")
        return None, None, None, pd.DataFrame()
    except Exception as e:
        st.error(f"HF Train: Error splitting data: {e}"); return None, None, None, pd.DataFrame()

    # Progress setup
    progress_bar = st.progress(0)
    progress_text = st.empty()
    status_text = st.empty()

    try:
        # 3. Load Model/Tokenizer
        status_text.text("HF Train: Loading tokenizer...")
        progress_bar.progress(0.1)
        with st.spinner(f'Loading tokenizer {model_choice}...'):
            tokenizer = AutoTokenizer.from_pretrained(model_choice)

        status_text.text("HF Train: Loading model...")
        progress_bar.progress(0.2)
        with st.spinner(f'Loading model {model_choice}...'):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_choice, num_labels=n_labels, problem_type="multi_label_classification", ignore_mismatched_sizes=True)
            if model.config.num_labels != n_labels:
                model.config.num_labels = n_labels

        # 4. Tokenize Sets
        status_text.text("HF Train: Tokenizing...")
        progress_bar.progress(0.4)
        with st.spinner('Tokenizing...'):
            train_texts_clean = [str(t) if pd.notna(t) else "" for t in train_texts]
            val_texts_clean = [str(t) if pd.notna(t) else "" for t in val_texts]
            train_encodings = tokenizer(train_texts_clean, truncation=True, padding=True, max_length=512)
            val_encodings = tokenizer(val_texts_clean, truncation=True, padding=True, max_length=512)

        # 5. Create Datasets
        status_text.text("HF Train: Creating datasets...")
        progress_bar.progress(0.5)
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                 self.encodings = encodings
                 self.labels = labels
            def __getitem__(self, idx):
                 item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                 item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                 return item
            def __len__(self):
                 return len(self.encodings.get('input_ids', [])) # Safer access

        # Use correct encoded labels variables here
        train_dataset = TextDataset(train_encodings, train_labels_encoded)
        eval_dataset = TextDataset(val_encodings, val_labels_encoded)
        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            raise ValueError("Dataset empty after encoding.")

        # 6. Training Arguments
        status_text.text("HF Train: Setting up training args...")
        progress_bar.progress(0.55)
        is_large = any(m in model_choice for m in ["large", "bert-base", "roberta-base"])
        bs = max(1, (4 if is_large else 8) // (2 if torch.cuda.is_available() else 1))
        gas = max(1, 16 // bs)
        ws = int(os.environ.get("WORLD_SIZE", 1))
        updates_per_epoch = max(1, len(train_dataset) // (bs * gas * ws)) + (1 if len(train_dataset) % (bs * gas * ws) != 0 else 0)
        total_steps = max(1, updates_per_epoch * num_epochs)
        training_args = TrainingArguments(
            output_dir='./results_hf_training', # Changed dir name slightly
            num_train_epochs=num_epochs,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs * 2,
            gradient_accumulation_steps=gas,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir='./logs_hf_training', # Changed dir name slightly
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=torch.cuda.is_available(),
            dataloader_num_workers=min(4, os.cpu_count() // 2) if torch.cuda.is_available() else 0
        )
        progress_callback = StProgressCallback(progress_bar, progress_text, status_text, total_steps)

        # 7. Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[progress_callback]
        )
        progress_bar.progress(0.6)

        # 8. Train
        status_text.text("HF Train: Starting training...")
        train_result = trainer.train()
        metrics = train_result.metrics
        st.write("Training completed. Metrics:")
        st.json(metrics)

        # 9. Extract Rules (using original full data)
        status_text.text("HF Train: Extracting rules...")
        progress_bar.progress(0.95)
        # Pass the correct variable containing all encoded labels
        rules_df = extract_hf_rules(all_train_texts, all_encoded_labels, label_map)

        progress_bar.progress(1.0)
        status_text.text("HF Training Done!")
        progress_text.text("")
        best_model = trainer.model
        best_model.cpu()
        return best_model, tokenizer, label_map, rules_df

    except Exception as e:
        st.error(f"HF Train Error: {e}")
        st.error(traceback.format_exc())
        progress_text.text("HF Train Failed.")
        return None, None, None, pd.DataFrame()


# --- Model Saving/Loading ---
# (Keep save_hf_model_artifacts and load_hf_model_artifacts as they were - accepting path arg)
def save_hf_model_artifacts(model: Any, tokenizer: Any, label_map: Dict[str, int], rules_df: pd.DataFrame, save_path: str):
    """Saves the HF model artifacts to the specified user path."""
    if model is None or tokenizer is None or label_map is None or rules_df is None: st.error("HF Save: Components missing."); return False
    try:
        save_path_obj = Path(save_path); st.info(f"HF Save: Saving to '{save_path_obj}'...")
        save_path_obj.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path_obj); tokenizer.save_pretrained(save_path_obj)
        with open(save_path_obj / "label_map.json", 'w') as f: json.dump(label_map, f, indent=4)
        cols_to_save = ['Label', 'Keywords', 'Confidence Threshold']; rules_df_to_save = rules_df.copy()
        for col in cols_to_save:
            if col not in rules_df_to_save.columns:
                if col == 'Keywords': rules_df_to_save[col] = 'N/A'
                elif col == 'Confidence Threshold': rules_df_to_save[col] = 0.5
                else: rules_df_to_save[col] = None
        rules_df_to_save = rules_df_to_save[cols_to_save]
        rules_df_to_save['Confidence Threshold'] = pd.to_numeric(rules_df_to_save['Confidence Threshold'], errors='coerce').fillna(0.5).clip(0.05, 0.95)
        rules_df_to_save.to_csv(save_path_obj / "rules.csv", index=False)
        st.success(f"✅ HF Model saved to '{save_path_obj}'"); return True
    except Exception as e: st.error(f"HF Save Error: {e}"); st.error(traceback.format_exc()); return False

def load_hf_model_artifacts(load_path: str) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, int]], Optional[pd.DataFrame]]:
    """Loads HF model artifacts from the specified user path."""
    st.info(f"HF Load: Attempting load from '{load_path}'...")
    load_path_obj = Path(load_path);
    if not load_path_obj.is_dir(): st.error(f"HF Load Error: Dir not found: '{load_path_obj}'"); return None, None, None, None
    model, tokenizer, label_map, rules_df = None, None, None, None
    try:
        label_map_path = load_path_obj / "label_map.json"; rules_path = load_path_obj / "rules.csv"
        config_path = load_path_obj / "config.json"; tok_config_path = load_path_obj / "tokenizer_config.json"
        if not all(p.exists() for p in [config_path, tok_config_path, label_map_path]): st.error("HF Load Error: Essential file(s) missing."); return None, None, None, None
        with open(label_map_path, 'r') as f: label_map = json.load(f)
        required_rules_cols = ['Label', 'Keywords', 'Confidence Threshold']
        if not rules_path.exists():
            st.warning("HF Load: Rules file not found. Creating defaults.")
            rules_df = pd.DataFrame({'Label': list(label_map.keys()), 'Keywords': 'N/A (File not found)', 'Confidence Threshold': 0.5})
        else:
             rules_df = pd.read_csv(rules_path)
             if 'Label' not in rules_df.columns: raise ValueError("Rules file missing 'Label'.")
             if 'Keywords' not in rules_df.columns: rules_df['Keywords'] = "N/A"
             if 'Confidence Threshold' not in rules_df.columns: rules_df['Confidence Threshold'] = 0.5
             rules_df['Confidence Threshold'] = pd.to_numeric(rules_df['Confidence Threshold'], errors='coerce').fillna(0.5).clip(0.05, 0.95)
             rules_df['Keywords'] = rules_df['Keywords'].fillna('N/A')
             # Select only standard columns, ensure order if needed
             rules_df = rules_df[required_rules_cols]
             if set(rules_df['Label']) != set(label_map.keys()): st.warning("HF Load Warn: Rules labels mismatch map.")

        with st.spinner("HF Load: Loading model..."): model = AutoModelForSequenceClassification.from_pretrained(load_path_obj, ignore_mismatched_sizes=True)
        if model.config.num_labels != len(label_map): model.config.num_labels = len(label_map)
        with st.spinner("HF Load: Loading tokenizer..."): tokenizer = AutoTokenizer.from_pretrained(load_path_obj)
        st.success(f"✅ HF Model loaded from '{load_path_obj}'"); return model, tokenizer, label_map, rules_df
    except Exception as e: st.error(f"HF Load Error: {e}"); st.error(traceback.format_exc()); return None, None, None, None


# --- Rule Extraction using Chi-Squared (Corrected) ---
def extract_hf_rules(
    full_train_texts: List[str],
    full_train_labels_encoded: List[np.ndarray], # Expecting full encoded labels here
    label_map: Dict[str, int]
    ) -> pd.DataFrame:
    """Extracts keyword associations using Chi-Squared."""
    st.info("HF Rules: Extracting keyword associations (Chi-Squared)...")
    required_columns = ['Label', 'Keywords', 'Confidence Threshold'] # Define standard cols
    default_entry = {'Keywords': "N/A", 'Confidence Threshold': 0.5} # Default values

    if not full_train_texts or full_train_labels_encoded is None or not label_map:
        st.warning("HF Rules: Cannot extract - missing data/map.")
        return pd.DataFrame(columns=required_columns)

    reverse_label_map = {v: k for k, v in label_map.items()}
    rules = []
    num_labels = len(label_map)
    train_labels_array = np.array(full_train_labels_encoded) # Ensure it's a numpy array

    if train_labels_array.shape[0] != len(full_train_texts) or train_labels_array.shape[1] != num_labels:
        st.error(f"HF Rules: Shape mismatch (Texts: {len(full_train_texts)}, Labels: {train_labels_array.shape}).")
        return pd.DataFrame(columns=required_columns)

    # --- Vectorization ---
    try:
        with st.spinner("HF Rules: Vectorizing text..."):
            vectorizer = CountVectorizer(max_features=1500, stop_words='english', binary=False, min_df=3)
            X = vectorizer.fit_transform(full_train_texts)
            feature_names = vectorizer.get_feature_names_out()
            if X.shape[0] == 0 or X.shape[1] == 0:
                 st.warning("HF Rules: Vectorization resulted in an empty matrix.")
                 # Return default DF structure for all labels
                 return pd.DataFrame([{ 'Label': lbl, **default_entry, 'Keywords': 'N/A (Empty Matrix)'} for lbl in label_map.keys()])
    except Exception as e:
        st.warning(f"HF Rules: Vectorization error: {e}")
        return pd.DataFrame([{ 'Label': lbl, **default_entry, 'Keywords': 'N/A (Vectorization Error)'} for lbl in label_map.keys()])

    # --- Chi-Squared Calculation ---
    st.info("HF Rules: Calculating Chi-Squared scores...")
    with st.spinner("Analyzing feature relevance..."):
        for label_idx in range(num_labels):
            label_name = reverse_label_map.get(label_idx, f"Unknown_{label_idx}")
            y = train_labels_array[:, label_idx]

            if np.std(y) < 1e-9: keywords = "N/A (No variance)"
            else:
                try:
                    chi2_scores, _ = chi2(X, y)
                    valid_scores = ~np.isnan(chi2_scores) # Mask for valid scores
                    feature_scores = sorted(
                        zip(feature_names[valid_scores], chi2_scores[valid_scores]),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    top_features = feature_scores[:7] # Get top 7 valid features
                    keywords = ', '.join([word for word, score in top_features]) if top_features else "N/A (No features found)"
                except Exception as e_chi2:
                    st.warning(f"HF Rules: Chi2 Error for '{label_name}': {e_chi2}"); keywords = "N/A (Calculation Error)"

            rules.append({'Label': label_name, **default_entry, 'Keywords': keywords})

    st.success("HF Rules: Extraction finished.")
    return pd.DataFrame(rules)


# --- HF Classification (Uses Edited Keywords) ---

def classify_texts_with_hf(texts: List[str], model: Any, tokenizer: Any, label_map: Dict[str, int], rules_df: Optional[pd.DataFrame]) -> List[List[str]]:
    """Classifies texts using HF model and applies keyword overrides from the EDITED 'Keywords' column."""
    if not texts: return []
    if model is None or tokenizer is None or label_map is None: st.error("HF Classify: Missing components."); return [["Error"] for _ in texts]

    thresholds, keyword_override_map = {}, {}; default_threshold = config.DEFAULT_HF_THRESHOLD

    if rules_df is None or rules_df.empty or 'Label' not in rules_df.columns or 'Confidence Threshold' not in rules_df.columns or 'Keywords' not in rules_df.columns:
         st.info(f"HF Classify: Rules missing/invalid. Using defaults."); thresholds = {l: default_threshold for l in label_map.keys()}
    else:
        st.info(f"HF Classify: Using thresholds & keywords from rules.");
        try:
            rules_df['Confidence Threshold'] = pd.to_numeric(rules_df['Confidence Threshold'], errors='coerce').fillna(default_threshold).clip(0.05, 0.95); thresholds = dict(zip(rules_df['Label'], rules_df['Confidence Threshold']))
            for l in label_map.keys():
                if l not in thresholds: thresholds[l] = default_threshold
            rules_df['Keywords'] = rules_df['Keywords'].fillna('')
            for _, r in rules_df.iterrows():
                 kwl = [kw.strip().lower() for kw in str(r['Keywords']).split(',') if kw.strip() and "N/A" not in kw]
                 if kwl: keyword_override_map[r['Label']] = kwl
        except Exception as e: st.error(f"HF Classify: Rules Error: {e}. Using defaults."); thresholds = {l: default_threshold for l in label_map.keys()}; keyword_override_map = {}

    st.info("HF Classify: Starting..."); reverse_map={v:k for k,v in label_map.items()}; n_lbl=len(label_map); results=[]
    bs=16; pb=st.progress(0); pt=st.empty(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); st.write(f"HF Classify on: {dev}")
    model.to(dev); model.eval(); texts_cl=[str(t) if pd.notna(t) else "" for t in texts]; total_txt=len(texts_cl)

    for i in range(0, total_txt, bs):
        batch_txt = texts_cl[i:min(i+bs, total_txt)]
        if not batch_txt:
            continue
        cur_bs = len(batch_txt)
        pt.text(f"HF Classifying: {i+1}-{i+cur_bs}/{total_txt}...")
        try:
            inputs = tokenizer(batch_txt, truncation=True, padding=True, return_tensors="pt", max_length=512)
            inputs = {k:v.to(dev) for k,v in inputs.items()}
            with torch.no_grad(): 
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
            batch_final = []
            for j in range(cur_bs):
                p_row = probs[j]
                txt_low = batch_txt[j].lower()
                init_lbls = set()
                # Determine initial labels based on thresholds
                for l_idx in range(n_lbl):
                    l_name = reverse_map.get(l_idx, f"Unknown_{l_idx}")
                    thr = thresholds.get(l_name, default_threshold)
                    if p_row[l_idx] >= thr:
                        init_lbls.add(l_name)
                # Apply keyword overrides
                final_lbls = init_lbls.copy()
                if keyword_override_map:
                    for l_name, kws in keyword_override_map.items():
                        if l_name in final_lbls:
                            continue
                        found = False
                        for kw in kws:
                            if kw:
                                try:
                                    if re.search(r'\b'+re.escape(kw)+r'\b', txt_low):
                                        found = True
                                        break
                                except re.error:
                                    if kw in txt_low:
                                        found = True
                                        break
                        if found:
                            final_lbls.add(l_name)
                # Fallback to highest probability if no labels
                if not final_lbls and len(p_row) > 0:
                    h_idx = np.argmax(p_row)
                    fb_lbl = reverse_map.get(h_idx, "Unknown")
                    if p_row[h_idx] > 0.1:
                        final_lbls.add(fb_lbl)
                batch_final.append(list(final_lbls))
            results.extend(batch_final)
        except Exception as e:
            st.error(f"HF Classify Err (B {i}): {e}")
            st.error(traceback.format_exc())
            results.extend([["Error"] for _ in range(cur_bs)])
        prog = min(1.0, (i + cur_bs) / total_txt) if total_txt > 0 else 1.0
        pb.progress(prog)
    pt.text("HF Classification done!")
    st.success("HF Classification finished.")
    st.success("View results in the Results tab.")
    return results