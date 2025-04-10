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
class StProgressCallback(TrainerCallback):
    """Callback to update Streamlit progress bar during HF training."""
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
def compute_metrics(p: EvalPrediction):
    """
    Computes multi-label classification metrics for Hugging Face Trainer.

    Args:
        p: An EvalPrediction object containing predictions and label_ids.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics.
    """
    # Extract logits and labels
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    # Apply sigmoid and threshold to get binary predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    threshold = 0.5
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    # Ensure integer types for metric calculation
    y_true = labels.astype(int)
    y_pred = y_pred.astype(int)

    # Calculate metrics
    subset_accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    micro_precision, micro_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    return {
        'subset_accuracy': subset_accuracy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall
    }

# --- Model Training ---
def train_hf_model(
    all_train_texts: List[str],
    all_train_labels_list: List[List[str]],
    model_choice: str,
    num_epochs: int,
    validation_split_ratio: float = config.DEFAULT_VALIDATION_SPLIT
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, int]], pd.DataFrame]:
    """
    Trains the Hugging Face classification model with validation split.

    Orchestrates the entire training process including data preparation,
    model loading, training, and rule extraction.

    Args:
        all_train_texts: List of all training text examples.
        all_train_labels_list: List of lists, where each inner list contains the string labels for a text.
        model_choice: Name of the Hugging Face model to use (e.g., 'bert-base-uncased').
        num_epochs: Number of training epochs.
        validation_split_ratio: Fraction of data to use for validation.

    Returns:
        A tuple containing:
        - Trained model object (or None on error).
        - Tokenizer object (or None on error).
        - Label map (dictionary mapping label names to indices) (or None on error).
        - DataFrame containing extracted rules (or empty DataFrame on error).
    """
    st.info(f"Starting HF model training: {model_choice} for {num_epochs} epochs ({validation_split_ratio*100:.1f}% validation)...")
    model, tokenizer, label_map, rules_df = None, None, None, pd.DataFrame()

    # --- 1. Process Labels ---
    label_map, all_encoded_labels, num_labels = _process_labels(all_train_labels_list)
    if label_map is None:
        return None, None, None, pd.DataFrame() # Error already logged in helper

    # --- 2. Split Data ---
    split_data = _split_data(all_train_texts, all_encoded_labels, validation_split_ratio)
    if split_data is None:
        return None, None, None, pd.DataFrame() # Error already logged in helper
    train_texts, val_texts, train_labels_encoded, val_labels_encoded = split_data

    # --- Progress Bar Setup ---
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    status_text = st.empty()
    status_text.text("HF Train: Initializing...")

    try:
        # --- 3. Load Model & Tokenizer ---
        tokenizer, model = _load_model_and_tokenizer(model_choice, num_labels, progress_bar, status_text)
        if tokenizer is None or model is None:
            raise ValueError("Failed to load model or tokenizer.") # Error logged in helper

        # --- 4. Tokenize Data ---
        train_encodings, val_encodings = _tokenize_data(
            tokenizer, train_texts, val_texts, progress_bar, status_text
        )
        if train_encodings is None or val_encodings is None:
             raise ValueError("Failed to tokenize data.") # Error logged in helper

        # --- 5. Create Datasets ---
        train_dataset, eval_dataset = _create_datasets(
            train_encodings, train_labels_encoded, val_encodings, val_labels_encoded, progress_bar, status_text
        )
        if train_dataset is None or eval_dataset is None:
            raise ValueError("Failed to create datasets.") # Error logged in helper

        # --- 6. Setup Training ---
        training_args, progress_callback, total_steps = _setup_training_arguments(
            model_choice, num_epochs, len(train_dataset), progress_bar, progress_text, status_text
        )
        if training_args is None:
             raise ValueError("Failed to set up training arguments.") # Error logged in helper

        # --- 7. Initialize Trainer ---
        trainer = _initialize_trainer(
            model, training_args, train_dataset, eval_dataset, progress_callback
        )
        progress_bar.progress(0.6, text="HF Train: Trainer Initialized.")

        # --- 8. Train Model ---
        status_text.text("HF Train: Starting training loop...")
        train_result = trainer.train()
        metrics = train_result.metrics
        st.write("Training completed. Final Metrics:")
        st.json(metrics)
        progress_bar.progress(0.95, text="HF Train: Training Finished.")

        # --- 9. Extract Rules ---
        status_text.text("HF Train: Extracting keyword rules...")
        # Use the original full training data (before splitting) for rule extraction
        rules_df = extract_hf_rules(all_train_texts, all_encoded_labels, label_map)
        progress_bar.progress(1.0)
        status_text.text("HF Training Pipeline Done!")
        progress_text.text("") # Clear progress text

        # --- 10. Finalize and Return ---
        best_model = trainer.model
        best_model.cpu() # Move model to CPU before returning
        return best_model, tokenizer, label_map, rules_df

    except Exception as e:
        st.error(f"HF Train Error: {e}")
        st.error(traceback.format_exc())
        status_text.text("HF Train Failed.")
        progress_text.text("")
        progress_bar.progress(1.0) # Ensure progress bar completes on error
        return None, None, None, pd.DataFrame()


# --- Helper Functions for train_hf_model ---

def _process_labels(all_train_labels_list: List[List[str]]) -> Tuple[Optional[Dict[str, int]], Optional[List[np.ndarray]], Optional[int]]:
    """Processes raw labels into a map and encoded format."""
    st.write("HF Train Helper: Processing labels...")
    try:
        all_labels_set = set(label for sublist in all_train_labels_list for label in sublist if label)
        if not all_labels_set:
            st.error("HF Train Helper: No valid labels found in the training data.")
            return None, None, None
        unique_labels = sorted(list(all_labels_set))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        num_labels = len(unique_labels)
        st.write(f"HF Train Helper: Found {num_labels} unique labels.")

        def encode_labels(labels: List[str]) -> np.ndarray:
            encoded = np.zeros(num_labels, dtype=np.float32)
            for label in labels:
                if label in label_map:
                    encoded[label_map[label]] = 1.0
            return encoded

        all_encoded_labels = [encode_labels(labels) for labels in all_train_labels_list]
        return label_map, all_encoded_labels, num_labels
    except Exception as e:
        st.error(f"HF Train Helper: Error processing labels: {e}")
        return None, None, None


def _split_data(
    all_texts: List[str],
    all_encoded_labels: List[np.ndarray],
    validation_split_ratio: float
) -> Optional[Tuple[List[str], List[str], List[np.ndarray], List[np.ndarray]]]:
    """Splits data into training and validation sets."""
    st.write("HF Train Helper: Splitting data...")
    try:
        train_texts, val_texts, train_labels_encoded, val_labels_encoded = train_test_split(
            all_texts, all_encoded_labels, test_size=validation_split_ratio, random_state=42
        )
        if not train_texts or not val_texts:
            raise ValueError("Created an empty training or validation set after split. Check data or split ratio.")
        st.success(f"HF Train Helper: Data split complete ({len(train_texts)} train, {len(val_texts)} validation).")
        return train_texts, val_texts, train_labels_encoded, val_labels_encoded
    except ValueError as e:
        if "test_size" in str(e) or "cannot be larger" in str(e) or "should be between" in str(e):
             st.error(f"HF Train Helper: Invalid validation split ratio ({validation_split_ratio}). Must be > 0 and < 1.")
        elif "Found input variables with inconsistent numbers of samples" in str(e):
             st.error(f"HF Train Helper: Mismatch between number of texts ({len(all_texts)}) and labels ({len(all_encoded_labels)}). Cannot split.")
        else:
             st.error(f"HF Train Helper: Error splitting data: {e}")
        return None
    except Exception as e:
        st.error(f"HF Train Helper: Unexpected error during data splitting: {e}")
        return None


def _load_model_and_tokenizer(
    model_choice: str, num_labels: int, progress_bar: Any, status_text: Any
) -> Tuple[Optional[Any], Optional[Any]]:
    """Loads the Hugging Face tokenizer and model."""
    tokenizer, model = None, None
    try:
        status_text.text(f"HF Train Helper: Loading tokenizer '{model_choice}'...")
        progress_bar.progress(0.1)
        with st.spinner(f'Loading tokenizer {model_choice}...'):
            tokenizer = AutoTokenizer.from_pretrained(model_choice)

        status_text.text(f"HF Train Helper: Loading model '{model_choice}'...")
        progress_bar.progress(0.2)
        with st.spinner(f'Loading model {model_choice} for {num_labels} labels...'):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_choice,
                num_labels=num_labels,
                problem_type="multi_label_classification",
                # Allow loading weights even if classifier head size differs (will be resized)
                ignore_mismatched_sizes=True
            )
            # Ensure the model's config reflects the correct number of labels after loading
            if model.config.num_labels != num_labels:
                st.warning(f"HF Train Helper: Model config label count ({model.config.num_labels}) differs from required ({num_labels}). Adjusting config.")
                model.config.num_labels = num_labels
        st.success(f"HF Train Helper: Model and tokenizer '{model_choice}' loaded.")
        return tokenizer, model
    except OSError as e:
        st.error(f"HF Train Helper: Could not find model/tokenizer '{model_choice}'. Check name or internet connection. Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"HF Train Helper: Error loading model/tokenizer: {e}")
        return None, None


def _tokenize_data(
    tokenizer: Any, train_texts: List[str], val_texts: List[str], progress_bar: Any, status_text: Any
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Tokenizes the training and validation text data."""
    status_text.text("HF Train Helper: Tokenizing text data...")
    progress_bar.progress(0.4)
    try:
        with st.spinner('Tokenizing training and validation sets...'):
            # Ensure all inputs are strings, replace None/NaN with empty string
            train_texts_clean = [str(t) if pd.notna(t) else "" for t in train_texts]
            val_texts_clean = [str(t) if pd.notna(t) else "" for t in val_texts]

            train_encodings = tokenizer(train_texts_clean, truncation=True, padding=True, max_length=512)
            val_encodings = tokenizer(val_texts_clean, truncation=True, padding=True, max_length=512)
        st.success("HF Train Helper: Tokenization complete.")
        return train_encodings, val_encodings
    except Exception as e:
        st.error(f"HF Train Helper: Error during tokenization: {e}")
        return None, None


class TextDataset(torch.utils.data.Dataset):
    """Simple PyTorch Dataset for Hugging Face text classification."""
    def __init__(self, encodings: Dict[str, Any], labels: List[np.ndarray]):
        self.encodings = encodings
        self.labels = labels
        # Basic validation
        if not self.encodings or 'input_ids' not in self.encodings:
            raise ValueError("Encodings dictionary is invalid or missing 'input_ids'.")
        if len(self.encodings['input_ids']) != len(self.labels):
            raise ValueError(f"Mismatch between number of encodings ({len(self.encodings['input_ids'])}) and labels ({len(self.labels)}).")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Ensure all encoding values are converted to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Ensure labels are float tensors as expected by HF for multi-label
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self) -> int:
        # Safely get length from 'input_ids'
        return len(self.encodings.get('input_ids', []))


def _create_datasets(
    train_encodings: Dict[str, Any], train_labels_encoded: List[np.ndarray],
    val_encodings: Dict[str, Any], val_labels_encoded: List[np.ndarray],
    progress_bar: Any, status_text: Any
) -> Tuple[Optional[TextDataset], Optional[TextDataset]]:
    """Creates PyTorch datasets from tokenized data and labels."""
    status_text.text("HF Train Helper: Creating PyTorch datasets...")
    progress_bar.progress(0.5)
    try:
        train_dataset = TextDataset(train_encodings, train_labels_encoded)
        eval_dataset = TextDataset(val_encodings, val_labels_encoded)
        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            raise ValueError("Created an empty training or evaluation dataset after encoding.")
        st.success("HF Train Helper: Datasets created successfully.")
        return train_dataset, eval_dataset
    except ValueError as e:
        st.error(f"HF Train Helper: Error creating dataset: {e}")
        return None, None
    except Exception as e:
        st.error(f"HF Train Helper: Unexpected error creating dataset: {e}")
        return None, None


def _setup_training_arguments(
    model_choice: str, num_epochs: int, train_dataset_len: int,
    progress_bar: Any, progress_text: Any, status_text: Any
) -> Tuple[Optional[TrainingArguments], Optional[StProgressCallback], Optional[int]]:
    """Sets up Hugging Face TrainingArguments and the progress callback."""
    status_text.text("HF Train Helper: Configuring training arguments...")
    progress_bar.progress(0.55)
    try:
        # --- Determine Batch Size & Gradient Accumulation ---
        # Heuristic: Smaller batch size for larger models or if no GPU
        is_large_model = any(m in model_choice for m in ["large", "roberta-base", "deberta-v3-large"]) # Add more large models if needed
        has_gpu = torch.cuda.is_available()
        base_batch_size = 4 if is_large_model else 8
        effective_batch_size = base_batch_size // (1 if has_gpu else 2) # Halve if no GPU
        train_batch_size = max(1, effective_batch_size)
        eval_batch_size = train_batch_size * 2 # Often can use larger batch for eval

        # Aim for a virtual batch size of ~16 if possible
        gradient_accumulation_steps = max(1, 16 // train_batch_size)

        st.write(f"HF Train Helper: Using Train Batch Size: {train_batch_size}, Grad Accumulation: {gradient_accumulation_steps} (GPU: {has_gpu})")

        # --- Calculate Total Steps ---
        world_size = int(os.environ.get("WORLD_SIZE", 1)) # For distributed training awareness
        updates_per_epoch = max(1, train_dataset_len // (train_batch_size * gradient_accumulation_steps * world_size)) + \
                            (1 if train_dataset_len % (train_batch_size * gradient_accumulation_steps * world_size) != 0 else 0)
        total_training_steps = max(1, updates_per_epoch * num_epochs)
        st.write(f"HF Train Helper: Estimated total training steps: {total_training_steps}")

        # --- Define Training Arguments ---
        training_args = TrainingArguments(
            output_dir='./results_hf_training',          # Directory for checkpoints and logs
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,                             # Linear warmup over 10% of training
            weight_decay=0.01,                            # Regularization
            logging_dir='./logs_hf_training',             # Directory for TensorBoard logs
            logging_strategy="epoch",                     # Log metrics every epoch
            evaluation_strategy="epoch",                  # Evaluate every epoch
            save_strategy="epoch",                        # Save checkpoint every epoch
            save_total_limit=2,                           # Keep only the last 2 checkpoints
            load_best_model_at_end=True,                  # Load the best model found during training
            metric_for_best_model="micro_f1",             # Metric to determine the "best" model
            greater_is_better=True,                       # Higher micro_f1 is better
            report_to="none",                             # Disable reporting to external services (like W&B)
            fp16=has_gpu,                                 # Enable mixed precision training if GPU is available
            dataloader_pin_memory=has_gpu,                # Pin memory for faster data transfer if GPU
            # Use multiple workers for data loading if GPU available and CPU cores allow
            dataloader_num_workers=min(4, os.cpu_count() // 2) if has_gpu else 0
        )

        # --- Setup Progress Callback ---
        progress_callback = StProgressCallback(progress_bar, progress_text, status_text, total_training_steps)

        st.success("HF Train Helper: Training arguments configured.")
        return training_args, progress_callback, total_training_steps

    except Exception as e:
        st.error(f"HF Train Helper: Error setting up training arguments: {e}")
        return None, None, None


def _initialize_trainer(
    model: Any, training_args: TrainingArguments, train_dataset: TextDataset,
    eval_dataset: TextDataset, progress_callback: StProgressCallback
) -> Trainer:
    """Initializes the Hugging Face Trainer."""
    st.write("HF Train Helper: Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics, # Function to calculate metrics during evaluation
        callbacks=[progress_callback]     # Custom callback for Streamlit progress updates
    )
    st.success("HF Train Helper: Trainer initialized.")
    return trainer


# --- Model Saving/Loading ---
def save_hf_model_artifacts(model: Any, tokenizer: Any, label_map: Dict[str, int], rules_df: pd.DataFrame, save_path: str):
    """
    Saves the HF model, tokenizer, label map, and rules DataFrame to a specified directory.

    Args:
        model: The trained Hugging Face model object.
        tokenizer: The Hugging Face tokenizer object.
        label_map: Dictionary mapping label names to integer indices.
        rules_df: DataFrame containing rules ('Label', 'Keywords', 'Confidence Threshold').
        save_path: The directory path where artifacts should be saved.

    Returns:
        True if saving was successful, False otherwise.
    """
    if model is None or tokenizer is None or label_map is None or rules_df is None:
        st.error("HF Save Error: One or more required components (model, tokenizer, label_map, rules_df) are missing. Cannot save.")
        return False

    try:
        save_path_obj = Path(save_path)
        st.info(f"HF Save: Saving model artifacts to '{save_path_obj}'...")
        save_path_obj.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

        # Save model and tokenizer
        model.save_pretrained(save_path_obj)
        tokenizer.save_pretrained(save_path_obj)

        # Save label map
        with open(save_path_obj / "label_map.json", 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=4)

        # --- Prepare and Save Rules DataFrame ---
        st.write("HF Save: Preparing rules DataFrame for saving...")
        rules_df_to_save = rules_df.copy()
        required_cols = ['Label', 'Keywords', 'Confidence Threshold']
        default_threshold = config.DEFAULT_HF_THRESHOLD # Use config default

        # Ensure required columns exist, adding defaults if necessary
        for col in required_cols:
            if col not in rules_df_to_save.columns:
                st.warning(f"HF Save: Rules DataFrame missing column '{col}'. Adding default values.")
                if col == 'Keywords':
                    rules_df_to_save[col] = 'N/A (Auto-added)'
                elif col == 'Confidence Threshold':
                    rules_df_to_save[col] = default_threshold
                else: # Should only be 'Label', which is critical
                     st.error(f"HF Save Error: Critical column '{col}' missing in rules_df. Cannot reliably save rules.")
                     # Optionally save anyway with placeholder labels, or return False
                     # For now, let's add placeholder labels if possible, but warn heavily
                     if 'Label' not in rules_df_to_save.columns and label_map:
                         rules_df_to_save['Label'] = list(label_map.keys())[:len(rules_df_to_save)] # Risky if lengths mismatch
                     else:
                         return False # Cannot proceed without labels

        # Select only the required columns in the standard order
        rules_df_to_save = rules_df_to_save[required_cols]

        # Clean and validate 'Confidence Threshold'
        rules_df_to_save['Confidence Threshold'] = pd.to_numeric(rules_df_to_save['Confidence Threshold'], errors='coerce')
        rules_df_to_save['Confidence Threshold'] = rules_df_to_save['Confidence Threshold'].fillna(default_threshold).clip(0.05, 0.95)

        # Clean 'Keywords' (ensure it's string, fill NaNs)
        rules_df_to_save['Keywords'] = rules_df_to_save['Keywords'].fillna('N/A').astype(str)

        # Save the cleaned rules DataFrame
        rules_path = save_path_obj / "rules.csv"
        rules_df_to_save.to_csv(rules_path, index=False, encoding='utf-8')
        st.write(f"HF Save: Rules saved to '{rules_path}'.")

        st.success(f"✅ HF Model artifacts successfully saved to '{save_path_obj}'")
        return True

    except Exception as e:
        st.error(f"HF Save Error: Failed to save model artifacts: {e}")
        st.error(traceback.format_exc())
        return False


def load_hf_model_artifacts(load_path: str) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, int]], Optional[pd.DataFrame]]:
    """
    Loads HF model artifacts (model, tokenizer, label map, rules) from a specified directory.

    Args:
        load_path: The directory path from which to load artifacts.

    Returns:
        A tuple containing:
        - Loaded model object (or None on error).
        - Loaded tokenizer object (or None on error).
        - Loaded label map (dictionary) (or None on error).
        - Loaded rules DataFrame (or None on error).
    """
    st.info(f"HF Load: Attempting to load model artifacts from '{load_path}'...")
    load_path_obj = Path(load_path)
    default_threshold = config.DEFAULT_HF_THRESHOLD

    if not load_path_obj.is_dir():
        st.error(f"HF Load Error: Directory not found: '{load_path_obj}'")
        return None, None, None, None

    model, tokenizer, label_map, rules_df = None, None, None, None

    try:
        # --- Check for essential files ---
        model_config_path = load_path_obj / "config.json"
        tokenizer_config_path = load_path_obj / "tokenizer_config.json"
        label_map_path = load_path_obj / "label_map.json"
        rules_path = load_path_obj / "rules.csv" # Rules file is optional but preferred

        required_files = [model_config_path, tokenizer_config_path, label_map_path]
        missing_files = [p for p in required_files if not p.exists()]
        if missing_files:
            st.error(f"HF Load Error: Essential file(s) missing in '{load_path_obj}': {', '.join(p.name for p in missing_files)}")
            return None, None, None, None

        # --- Load Label Map ---
        st.write("HF Load: Loading label map...")
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        if not isinstance(label_map, dict) or not label_map:
             raise ValueError("Loaded label_map.json is invalid or empty.")
        st.write(f"HF Load: Label map loaded ({len(label_map)} labels).")

        # --- Load Rules DataFrame (with fallback) ---
        required_rules_cols = ['Label', 'Keywords', 'Confidence Threshold']
        if rules_path.exists():
            st.write("HF Load: Loading rules file...")
            try:
                rules_df = pd.read_csv(rules_path, encoding='utf-8')
                # Validate and clean loaded rules
                if 'Label' not in rules_df.columns:
                    raise ValueError("Rules file is missing the critical 'Label' column.")

                # Add missing optional columns with defaults
                if 'Keywords' not in rules_df.columns:
                    st.warning("HF Load: Rules file missing 'Keywords' column. Adding default 'N/A'.")
                    rules_df['Keywords'] = 'N/A'
                if 'Confidence Threshold' not in rules_df.columns:
                    st.warning(f"HF Load: Rules file missing 'Confidence Threshold' column. Adding default {default_threshold}.")
                    rules_df['Confidence Threshold'] = default_threshold

                # Select and order standard columns
                rules_df = rules_df[required_rules_cols]

                # Clean data types and values
                rules_df['Confidence Threshold'] = pd.to_numeric(rules_df['Confidence Threshold'], errors='coerce').fillna(default_threshold).clip(0.05, 0.95)
                rules_df['Keywords'] = rules_df['Keywords'].fillna('N/A').astype(str)
                rules_df['Label'] = rules_df['Label'].astype(str) # Ensure labels are strings

                # Check for consistency between rules labels and label map
                rules_labels = set(rules_df['Label'])
                map_labels = set(label_map.keys())
                if rules_labels != map_labels:
                    st.warning(f"HF Load Warning: Labels in rules.csv do not exactly match labels in label_map.json.")
                    st.warning(f"  Labels only in rules: {rules_labels - map_labels}")
                    st.warning(f"  Labels only in map: {map_labels - rules_labels}")
                    # Optional: Could try to reconcile, e.g., keep only rules for labels in map
                    rules_df = rules_df[rules_df['Label'].isin(map_labels)]
                    st.warning("  Keeping only rules for labels present in the label map.")

                st.write(f"HF Load: Rules DataFrame loaded and validated ({len(rules_df)} rules).")

            except Exception as e_rules:
                st.error(f"HF Load Error: Failed to load or process '{rules_path}': {e_rules}. Proceeding without rules.")
                rules_df = None # Fallback to no rules if loading/processing fails
        else:
            st.warning(f"HF Load: Rules file ('{rules_path.name}') not found in '{load_path_obj}'. Model will use default thresholds and no keyword overrides.")
            # Create a default DataFrame structure if needed downstream, matching the label map
            rules_df = pd.DataFrame({
                'Label': list(label_map.keys()),
                'Keywords': 'N/A (File not found)',
                'Confidence Threshold': default_threshold
            })
            st.write("HF Load: Created default rules structure based on label map.")


        # --- Load Model ---
        st.write("HF Load: Loading model...")
        with st.spinner("Loading model weights..."):
            model = AutoModelForSequenceClassification.from_pretrained(
                load_path_obj,
                # Allow loading weights even if classifier head size differs from saved config
                # This is important if the model was fine-tuned on a different number of labels previously
                ignore_mismatched_sizes=True
            )
        # Crucially, ensure the loaded model's config matches the *loaded label map* size
        num_labels_from_map = len(label_map)
        if model.config.num_labels != num_labels_from_map:
            st.warning(f"HF Load: Model's internal config label count ({model.config.num_labels}) differs from loaded label map ({num_labels_from_map}). Adjusting model config to match label map.")
            model.config.num_labels = num_labels_from_map
            # Potentially resize the classifier layer if necessary, though ignore_mismatched_sizes often handles this implicitly
            # model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels_from_map) # Example if manual resize needed
        st.write("HF Load: Model loaded.")

        # --- Load Tokenizer ---
        st.write("HF Load: Loading tokenizer...")
        with st.spinner("Loading tokenizer configuration..."):
            tokenizer = AutoTokenizer.from_pretrained(load_path_obj)
        st.write("HF Load: Tokenizer loaded.")

        st.success(f"✅ HF Model artifacts successfully loaded from '{load_path_obj}'")
        return model, tokenizer, label_map, rules_df

    except json.JSONDecodeError as e_json:
        st.error(f"HF Load Error: Failed to decode JSON file (likely label_map.json): {e_json}")
        return None, None, None, None
    except ValueError as e_val:
        st.error(f"HF Load Error: Data validation issue: {e_val}")
        return None, None, None, None
    except OSError as e_os:
         st.error(f"HF Load Error: File system error (e.g., permission denied, file not found during model/tokenizer load): {e_os}")
         return None, None, None, None
    except Exception as e:
        st.error(f"HF Load Error: An unexpected error occurred: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None


# --- Rule Extraction using Chi-Squared ---
def extract_hf_rules(
    full_train_texts: List[str],
    full_train_labels_encoded: List[np.ndarray], # Expects encoded labels for the *entire* training set
    label_map: Dict[str, int]
    ) -> pd.DataFrame:
    """
    Extracts potential keyword associations for each label using Chi-Squared feature selection.

    This function analyzes the provided training data to find words that are statistically
    significant for each label, providing a starting point for rule-based overrides.

    Args:
        full_train_texts: List of all text examples from the *entire* training set.
        full_train_labels_encoded: List of multi-hot encoded labels (NumPy arrays)
                                   corresponding to `full_train_texts`.
        label_map: Dictionary mapping label names (str) to their integer indices (int).

    Returns:
        A pandas DataFrame with columns ['Label', 'Keywords', 'Confidence Threshold'].
        'Keywords' contains comma-separated top terms based on Chi-Squared scores.
        'Confidence Threshold' is initialized to a default value from config.
        Returns an empty or partially filled DataFrame on error or if no keywords found.
    """
    st.info("HF Rules Extractor: Starting keyword association analysis (Chi-Squared)...")
    required_columns = ['Label', 'Keywords', 'Confidence Threshold']
    default_threshold = config.DEFAULT_HF_THRESHOLD
    default_keywords = "N/A" # Default keyword string

    # --- Input Validation ---
    if not full_train_texts or full_train_labels_encoded is None or not label_map:
        st.warning("HF Rules Extractor: Cannot extract rules - missing training texts, encoded labels, or label map.")
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=required_columns)

    if not isinstance(full_train_texts, list) or not isinstance(full_train_labels_encoded, list):
         st.error("HF Rules Extractor: Texts and encoded labels must be lists.")
         return pd.DataFrame(columns=required_columns)

    if len(full_train_texts) != len(full_train_labels_encoded):
         st.error(f"HF Rules Extractor: Mismatch between number of texts ({len(full_train_texts)}) and labels ({len(full_train_labels_encoded)}).")
         return pd.DataFrame(columns=required_columns)

    try:
        # Ensure encoded labels are a NumPy array of integers for chi2
        train_labels_array = np.array(full_train_labels_encoded, dtype=int)
    except ValueError as e:
        st.error(f"HF Rules Extractor: Could not convert encoded labels to NumPy array (check for consistent shapes?): {e}")
        return pd.DataFrame(columns=required_columns)

    num_labels = len(label_map)
    num_texts = len(full_train_texts)

    if train_labels_array.shape != (num_texts, num_labels):
        st.error(f"HF Rules Extractor: Shape mismatch. Texts: {num_texts}, Encoded Labels Shape: {train_labels_array.shape}, Expected Labels Dim: {num_labels}.")
        return pd.DataFrame(columns=required_columns)

    # --- Create Reverse Label Map ---
    reverse_label_map = {v: k for k, v in label_map.items()}
    all_labels_in_map = list(label_map.keys()) # For creating default entries later

    # --- Vectorization ---
    st.write("HF Rules Extractor: Vectorizing text data...")
    try:
        with st.spinner("Vectorizing text using CountVectorizer..."):
            # Use CountVectorizer: good for interpretability with Chi-Squared
            vectorizer = CountVectorizer(
                max_features=1500,      # Limit vocabulary size
                stop_words='english',   # Remove common English stop words
                binary=False,           # Use term frequency (counts)
                min_df=3                # Ignore terms appearing in < 3 documents
            )
            # Ensure texts are strings
            cleaned_texts = [str(text) if pd.notna(text) else "" for text in full_train_texts]
            X = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()

            if X.shape[0] == 0 or X.shape[1] == 0:
                 st.warning("HF Rules Extractor: Vectorization resulted in an empty feature matrix (X). Check input text or vectorizer parameters.")
                 # Return default DF structure for all labels
                 return pd.DataFrame([{
                     'Label': lbl,
                     'Keywords': f'{default_keywords} (Empty Matrix)',
                     'Confidence Threshold': default_threshold
                 } for lbl in all_labels_in_map])

            st.write(f"HF Rules Extractor: Vectorization complete. Shape: {X.shape}")

    except Exception as e:
        st.error(f"HF Rules Extractor: Error during text vectorization: {e}")
        # Return default DF structure
        return pd.DataFrame([{
            'Label': lbl,
            'Keywords': f'{default_keywords} (Vectorization Error)',
            'Confidence Threshold': default_threshold
        } for lbl in all_labels_in_map])

    # --- Chi-Squared Calculation Per Label ---
    st.info("HF Rules Extractor: Calculating Chi-Squared scores for each label...")
    rules_list = []
    num_features = len(feature_names)

    with st.spinner("Analyzing feature relevance for each label..."):
        for label_idx in range(num_labels):
            label_name = reverse_label_map.get(label_idx, f"UnknownLabel_{label_idx}")
            y = train_labels_array[:, label_idx] # Target vector for this label

            # Default keywords in case of issues
            current_keywords = default_keywords

            # Check if the target variable has variance (Chi2 requires at least two classes)
            if np.std(y) < 1e-9: # Check for near-zero variance
                st.warning(f"HF Rules Extractor: Skipping Chi2 for label '{label_name}' due to zero or near-zero variance in labels.")
                current_keywords = f"{default_keywords} (No Variance)"
            else:
                try:
                    # Calculate Chi-Squared scores
                    chi2_scores, _ = chi2(X, y) # Returns scores and p-values

                    # Handle potential NaNs or Infs in scores (can happen with sparse data)
                    valid_scores_mask = ~np.isnan(chi2_scores) & ~np.isinf(chi2_scores)
                    if not np.any(valid_scores_mask):
                         st.warning(f"HF Rules Extractor: No valid Chi2 scores found for label '{label_name}'.")
                         current_keywords = f"{default_keywords} (No Valid Scores)"
                    else:
                        # Combine valid feature names and their scores
                        feature_scores = sorted(
                            zip(feature_names[valid_scores_mask], chi2_scores[valid_scores_mask]),
                            key=lambda item: item[1], # Sort by score (higher is better)
                            reverse=True
                        )

                        # Select top N features (e.g., 7)
                        top_n = 7
                        top_features = feature_scores[:top_n]

                        if top_features:
                            # Join the keywords (feature names) with commas
                            current_keywords = ', '.join([word for word, score in top_features])
                        else:
                            st.warning(f"HF Rules Extractor: No significant features found for label '{label_name}' after Chi2 calculation.")
                            current_keywords = f"{default_keywords} (No Significant Features)"

                except ValueError as e_chi2_val:
                     # This often happens if a label has too few positive examples relative to features
                     st.warning(f"HF Rules Extractor: Chi2 ValueError for label '{label_name}' (likely low sample count or variance issue): {e_chi2_val}")
                     current_keywords = f"{default_keywords} (Low Variance/Samples)"
                except Exception as e_chi2:
                    st.error(f"HF Rules Extractor: Unexpected error during Chi2 calculation for label '{label_name}': {e_chi2}")
                    current_keywords = f"{default_keywords} (Calculation Error)"

            # Append the result for this label
            rules_list.append({
                'Label': label_name,
                'Keywords': current_keywords,
                'Confidence Threshold': default_threshold # Initialize with default threshold
            })

    st.success("HF Rules Extractor: Keyword extraction finished.")
    return pd.DataFrame(rules_list, columns=required_columns)


# --- HF Classification (Applies Rules) ---

def classify_texts_with_hf(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    label_map: Dict[str, int],
    rules_df: Optional[pd.DataFrame]
) -> List[List[str]]:
    """
    Classifies a list of texts using a trained Hugging Face model and tokenizer.

    Applies confidence thresholds and keyword overrides based on the provided rules_df.
    Includes a fallback mechanism to assign the highest probability label if no
    other labels meet the criteria.

    Args:
        texts: A list of strings to classify.
        model: The trained Hugging Face model object.
        tokenizer: The Hugging Face tokenizer object.
        label_map: Dictionary mapping label names to integer indices.
        rules_df: DataFrame containing rules with columns 'Label', 'Keywords',
                  'Confidence Threshold'. If None or invalid, uses defaults.

    Returns:
        A list of lists, where each inner list contains the predicted label strings
        for the corresponding input text. Returns [['Error']] per text on critical failure.
    """
    if not texts:
        st.warning("HF Classify: Input text list is empty.")
        return []
    if model is None or tokenizer is None or label_map is None:
        st.error("HF Classify Error: Missing required components (model, tokenizer, or label_map).")
        return [["Error: Missing Model Components"] for _ in texts]

    # --- 1. Load and Prepare Rules ---
    thresholds, keyword_override_map = _prepare_rules_for_classification(rules_df, label_map)

    # --- 2. Setup for Classification ---
    st.info("HF Classify: Starting classification...")
    reverse_label_map = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)
    all_results = []
    batch_size = 16 # Adjust based on available memory (especially GPU)
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"HF Classify: Running on device: {device}")

    try:
        model.to(device) # Move model to the appropriate device
        model.eval()     # Set model to evaluation mode

        # Ensure texts are strings, handle potential NaN/None values
        cleaned_texts = [str(t) if pd.notna(t) else "" for t in texts]
        total_texts = len(cleaned_texts)

        # --- 3. Process Texts in Batches ---
        for i in range(0, total_texts, batch_size):
            batch_texts = cleaned_texts[i : min(i + batch_size, total_texts)]
            if not batch_texts: # Should not happen with the loop condition, but safe check
                continue

            current_batch_size = len(batch_texts)
            progress_text.text(f"HF Classifying: Processing texts {i+1}-{i+current_batch_size} of {total_texts}...")

            try:
                # --- Tokenization and Prediction ---
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512 # Standard max length for many BERT-like models
                )
                # Move input tensors to the same device as the model
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad(): # Disable gradient calculations for inference
                    outputs = model(**inputs)
                    # Apply sigmoid to logits for multi-label probabilities
                    probabilities = torch.sigmoid(outputs.logits).cpu().numpy() # Move results back to CPU

                # --- Apply Rules and Fallback Logic per Text in Batch ---
                batch_final_labels = []
                for j in range(current_batch_size): # Index within the current batch
                    text_idx = i + j # Overall index in the original list
                    text_probabilities = probabilities[j] # Probabilities for this text
                    original_text = batch_texts[j] # Original text for keyword matching

                    final_labels_for_text = _apply_rules_and_fallback(
                        text_probabilities,
                        original_text,
                        thresholds,
                        keyword_override_map,
                        reverse_label_map,
                        num_labels
                    )
                    batch_final_labels.append(final_labels_for_text)

                all_results.extend(batch_final_labels)

            except Exception as e_batch:
                st.error(f"HF Classify Error: Failed processing batch starting at index {i}: {e_batch}")
                st.error(traceback.format_exc())
                # Add error placeholders for texts in the failed batch
                all_results.extend([["Error: Batch Processing Failed"] for _ in range(current_batch_size)])

            # Update progress bar
            progress = min(1.0, (i + current_batch_size) / total_texts) if total_texts > 0 else 1.0
            progress_bar.progress(progress)

        # --- 4. Finalization ---
        progress_text.text(f"HF Classification completed for {total_texts} texts.")
        st.success("HF Classification finished successfully.")
        # Consider removing this or making it optional - results are returned
        # st.success("View results in the Results tab.")
        return all_results

    except Exception as e_outer:
         st.error(f"HF Classify Error: A critical error occurred during classification setup or loop: {e_outer}")
         st.error(traceback.format_exc())
         # Return errors for all texts if setup fails
         return [["Error: Classification Setup Failed"] for _ in texts]
    finally:
        # Ensure model is moved back to CPU if it was on GPU, to free VRAM
        # model.cpu() # Or handle device management more carefully if model is reused
        pass


def _prepare_rules_for_classification(
    rules_df: Optional[pd.DataFrame],
    label_map: Dict[str, int]
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Processes the rules DataFrame to extract thresholds and keyword overrides.

    Args:
        rules_df: DataFrame with 'Label', 'Confidence Threshold', 'Keywords'.
        label_map: Dictionary mapping label names to indices.

    Returns:
        A tuple containing:
        - thresholds (Dict[str, float]): Map from label name to confidence threshold.
        - keyword_override_map (Dict[str, List[str]]): Map from label name to list of keywords.
    """
    default_threshold = config.DEFAULT_HF_THRESHOLD
    thresholds: Dict[str, float] = {}
    keyword_override_map: Dict[str, List[str]] = {}

    all_known_labels = set(label_map.keys())

    if rules_df is None or rules_df.empty or not all(col in rules_df.columns for col in ['Label', 'Confidence Threshold', 'Keywords']):
        st.warning("HF Classify Helper: Rules DataFrame is missing, empty, or lacks required columns ('Label', 'Confidence Threshold', 'Keywords'). Using default threshold for all labels and no keyword overrides.")
        thresholds = {label: default_threshold for label in all_known_labels}
        keyword_override_map = {}
    else:
        st.info("HF Classify Helper: Processing thresholds and keywords from rules DataFrame...")
        processed_rules = rules_df.copy()
        try:
            # Clean and validate thresholds
            processed_rules['Confidence Threshold'] = pd.to_numeric(processed_rules['Confidence Threshold'], errors='coerce')
            processed_rules['Confidence Threshold'] = processed_rules['Confidence Threshold'].fillna(default_threshold).clip(0.05, 0.95)
            thresholds = dict(zip(processed_rules['Label'], processed_rules['Confidence Threshold']))

            # Ensure all labels from label_map have a threshold
            for label in all_known_labels:
                if label not in thresholds:
                    st.warning(f"HF Classify Helper: Label '{label}' from label_map not found in rules thresholds. Assigning default threshold {default_threshold}.")
                    thresholds[label] = default_threshold

            # Process keywords
            processed_rules['Keywords'] = processed_rules['Keywords'].fillna('').astype(str)
            for _, row in processed_rules.iterrows():
                label = row['Label']
                if label not in all_known_labels:
                    st.warning(f"HF Classify Helper: Rule found for label '{label}' which is not in the current label_map. Ignoring keywords for this rule.")
                    continue

                # Split keywords, clean them, and filter out 'N/A' or empty strings
                keywords_raw = str(row['Keywords']).split(',')
                keywords_clean = [kw.strip().lower() for kw in keywords_raw if kw.strip() and "N/A" not in kw]

                if keywords_clean:
                    keyword_override_map[label] = keywords_clean
            st.write(f"HF Classify Helper: Loaded thresholds for {len(thresholds)} labels and keyword rules for {len(keyword_override_map)} labels.")

        except Exception as e:
            st.error(f"HF Classify Helper: Error processing rules DataFrame: {e}. Falling back to default thresholds and no keywords.")
            thresholds = {label: default_threshold for label in all_known_labels}
            keyword_override_map = {}

    return thresholds, keyword_override_map


def _apply_rules_and_fallback(
    probabilities: np.ndarray,
    text: str,
    thresholds: Dict[str, float],
    keyword_override_map: Dict[str, List[str]],
    reverse_label_map: Dict[int, str],
    num_labels: int,
    fallback_threshold: float = 0.1 # Minimum probability for fallback label
) -> List[str]:
    """
    Applies thresholds, keyword overrides, and fallback logic to determine final labels for a single text.

    Args:
        probabilities: NumPy array of probabilities for each label for this text.
        text: The original text string (used for keyword matching).
        thresholds: Dictionary mapping label names to confidence thresholds.
        keyword_override_map: Dictionary mapping label names to lists of keywords.
        reverse_label_map: Dictionary mapping label indices to label names.
        num_labels: Total number of labels.
        fallback_threshold: Minimum probability required to assign the highest-probability label as a fallback.

    Returns:
        A list of predicted label strings for the text.
    """
    default_threshold = config.DEFAULT_HF_THRESHOLD # Consistent default
    text_lower = text.lower() # Pre-lower for efficient keyword matching
    initial_labels = set()

    # 1. Determine initial labels based on model probabilities and thresholds
    for label_idx in range(num_labels):
        label_name = reverse_label_map.get(label_idx)
        if label_name is None: continue # Should not happen if maps are consistent

        threshold = thresholds.get(label_name, default_threshold)
        if probabilities[label_idx] >= threshold:
            initial_labels.add(label_name)

    # 2. Apply keyword overrides (add labels if keyword found, even if below threshold)
    final_labels = initial_labels.copy()
    if keyword_override_map:
        for label_name, keywords in keyword_override_map.items():
            # Only add if not already present based on threshold
            if label_name in final_labels:
                continue

            # Check if any keyword for this label is present in the text
            found_keyword = False
            for keyword in keywords:
                if not keyword: continue # Skip empty keywords just in case
                try:
                    # Use regex for whole word matching (\b)
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                        found_keyword = True
                        break
                except re.error:
                    # Fallback to simple substring check if regex fails (e.g., special chars in keyword)
                    st.warning(f"HF Classify Keyword Regex Error for keyword: '{keyword}'. Using simple substring check.", icon="⚠️")
                    if keyword in text_lower:
                        found_keyword = True
                        break
            # If a keyword was found, add the label
            if found_keyword:
                final_labels.add(label_name)

    # 3. Fallback: If no labels assigned yet, assign the single highest probability label
    #    (only if its probability exceeds the fallback_threshold)
    if not final_labels and len(probabilities) > 0:
        highest_prob_idx = np.argmax(probabilities)
        highest_prob_value = probabilities[highest_prob_idx]

        if highest_prob_value > fallback_threshold:
            fallback_label = reverse_label_map.get(highest_prob_idx)
            if fallback_label:
                final_labels.add(fallback_label)
                # Optional: Log when fallback is used for debugging/analysis
                # st.caption(f"Fallback applied: '{fallback_label}' (Prob: {highest_prob_value:.3f})")

    return sorted(list(final_labels)) # Return sorted list for consistency
