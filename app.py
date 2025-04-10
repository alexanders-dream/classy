# app.py
"""Main Streamlit application file for the Hierarchical Text Classifier."""

# Standard Library Imports
import os
import re
from pathlib import Path
import traceback
from typing import List

# Third-Party Imports
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Local Application Imports
import config
import utils
import ui_components
import hf_classifier
import llm_classifier

# --- Load Environment Variables ---
# Load early to ensure availability
load_dotenv()

# --- Helper Functions ---

def list_saved_hf_models(base_path: Path) -> List[str]:
    """
    Lists subdirectories in the base_path, assuming they are saved Hugging Face models.
    Creates the base path if it doesn't exist.

    Args:
        base_path: The Path object representing the directory containing saved models.

    Returns:
        A sorted list of model directory names, or an empty list if an error occurs.
    """
    try:
        base_path.mkdir(parents=True, exist_ok=True)
        return sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    except Exception as e:
        st.error(f"Error listing models in {base_path}: {e}")
        return []

def sanitize_foldername(name: str) -> str:
    """
    Cleans a string to be suitable as a folder name.
    Removes leading/trailing whitespace and problematic characters,
    replaces multiple underscores with a single one, and ensures
    a default name if the result is empty.

    Args:
        name: The input string name.

    Returns:
        A sanitized string suitable for use as a folder name.
    """
    if not isinstance(name, str):
        name = str(name) # Attempt to convert non-strings
    name = name.strip()
    # Remove characters not allowed in folder names (allow letters, numbers, underscore, hyphen, period)
    name = re.sub(r'[^\w\-\.]+', '_', name)
    # Remove leading/trailing underscores or periods that might cause issues
    name = name.strip('_.')
    # Replace multiple consecutive underscores with a single one
    name = re.sub(r'_+', '_', name)
    # Return a default name if the sanitized name is empty
    return name if name else "unnamed_model"

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Text Classifier",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Moved detailed init logic to utils.init_session_state()
if 'session_initialized' not in st.session_state:
    utils.init_session_state() # This should handle all default initializations
    st.session_state.session_initialized = True

# --- Main App Title ---
st.title("🏷️ AI Hierarchical Text Classifier")
st.markdown("Classify text using Hugging Face models or Large Language Models (LLMs).")

# --- Workflow Selection ---
st.sidebar.title("🛠️ Workflow")
workflow_options = ["LLM Categorization", "Hugging Face Model"]
selected_workflow = st.sidebar.radio(
    "Choose Method:",
    options=workflow_options,
    key='selected_workflow',
    horizontal=True
)
st.sidebar.markdown("---")

# --- Display LLM Sidebar ---
llm_ready = False # Default value
if selected_workflow == "LLM Categorization":
    ui_components.display_llm_sidebar()
    llm_ready = st.session_state.get('llm_client') is not None

# --- Main Application Tabs ---
tab_setup, tab_hierarchy, tab_classify, tab_results = st.tabs([
    "1. Data Setup",
    "2. Hierarchy",
    "3. Classification",
    "4. Results"
])

# === Tab 1: Data Setup ===
with tab_setup:
    st.header("1. Data Upload and Column Selection")
    st.markdown("Upload your data and select the text column for classification.")

    col_pred, col_train = st.columns(2)

    # --- Column 1: Prediction Data ---
    with col_pred:
        st.subheader("Data to Classify")
        help_pred = "Upload a CSV or Excel file containing the text data you want to categorize."
        uncategorized_file = st.file_uploader(
            "Upload Unlabeled Data",
            type=['csv', 'xlsx', 'xls'],
            key="uncat_uploader_main",
            help=help_pred
        )

        # Load prediction data if a new file is uploaded
        if uncategorized_file and uncategorized_file.file_id != st.session_state.get('uncategorized_file_key'):
            with st.spinner(f"Loading '{uncategorized_file.name}'..."):
                st.session_state.uncategorized_df = utils.load_data(uncategorized_file)
            # Reset related state variables
            st.session_state.uncategorized_file_key = uncategorized_file.file_id
            st.session_state.uncat_text_col = None
            st.session_state.results_df = None # Clear previous results
            st.session_state.app_stage = 'file_uploaded'
            # No rerun needed here, handled by Streamlit's flow

        # Display prediction data preview and column selector
        if st.session_state.uncategorized_df is not None:
            uncat_df = st.session_state.uncategorized_df
            st.success(f"Prediction data loaded ({uncat_df.shape[0]} rows).")
            with st.expander("Preview Prediction Data"):
                st.dataframe(uncat_df.head())

            st.markdown("**Select Text Column:**")
            uncat_df.columns = uncat_df.columns.astype(str) # Ensure string column names
            uncat_cols = [""] + uncat_df.columns.tolist() # Add empty option
            current_uncat_col = st.session_state.uncat_text_col
            default_uncat_idx = 0
            if current_uncat_col in uncat_cols:
                try:
                    default_uncat_idx = uncat_cols.index(current_uncat_col)
                except ValueError: # Should not happen if check passes, but safety first
                    st.session_state.uncat_text_col = None # Reset if invalid
            elif current_uncat_col: # If it was set but not found (e.g., file changed)
                 st.session_state.uncat_text_col = None # Reset

            selected_uncat_col = st.selectbox(
                "Select column containing text to classify:",
                options=uncat_cols,
                index=default_uncat_idx,
                key="uncat_text_select_main",
                label_visibility="collapsed"
            )

            # Update session state if selection changes
            if selected_uncat_col and selected_uncat_col != st.session_state.uncat_text_col:
                st.session_state.uncat_text_col = selected_uncat_col
                st.session_state.app_stage = 'column_selected'
                st.rerun() # Rerun to reflect selection change immediately
            elif selected_uncat_col:
                st.caption(f"Using column: **'{selected_uncat_col}'**")
        else:
            st.info("Upload the data you want to classify.")

    # --- Column 2: Training Data (HF Only) ---
    with col_train:
        if selected_workflow == "Hugging Face Model":
            st.subheader("Training Data (Optional)")
            help_train = "For HF: Upload labeled data (CSV/Excel) with text and hierarchy columns."
            categorized_file = st.file_uploader(
                "Upload Labeled Data (for HF Training)",
                type=['csv', 'xlsx', 'xls'],
                key="cat_uploader_main",
                help=help_train
            )

            # Load training data if a new file is uploaded
            if categorized_file and categorized_file.file_id != st.session_state.get('categorized_file_key'):
                with st.spinner(f"Loading '{categorized_file.name}'..."):
                    st.session_state.categorized_df = utils.load_data(categorized_file)
                # Reset related state variables
                st.session_state.categorized_file_key = categorized_file.file_id
                st.session_state.cat_text_col = None
                for level in config.HIERARCHY_LEVELS:
                    st.session_state[f'cat_{level.lower()}_col'] = None
                # Reset model state as new training data invalidates old model/rules
                st.session_state.hf_model_ready = False
                st.session_state.hf_model = None
                st.session_state.hf_tokenizer = None
                st.session_state.hf_label_map = None
                st.session_state.hf_rules = pd.DataFrame(columns=config.HF_RULE_COLUMNS)
                # No rerun needed here

            # Display training data preview and column selectors
            if st.session_state.categorized_df is not None:
                cat_df = st.session_state.categorized_df
                st.success(f"Training data loaded ({cat_df.shape[0]} rows).")
                with st.expander("Preview Training Data"):
                    st.dataframe(cat_df.head())

                # Form for selecting HF training columns
                with st.form("hf_col_sel_form"):
                    st.markdown("**Select Columns for HF Training:**")
                    cat_df.columns = cat_df.columns.astype(str) # Ensure string column names
                    available_cat_cols = cat_df.columns.tolist()

                    # Text Column Selection
                    current_cat_text_col = st.session_state.cat_text_col
                    default_cat_text_idx = 0
                    if current_cat_text_col in available_cat_cols:
                         try:
                             default_cat_text_idx = available_cat_cols.index(current_cat_text_col)
                         except ValueError:
                             st.session_state.cat_text_col = None # Reset
                    elif current_cat_text_col:
                         st.session_state.cat_text_col = None # Reset

                    selected_cat_text_col = st.selectbox(
                        "Text Column:",
                        available_cat_cols,
                        index=default_cat_text_idx,
                        key="cat_text_sel_main"
                    )

                    # Hierarchy Column Selection
                    st.markdown("Hierarchy Columns:")
                    selected_hierarchy_cols = {}
                    for level in config.HIERARCHY_LEVELS:
                        level_key = f'cat_{level.lower()}_col'
                        current_level_col = st.session_state.get(level_key, "(None)")
                        # Options exclude the selected text column and "(None)"
                        options = ["(None)"] + [c for c in available_cat_cols if c != selected_cat_text_col]
                        default_level_idx = 0
                        if current_level_col in options:
                            try:
                                default_level_idx = options.index(current_level_col)
                            except ValueError: pass # Keep default 0

                        selected_hierarchy_cols[level] = st.selectbox(
                            f"{level}:",
                            options,
                            index=default_level_idx,
                            key=f"{level_key}_sel_main"
                        )

                    # Form Submission
                    submitted_hf_cols = st.form_submit_button("Confirm HF Columns")
                    if submitted_hf_cols:
                        active_selections = {
                            level: col for level, col in selected_hierarchy_cols.items()
                            if col and col != "(None)"
                        }
                        # Validation
                        if not selected_cat_text_col:
                            st.warning("Please select the Text Column.")
                        elif not active_selections:
                            st.warning("Please select at least one Hierarchy Column.")
                        elif len(list(active_selections.values())) != len(set(list(active_selections.values()))):
                            st.warning("Hierarchy columns must be unique.")
                        else:
                            # Update session state on successful validation
                            st.session_state.cat_text_col = selected_cat_text_col
                            for level, col in selected_hierarchy_cols.items():
                                st.session_state[f'cat_{level.lower()}_col'] = col
                            st.success("HF training columns confirmed.")
                            # Reset model state as column selections changed
                            st.session_state.hf_model_ready = False
                            st.session_state.hf_model = None
                            st.session_state.hf_tokenizer = None
                            st.session_state.hf_label_map = None
                            st.session_state.hf_rules = pd.DataFrame(columns=config.HF_RULE_COLUMNS)
                            st.rerun() # Rerun to reflect confirmation

            else: # No training data uploaded
                st.info("Upload labeled data if you plan to train a Hugging Face model.")
        else: # LLM workflow selected
            st.info("Training data upload is only needed for the Hugging Face Model workflow.")


# === Tab 2: Hierarchy Definition ===
with tab_hierarchy:
    st.header("2. Define Classification Hierarchy")

    # --- LLM Workflow: Hierarchy Definition ---
    if selected_workflow == "LLM Categorization":
        st.markdown("Define the classification structure for the LLM. You can use the editor below, generate an AI suggestion based on your data, or refine an existing hierarchy.")

        st.subheader("🤖 AI Hierarchy Suggestion")

        # Check prerequisites for suggestion
        uncat_df = st.session_state.get('uncategorized_df')
        uncat_text_col = st.session_state.get('uncat_text_col')
        suggestion_possible = (uncat_df is not None and uncat_text_col)

        if suggestion_possible:
            # --- Prepare for AI Suggestion ---
            slider_enabled = False
            max_slider_val = config.MIN_LLM_SAMPLE_SIZE
            default_slider_val = config.MIN_LLM_SAMPLE_SIZE
            min_texts_needed = config.MIN_LLM_SAMPLE_SIZE

            try:
                if uncat_text_col in uncat_df.columns:
                    # Attempt to get unique, non-numeric-like string values
                    text_series = uncat_df[uncat_text_col].dropna()
                    # Filter out purely numeric strings if possible, keep others
                    is_numeric_like = pd.to_numeric(text_series, errors='coerce').notna()
                    string_texts = text_series[~is_numeric_like].astype(str)
                    unique_texts = string_texts.unique()
                    unique_count = len(unique_texts)

                    if unique_count >= min_texts_needed:
                        max_slider_val = min(config.MAX_LLM_SAMPLE_SIZE, unique_count)
                        default_slider_val = min(config.DEFAULT_LLM_SAMPLE_SIZE, max_slider_val)
                        slider_enabled = llm_ready # Enable slider only if LLM is ready and enough data
                        if not llm_ready:
                             st.warning("LLM Client not configured (see sidebar). Cannot generate suggestion.")
                    else:
                        st.warning(f"Need at least {min_texts_needed} unique non-numeric text samples in '{uncat_text_col}' to generate suggestions.")
                        slider_enabled = False # Ensure disabled
                else:
                    st.warning(f"Selected text column '{uncat_text_col}' not found in the uploaded data.")
                    slider_enabled = False # Ensure disabled
            except Exception as e:
                st.error(f"Error preparing for AI suggestion: {e}")
                traceback.print_exc() # Log detailed error
                slider_enabled = False # Ensure disabled

            # --- AI Suggestion UI ---
            sample_size = st.slider(
                f"Number of unique text samples to use from '{uncat_text_col}':",
                min_value=config.MIN_LLM_SAMPLE_SIZE,
                max_value=max_slider_val,
                value=default_slider_val,
                step=50,
                key="ai_sample_slider_main",
                help="More samples provide more context to the LLM but take longer to process.",
                disabled=not slider_enabled
            )

            if st.button("🚀 Generate Suggestion", key="generate_ai_hierarchy_main", type="primary", disabled=not slider_enabled):
                if llm_ready: # Double-check LLM readiness
                    st.session_state.ai_suggestion_pending = None # Clear previous pending
                    with st.spinner("🧠 Asking AI to suggest a hierarchy based on sample data..."):
                        try:
                            # Reselect unique texts within the button click logic for safety
                            text_series = uncat_df[uncat_text_col].dropna()
                            is_numeric_like = pd.to_numeric(text_series, errors='coerce').notna()
                            string_texts = text_series[~is_numeric_like].astype(str)
                            unique_list = string_texts.unique().tolist()

                            if len(unique_list) >= min_texts_needed:
                                # Ensure sample size doesn't exceed available unique texts
                                actual_sample_size = min(len(unique_list), sample_size)
                                # Sample if needed, otherwise use all unique texts
                                if len(unique_list) > actual_sample_size:
                                    sampled_texts = pd.Series(unique_list).sample(actual_sample_size, random_state=42).tolist()
                                else:
                                    sampled_texts = unique_list

                                suggestion = llm_classifier.generate_hierarchy_suggestion(
                                    st.session_state.llm_client,
                                    sampled_texts
                                )
                                if suggestion:
                                    st.session_state.ai_suggestion_pending = suggestion
                                    st.success("✅ AI hierarchy suggestion generated!")
                                else:
                                    st.error("❌ AI failed to generate a suggestion. The response might have been empty or invalid.")
                                st.rerun() # Rerun to display suggestion options in the editor
                            else:
                                # This check might be redundant due to slider disabling, but good safety
                                st.warning(f"Insufficient unique text samples ({len(unique_list)} found) to generate suggestion. Need at least {min_texts_needed}.")
                        except Exception as e:
                            st.error(f"An error occurred during suggestion generation: {e}")
                            traceback.print_exc()
                else:
                    st.error("LLM Client is not ready. Please configure it in the sidebar.")
        else: # Suggestion not possible (no data/column)
             st.info("Upload prediction data and select the text column in '1. Data Setup' to enable AI suggestions.")

        st.divider()
        st.subheader("✏️ Hierarchy Editor")
        # The display_hierarchy_editor function handles showing the editor and applying suggestions
        hierarchy_valid = ui_components.display_hierarchy_editor(key_prefix="llm")

    # --- Hugging Face Workflow: Hierarchy Display ---
    elif selected_workflow == "Hugging Face Model":
        st.info("For the Hugging Face workflow, the hierarchy is implicitly defined by the columns selected in '1. Data Setup'.")
        active_hf_cols = {
            level: st.session_state.get(f'cat_{level.lower()}_col')
            for level in config.HIERARCHY_LEVELS
            if st.session_state.get(f'cat_{level.lower()}_col') and st.session_state.get(f'cat_{level.lower()}_col') != "(None)"
        }
        if active_hf_cols:
            st.markdown("**Selected Hierarchy Columns (from Training Data):**")
            for level, column_name in active_hf_cols.items():
                st.write(f"- **{level}:** `{column_name}`")
        else:
            st.warning("No hierarchy columns selected in '1. Data Setup'. Please select the relevant columns from your training data.")

# === Tab 3: Run Classification ===
with tab_classify:
    st.header("3. Run Classification")

    # Display completion message if classification just finished
    if st.session_state.get('classification_just_completed', False):
        st.success("✅ Classification Complete! View results in the '4. Results' tab.")
        # Reset the flag after displaying the message
        st.session_state.classification_just_completed = False

    # --- Hugging Face Workflow ---
    if selected_workflow == "Hugging Face Model":
        st.subheader("Hugging Face Model: Train, Load, and Classify")

        # --- Readiness Checks ---
        hf_training_data_loaded = st.session_state.get('categorized_df') is not None
        hf_training_cols_selected = (
            st.session_state.get('cat_text_col') and
            any(st.session_state.get(f'cat_{level.lower()}_col') and
                st.session_state.get(f'cat_{level.lower()}_col') != "(None)"
                for level in config.HIERARCHY_LEVELS)
        )
        hf_ready_for_training = hf_training_data_loaded and hf_training_cols_selected
        hf_model_is_ready = st.session_state.get('hf_model_ready', False)
        hf_prediction_data_ready = (
            st.session_state.get('uncategorized_df') is not None and
            st.session_state.get('uncat_text_col') is not None
        )
        hf_ready_for_classification = hf_model_is_ready and hf_prediction_data_ready

        # --- Option A: Train/Retrain Model ---
        with st.container(border=True):
            st.markdown("**Option A: Train or Retrain a Hugging Face Model**")
            if not hf_ready_for_training:
                st.warning("Training requires labeled data and selected columns in '1. Data Setup'.")

            # Training Parameters
            train_params_disabled = not hf_ready_for_training
            col_model, col_epochs, col_split = st.columns(3)
            with col_model:
                hf_base_model = st.selectbox(
                    "Base Model:",
                    options=["distilbert-base-uncased", "bert-base-uncased", "roberta-base"],
                    index=0,
                    disabled=train_params_disabled,
                    key="hf_base_model_select"
                )
            with col_epochs:
                hf_num_epochs = st.slider(
                    "Epochs:",
                    min_value=1, max_value=10, value=3, step=1,
                    disabled=train_params_disabled,
                    key="hf_epochs_slider"
                )
            with col_split:
                hf_val_split_percent = st.slider(
                    "Validation Split (%):",
                    min_value=5, max_value=50, value=int(config.DEFAULT_VALIDATION_SPLIT * 100), step=5,
                    help="Percentage of training data held out for validation during training.",
                    disabled=train_params_disabled,
                    key="hf_val_split_slider"
                )
                hf_val_split_ratio = hf_val_split_percent / 100.0

            # Training Button
            train_button_text = "🚀 Start HF Training" if not hf_model_is_ready else "🔄 Retrain HF Model"
            if st.button(train_button_text, type="primary", disabled=train_params_disabled):
                st.session_state.classification_just_completed = False # Reset flag

                # Reset existing model state if retraining
                if hf_model_is_ready:
                    st.info("Clearing previous model state before retraining...")
                    utils.reset_hf_model_state() # Use utility function

                # Prepare data for training
                hierarchy_map = {
                    level: st.session_state.get(f'cat_{level.lower()}_col')
                    for level in config.HIERARCHY_LEVELS
                }
                active_hierarchy_cols = {l: c for l, c in hierarchy_map.items() if c and c != "(None)"}

                if not st.session_state.cat_text_col or not active_hierarchy_cols:
                    st.error("Cannot train: Text column or hierarchy columns are not properly selected in Tab 1.")
                else:
                    with st.spinner("Preparing training data..."):
                        prepared_texts, prepared_labels = hf_classifier.prepare_hierarchical_training_data(
                            st.session_state.categorized_df,
                            st.session_state.cat_text_col,
                            hierarchy_map
                        )

                    if prepared_texts and prepared_labels:
                        with st.spinner(f"Training {hf_base_model} for {hf_num_epochs} epochs... This may take a while."):
                            try:
                                model, tokenizer, label_map, rules_df = hf_classifier.train_hf_model(
                                    prepared_texts,
                                    prepared_labels,
                                    hf_base_model,
                                    hf_num_epochs,
                                    hf_val_split_ratio
                                )
                                if model and tokenizer and label_map is not None:
                                    st.session_state.hf_model = model
                                    st.session_state.hf_tokenizer = tokenizer
                                    st.session_state.hf_label_map = label_map
                                    st.session_state.hf_rules = rules_df if rules_df is not None else pd.DataFrame(columns=config.HF_RULE_COLUMNS)
                                    st.session_state.hf_model_ready = True
                                    st.session_state.hf_model_save_name = f"trained_{hf_base_model.split('/')[-1]}" # Suggest a name
                                    st.success("✅ Hugging Face Model trained successfully!")
                                    st.rerun() # Rerun to update UI (e.g., enable save/classify)
                                else:
                                    st.error("❌ Model training failed. Check logs or parameters.")
                            except Exception as e:
                                st.error(f"An error occurred during training: {e}")
                                traceback.print_exc()
                    else:
                        st.error("❌ Data preparation for training failed. Check input data and column selections.")

            # --- Save Trained Model Section ---
            if hf_model_is_ready and st.session_state.get('hf_model'):
                st.success(f"HF Model Ready: '{st.session_state.get('hf_model_save_name', '(Untitled)')}'")
                st.markdown("**Save Current Trained Model:**")
                save_name = st.text_input(
                    "Model Save Name:",
                    value=st.session_state.get("hf_model_save_name", ""),
                    key="hf_save_name_input",
                    placeholder="e.g., product_classifier_v1"
                )
                if st.button("💾 Save Model", key="save_hf_model_button"):
                    st.session_state.classification_just_completed = False # Reset flag
                    sanitized_name = sanitize_foldername(save_name)
                    if not sanitized_name or sanitized_name == "unnamed_model":
                        st.warning("Please enter a valid name for saving the model.")
                    else:
                        save_directory = config.SAVED_HF_MODELS_BASE_PATH / sanitized_name
                        if save_directory.exists():
                            st.warning(f"Directory '{sanitized_name}' already exists. Saving will overwrite its contents.")

                        with st.spinner(f"Saving model to '{sanitized_name}'..."):
                            success = hf_classifier.save_hf_model_artifacts(
                                st.session_state.hf_model,
                                st.session_state.hf_tokenizer,
                                st.session_state.hf_label_map,
                                st.session_state.hf_rules,
                                str(save_directory) # Pass path as string
                            )
                            if success:
                                st.session_state.hf_model_save_name = sanitized_name # Update state with the saved name
                                st.success(f"Model successfully saved as '{sanitized_name}'.")
                            else:
                                st.error("❌ Failed to save model artifacts.")
            # Display nothing if model not ready and training not possible
            elif not hf_ready_for_training and not hf_model_is_ready:
                pass # Avoid showing save section if no model is ready/trainable

        st.markdown("---") # Separator

        # --- Option B: Load Existing Model ---
        with st.container(border=True):
            st.markdown("**Option B: Load an Existing Saved HF Model**")
            st.caption(f"Models are loaded from: `{config.SAVED_HF_MODELS_BASE_PATH}`")
            saved_models = list_saved_hf_models(config.SAVED_HF_MODELS_BASE_PATH)

            if not saved_models:
                st.info("No saved models found in the specified directory.")
            else:
                load_options = [""] + saved_models # Add empty option for placeholder
                current_model_name = st.session_state.get('hf_model_save_name')
                current_index = 0
                if current_model_name in load_options:
                    try:
                        current_index = load_options.index(current_model_name)
                    except ValueError: pass # Should not happen

                selected_model_to_load = st.selectbox(
                    "Select Model to Load:",
                    options=load_options,
                    index=current_index,
                    key="hf_load_model_select",
                    help="Choose the folder name of the saved model."
                )

                # Disable load button if a model is already loaded or no model is selected
                load_button_disabled = hf_model_is_ready or not selected_model_to_load
                if st.button("🔄 Load Selected Model", disabled=load_button_disabled):
                    st.session_state.classification_just_completed = False # Reset flag
                    if hf_model_is_ready:
                         # This case should be prevented by disabled button, but good safety check
                        st.warning("A model is already loaded. Cannot load another.")
                    elif selected_model_to_load:
                        load_directory = config.SAVED_HF_MODELS_BASE_PATH / selected_model_to_load
                        with st.spinner(f"Loading model '{selected_model_to_load}'..."):
                            try:
                                model, tokenizer, label_map, rules_df = hf_classifier.load_hf_model_artifacts(str(load_directory))
                                if model and tokenizer and label_map is not None:
                                    st.session_state.hf_model = model
                                    st.session_state.hf_tokenizer = tokenizer
                                    st.session_state.hf_label_map = label_map
                                    st.session_state.hf_rules = rules_df if rules_df is not None else pd.DataFrame(columns=config.HF_RULE_COLUMNS)
                                    st.session_state.hf_model_ready = True
                                    st.session_state.hf_model_save_name = selected_model_to_load # Update state
                                    st.success(f"✅ Model '{selected_model_to_load}' loaded successfully!")
                                    st.rerun() # Update UI
                                else:
                                    st.error(f"❌ Failed to load model '{selected_model_to_load}'. Check if the directory contains valid artifacts.")
                            except Exception as e:
                                st.error(f"An error occurred during loading: {e}")
                                traceback.print_exc()
                    # No else needed for !selected_model_to_load as button is disabled

            if hf_model_is_ready:
                st.success(f"Active Model: '{st.session_state.get('hf_model_save_name', 'Loaded Model')}'")

        st.markdown("---") # Separator

        # --- HF Rule Editor ---
        st.subheader("Review & Edit HF Classification Rules")
        current_rules = st.session_state.get('hf_rules')

        if hf_model_is_ready and current_rules is not None and not current_rules.empty:
            # Check for essential columns needed for editing
            if 'Label' in current_rules.columns and 'Confidence Threshold' in current_rules.columns:
                with st.form("hf_rules_edit_form"):
                    st.info("Edit Labels, add comma-separated Keywords (which force the label if found), and adjust Confidence Thresholds (0.05-0.95).")

                    # Prepare DataFrame for editing
                    rules_to_edit = current_rules.copy()
                    # Ensure 'Keywords' column exists and fill NaNs
                    if 'Keywords' not in rules_to_edit.columns:
                        rules_to_edit['Keywords'] = '' # Add empty column if missing
                    rules_to_edit['Keywords'] = rules_to_edit['Keywords'].fillna('')
                    # Ensure 'Confidence Threshold' is numeric, fill NaNs, and clip
                    rules_to_edit['Confidence Threshold'] = pd.to_numeric(rules_to_edit['Confidence Threshold'], errors='coerce')
                    rules_to_edit['Confidence Threshold'] = rules_to_edit['Confidence Threshold'].fillna(config.DEFAULT_HF_THRESHOLD)
                    rules_to_edit['Confidence Threshold'] = rules_to_edit['Confidence Threshold'].clip(0.05, 0.95)
                    # Ensure 'Label' is string
                    rules_to_edit['Label'] = rules_to_edit['Label'].astype(str)

                    # Define columns for the editor (ensure order and configuration)
                    column_order = ['Label', 'Keywords', 'Confidence Threshold']
                    # Include other columns if they exist, but don't configure them for editing
                    other_cols = [col for col in rules_to_edit.columns if col not in column_order]
                    display_cols = column_order + other_cols

                    edited_rules_df = st.data_editor(
                        rules_to_edit.sort_values('Label')[display_cols], # Sort for consistency, display defined cols
                        column_config={
                            "Label": st.column_config.TextColumn("Label (Editable)", required=True),
                            "Keywords": st.column_config.TextColumn("Keywords (Editable, comma-separated)"),
                            "Confidence Threshold": st.column_config.NumberColumn(
                                "Confidence Threshold (Editable)",
                                min_value=0.05, max_value=0.95, step=0.01, format="%.3f",
                                required=True
                            )
                            # Other columns will use default config (likely read-only)
                        },
                        use_container_width=True,
                        hide_index=True,
                        num_rows="dynamic", # Allow adding/deleting rows
                        key="hf_rules_data_editor"
                    )

                    submitted = st.form_submit_button("Save Rule Changes")
                    if submitted:
                        st.session_state.classification_just_completed = False # Reset flag
                        if isinstance(edited_rules_df, pd.DataFrame):
                            # --- Validation of edited rules ---
                            valid = True
                            if 'Label' not in edited_rules_df.columns or edited_rules_df['Label'].isnull().any() or (edited_rules_df['Label'] == '').any():
                                st.warning("Labels cannot be empty. Please provide a label for each rule.")
                                valid = False
                            if 'Confidence Threshold' not in edited_rules_df.columns:
                                st.warning("Confidence Threshold column is missing.") # Should not happen with config
                                valid = False
                            else:
                                # Coerce threshold again after editing
                                edited_rules_df['Confidence Threshold'] = pd.to_numeric(edited_rules_df['Confidence Threshold'], errors='coerce')
                                if edited_rules_df['Confidence Threshold'].isnull().any():
                                     st.warning("Confidence Threshold must be a valid number between 0.05 and 0.95.")
                                     valid = False
                                else:
                                     # Clip again after potential manual edits outside bounds
                                     edited_rules_df['Confidence Threshold'] = edited_rules_df['Confidence Threshold'].clip(0.05, 0.95)

                            if valid:
                                # Ensure Keywords column exists and fill NaNs after potential edits/deletions
                                if 'Keywords' not in edited_rules_df.columns:
                                     edited_rules_df['Keywords'] = ''
                                edited_rules_df['Keywords'] = edited_rules_df['Keywords'].fillna('')

                                # --- Compare with original rules ---
                                # Select only the core, comparable columns
                                compare_cols = ['Label', 'Keywords', 'Confidence Threshold']
                                # Ensure columns exist in both dataframes before comparison
                                current_compare_cols = [col for col in compare_cols if col in current_rules.columns]
                                edited_compare_cols = [col for col in compare_cols if col in edited_rules_df.columns]

                                if set(current_compare_cols) == set(edited_compare_cols): # Only compare if core columns match
                                    # Sort both by Label and reset index for accurate comparison
                                    current_comp_df = current_rules[current_compare_cols].sort_values('Label').reset_index(drop=True).astype(str)
                                    edited_comp_df = edited_rules_df[edited_compare_cols].sort_values('Label').reset_index(drop=True).astype(str)

                                    if not current_comp_df.equals(edited_comp_df):
                                        st.session_state.hf_rules = edited_rules_df.copy() # Update state
                                        st.success("✅ Rule changes saved successfully!")
                                    else:
                                        st.info("No changes detected in the rules.")
                                else:
                                     st.warning("Could not compare rules due to missing core columns in edited data.")

                        else: # Should not happen with data_editor
                            st.error("Data editor did not return a valid DataFrame.")
            elif 'Label' not in current_rules.columns or 'Confidence Threshold' not in current_rules.columns:
                 st.warning("Cannot edit rules: The loaded rules DataFrame is missing the required 'Label' or 'Confidence Threshold' columns.")
        elif hf_model_is_ready: # Model ready, but rules are None or empty
            st.info("The loaded/trained model does not have any associated rules to edit.")
        else: # Model not ready
            st.info("Train or load a Hugging Face model first to review or edit its rules.")

        st.divider() # Separator

        # --- Run HF Classification ---
        st.header("🚀 Run HF Classification")
        if not hf_prediction_data_ready:
             st.warning("Upload data to classify and select the text column in '1. Data Setup'.")
        if not hf_model_is_ready:
             st.warning("Train or load a Hugging Face model first.")

        if st.button("Classify using Hugging Face Model", type="primary", disabled=not hf_ready_for_classification):
            st.session_state.classification_just_completed = False # Reset flag
            uncat_df = st.session_state.uncategorized_df
            text_col = st.session_state.uncat_text_col

            if text_col and text_col in uncat_df.columns:
                texts_to_classify = uncat_df[text_col].fillna("").astype(str).tolist()
                if texts_to_classify:
                    with st.spinner("Classifying texts using the HF model..."):
                        try:
                            # Perform classification
                            raw_predicted_labels = hf_classifier.classify_texts_with_hf(
                                texts_to_classify,
                                st.session_state.hf_model,
                                st.session_state.hf_tokenizer,
                                st.session_state.hf_label_map,
                                st.session_state.hf_rules # Pass current rules
                            )

                            # Process results
                            st.session_state.raw_predicted_labels = raw_predicted_labels # Store raw for potential analysis
                            parsed_labels_dict = utils.parse_predicted_labels_to_columns(raw_predicted_labels)
                            results_df = uncat_df.copy()
                            predictions_df = pd.DataFrame(parsed_labels_dict, index=results_df.index)

                            # Add predicted columns (e.g., HF_Theme, HF_Category)
                            for level in config.HIERARCHY_LEVELS:
                                results_df[f"HF_{level}"] = predictions_df.get(level) # Use .get for safety

                            # Add raw labels column (optional, for inspection)
                            results_df["HF_Raw_Labels"] = [', '.join(map(str, label_list)) if label_list else None for label_list in raw_predicted_labels]

                            # Update session state
                            st.session_state.results_df = results_df
                            st.session_state.app_stage = 'categorized'
                            st.session_state.classification_just_completed = True # Set flag for success message
                            st.rerun() # Rerun to display results tab and success message

                        except Exception as e:
                            st.error(f"An error occurred during HF classification: {e}")
                            traceback.print_exc()
                else:
                    st.warning("The selected text column contains no text data to classify.")
            elif uncat_df is not None: # Data loaded, but column missing (shouldn't happen often with UI checks)
                st.error(f"Prediction column '{text_col}' not found in the uploaded data.")
            # No else needed if uncat_df is None, covered by initial checks

        elif not hf_ready_for_classification:
            st.info("Please ensure a Hugging Face model is loaded/trained and prediction data is ready.")

    # --- LLM Workflow: Classification ---
    elif selected_workflow == "LLM Categorization":
        st.subheader("🚀 Run LLM Classification")

        # --- Readiness Checks ---
        llm_hierarchy_defined = st.session_state.get('hierarchy_defined', False)
        llm_prediction_data_ready = (
            st.session_state.get('uncategorized_df') is not None and
            st.session_state.get('uncat_text_col') is not None
        )
        llm_ready_for_classification = (
            llm_ready and # From sidebar check
            llm_hierarchy_defined and
            llm_prediction_data_ready
        )

        # Display warnings if not ready
        if not llm_hierarchy_defined:
            st.warning("Define a valid hierarchy structure in '2. Hierarchy' tab first.")
        if not llm_ready:
            st.warning("LLM Client is not configured. Please set it up in the sidebar.")
        if not llm_prediction_data_ready:
            st.warning("Upload data to classify and select the text column in '1. Data Setup'.")

        # --- Classification Button ---
        if st.button("Classify using LLM", type="primary", disabled=not llm_ready_for_classification):
            st.session_state.classification_just_completed = False # Reset flag

            # Build hierarchy from the DataFrame stored in session state (created by editor)
            final_hierarchy = utils.build_hierarchy_from_df(st.session_state.get('hierarchy_df'))

            # Validate hierarchy structure before proceeding
            if final_hierarchy and final_hierarchy.get('themes'): # Basic check for themes
                uncat_df = st.session_state.uncategorized_df
                text_col = st.session_state.uncat_text_col

                if text_col and text_col in uncat_df.columns:
                    with st.spinner("🧠 Classifying texts using the LLM... This might take some time depending on data size and model."):
                        try:
                            # Call the LLM classification function
                            results_llm_df = llm_classifier.classify_texts_with_llm(
                                uncat_df,
                                text_col,
                                final_hierarchy,
                                st.session_state.llm_client
                            )

                            if results_llm_df is not None:
                                st.session_state.results_df = results_llm_df
                                st.session_state.app_stage = 'categorized'
                                st.session_state.classification_just_completed = True # Set flag
                                st.success("✅ LLM Classification complete!")
                                st.rerun() # Go to results tab / show completion message
                            else:
                                st.error("❌ LLM Classification failed. The process returned no results. Check LLM configuration or input data.")
                        except Exception as e:
                            st.error(f"An error occurred during LLM classification: {e}")
                            traceback.print_exc()
                else:
                    st.error(f"Selected prediction column '{text_col}' not found in the uploaded data.")
            else:
                st.error("Cannot run LLM classification: The defined hierarchy is invalid or empty. Please check '2. Hierarchy' tab.")

        elif not llm_ready_for_classification:
            st.info("Please complete all setup steps (LLM client, hierarchy, data upload) to enable LLM classification.")


# === Tab 4: Results ===
with tab_results:
    st.header("4. View Results")

    # Check if results exist in session state
    results_df = st.session_state.get('results_df')

    if results_df is not None:
        # Display the results DataFrame
        st.markdown("### Classification Results")
        # Use a copy to avoid modifying the original DataFrame in session state via display
        results_display_df = results_df.copy()
        st.dataframe(results_display_df, use_container_width=True)

        # --- Download Buttons ---
        st.markdown("### Download Results")
        col_dl_csv, col_dl_excel = st.columns(2)
        with col_dl_csv:
            try:
                csv_data = results_display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name='classification_results.csv',
                    mime='text/csv',
                    key='download_csv_button'
                )
            except Exception as e:
                st.error(f"Error generating CSV: {e}")
        with col_dl_excel:
            try:
                excel_data = utils.df_to_excel_bytes(results_display_df)
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name='classification_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download_excel_button'
                )
            except Exception as e:
                st.error(f"Error generating Excel: {e}")

        st.divider()

        # --- Workflow-Specific Summaries ---

        # Hugging Face Stats
        if selected_workflow == "Hugging Face Model":
            st.subheader("📊 Hugging Face Classification Stats")
            # Check if the necessary raw labels exist for stats calculation
            if st.session_state.get('raw_predicted_labels') is not None:
                try:
                    # Assuming display_hierarchical_stats works correctly with the df and prefix
                    utils.display_hierarchical_stats(results_display_df, prefix="HF_")
                except Exception as e:
                    st.error(f"An error occurred generating Hugging Face stats: {e}")
                    traceback.print_exc()
            else:
                st.info("Raw prediction labels not found, cannot display detailed HF stats.")

        # LLM Summary
        elif selected_workflow == "LLM Categorization":
            st.subheader("📊 LLM Classification Summary")
            try:
                total_rows = len(results_display_df)
                theme_col = 'LLM_Theme'
                category_col = 'LLM_Category'
                reasoning_col = 'LLM_Reasoning' # Often contains error messages

                # Calculate categorized rows (non-empty Theme/Category and not 'Error' in Theme)
                categorized_count = 0
                if theme_col in results_display_df.columns and category_col in results_display_df.columns:
                     categorized_mask = (
                         (results_display_df[theme_col].notna() | results_display_df[category_col].notna()) &
                         (results_display_df[theme_col] != 'Error') # Exclude rows marked as Error in Theme
                     )
                     categorized_count = results_display_df[categorized_mask].shape[0]
                elif theme_col in results_display_df.columns: # Fallback if only Theme exists
                     categorized_mask = (results_display_df[theme_col].notna() & (results_display_df[theme_col] != 'Error'))
                     categorized_count = results_display_df[categorized_mask].shape[0]


                # Calculate error rows more robustly
                error_count = 0
                error_indices = set()
                if theme_col in results_display_df.columns:
                    error_indices.update(results_display_df[results_display_df[theme_col] == 'Error'].index)
                if reasoning_col in results_display_df.columns:
                     # Check for "Error:" pattern in reasoning, especially when Theme might be NaN
                     reasoning_errors = results_display_df[
                         results_display_df[reasoning_col].astype(str).str.contains("Error:", na=False)
                     ].index
                     error_indices.update(reasoning_errors)
                error_count = len(error_indices)

                # Display Metrics
                col_met_total, col_met_cat, col_met_err = st.columns(3)
                with col_met_total:
                    st.metric("Total Rows Processed", f"{total_rows:,}")
                with col_met_cat:
                    st.metric("Successfully Categorized", f"{categorized_count:,}")
                with col_met_err:
                    if error_count > 0:
                        st.metric("Rows with Errors", f"{error_count:,}", delta=f"{error_count}", delta_color="inverse")
                    else:
                        st.metric("Rows with Errors", "0")


                # Display Theme Counts (if Theme column exists)
                if theme_col in results_display_df.columns:
                    st.markdown("**Theme Distribution:**")
                    theme_counts = results_display_df[theme_col].value_counts().reset_index()
                    theme_counts.columns = ['Theme', 'Count']
                    st.dataframe(theme_counts, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"An error occurred generating the LLM summary: {e}")
                traceback.print_exc()

    else: # No results_df found
        st.info("No classification results to display. Please run a classification in '3. Classification' tab.")

# --- Footer ---
st.sidebar.divider()
# Consider making the version dynamic if possible, e.g., reading from a file or config
APP_VERSION = "1.7-refactored"
st.sidebar.caption(f"AI Classifier App v{APP_VERSION}")

# --- Entry Point ---
# Standard practice for Python scripts, though less critical for Streamlit apps
# which are typically run via `streamlit run app.py`
if __name__ == "__main__":
    # Potential place for setup code if needed before Streamlit runs,
    # but most logic is handled within the Streamlit execution flow.
    pass # Keep pass if no specific main execution logic is needed
