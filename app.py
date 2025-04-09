import streamlit as st
import pandas as pd
import os
import traceback

# Import modules
import config
import utils # Ensure utils is imported
import ui_components
import hf_classifier
import llm_classifier

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Text Classifier",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv() # Load variables from .env file

# --- Initialize Session State ---
# This needs to run very first
if 'session_initialized' not in st.session_state:
    utils.init_session_state()
    st.session_state.session_initialized = True

# --- Main App Title ---
st.title("🏷️ AI Hierarchical Text Classifier")
st.markdown("Classify text data using either Hugging Face models or Large Language Models (LLMs).")

# --- Workflow Selection ---
st.sidebar.title("🛠️ Workflow")
workflow_options = ["LLM Categorization", "Hugging Face Model"]
selected_workflow = st.sidebar.radio(
    "Choose Classification Method:",
    options=workflow_options,
    key='selected_workflow', # Use key from init_session_state
    horizontal=True,
)
st.sidebar.markdown("---") # Separator

# --- Display LLM Sidebar (if LLM workflow selected) ---
if selected_workflow == "LLM Categorization":
    ui_components.display_llm_sidebar()
    llm_ready = st.session_state.get('llm_client') is not None
else:
    llm_ready = False # LLM not needed for HF workflow

# --- Main Application Tabs ---
tab_setup, tab_hierarchy, tab_classify, tab_results = st.tabs([
    "1. Data Setup",
    "2. Hierarchy Definition",
    "3. Run Classification",
    "4. View Results"
])


# === Tab 1: Data Setup ===
with tab_setup:
    st.header("1. Data Upload and Column Selection")
    st.markdown("Upload your data and select the relevant text column.")

    col_setup_1, col_setup_2 = st.columns(2)

    # --- Data for Classification (Prediction) ---
    with col_setup_1:
        st.subheader("Data to Classify")
        help_pred = "Upload the CSV/Excel file containing the text you want to categorize."
        uncategorized_file = st.file_uploader("Upload Unlabeled CSV/Excel", type=['csv','xlsx','xls'], key="uncat_uploader_main", help=help_pred)

        if uncategorized_file is not None and uncategorized_file.file_id != st.session_state.get('uncategorized_file_key'):
            with st.spinner("Loading prediction data..."):
                st.session_state.uncategorized_df = utils.load_data(uncategorized_file)
            st.session_state.uncategorized_file_key = uncategorized_file.file_id
            st.session_state.uncat_text_col = None # Reset column selection
            st.session_state.results_df = None # Clear previous results
            st.session_state.app_stage = 'file_uploaded' # Update stage
            st.rerun()

        if st.session_state.uncategorized_df is not None:
            st.success(f"Prediction data loaded ({st.session_state.uncategorized_df.shape[0]} rows).")
            with st.expander("Preview Prediction Data (First 5 Rows)"):
                st.dataframe(st.session_state.uncategorized_df.head(), use_container_width=True)

            # --- Select Text Column for Prediction ---
            st.markdown("**Select Text Column to Classify:**")
            df_uncat = st.session_state.uncategorized_df
            # Ensure columns are strings for consistency
            df_uncat.columns = df_uncat.columns.astype(str)
            cols_uncat = [""] + df_uncat.columns.tolist()
            current_uncat_col = st.session_state.uncat_text_col
            default_uncat_idx = 0
            try:
                 if current_uncat_col in cols_uncat: default_uncat_idx = cols_uncat.index(current_uncat_col)
                 # If the saved column is no longer valid, reset to 0
                 elif current_uncat_col is not None: default_uncat_idx = 0; st.session_state.uncat_text_col = None
            except ValueError: pass # Keep default 0 if not found

            uncat_text_col_select = st.selectbox(
                "Select column:",
                options=cols_uncat,
                index=default_uncat_idx,
                key="uncat_text_select_main",
                label_visibility="collapsed"
            )

            if uncat_text_col_select and uncat_text_col_select != st.session_state.uncat_text_col:
                st.session_state.uncat_text_col = uncat_text_col_select
                st.session_state.app_stage = 'column_selected' # Update stage
                st.rerun()
            elif uncat_text_col_select:
                 st.caption(f"Using column: **'{uncat_text_col_select}'**")

        else:
            st.info("Upload the data file you want to classify.")

    # --- Data for Training (Hugging Face Workflow Only) ---
    with col_setup_2:
        if selected_workflow == "Hugging Face Model":
            st.subheader("Training Data (for HF Model)")
            help_train = "Required for HF Workflow: Upload labeled data with text and hierarchy columns."
            categorized_file = st.file_uploader("Upload Labeled CSV/Excel", type=['csv','xlsx','xls'], key="cat_uploader_main", help=help_train)

            if categorized_file is not None and categorized_file.file_id != st.session_state.get('categorized_file_key'):
                with st.spinner("Loading training data..."):
                    st.session_state.categorized_df = utils.load_data(categorized_file)
                st.session_state.categorized_file_key = categorized_file.file_id
                # Reset HF specific state
                st.session_state.cat_text_col = None
                for level in config.HIERARCHY_LEVELS: st.session_state[f'cat_{level.lower()}_col'] = None
                st.session_state.hf_model_ready = False
                st.session_state.hf_model = None
                st.session_state.hf_tokenizer = None
                st.session_state.hf_label_map = None
                st.session_state.hf_rules = pd.DataFrame(columns=['Label', 'Keywords', 'Confidence Threshold', 'Custom Keywords'])
                st.rerun()

            if st.session_state.categorized_df is not None:
                st.success(f"Training data loaded ({st.session_state.categorized_df.shape[0]} rows).")
                with st.expander("Preview Training Data (First 5 Rows)"):
                    st.dataframe(st.session_state.categorized_df.head(), use_container_width=True)

                # --- Select Columns for HF Training ---
                with st.form("hf_column_selection_form"): # Use a form to confirm selections together
                    st.markdown("**Select Columns for HF Training:**")
                    df_cat = st.session_state.categorized_df
                    # Ensure columns are strings
                    df_cat.columns = df_cat.columns.astype(str)
                    available_cols_cat = df_cat.columns.tolist()
                    cols_with_none_cat = ["(None)"] + available_cols_cat

                    # Text Column
                    current_cat_text_val = st.session_state.cat_text_col
                    default_cat_text_idx = 0
                    try:
                         if current_cat_text_val in available_cols_cat: default_cat_text_idx = available_cols_cat.index(current_cat_text_val)
                         elif current_cat_text_val is not None: default_cat_text_idx = 0; st.session_state.cat_text_col = None # Reset if invalid
                    except ValueError: pass

                    cat_text_col_select = st.selectbox(
                        "Text Column (Training):", available_cols_cat,
                        index=default_cat_text_idx, key="cat_text_select_main"
                    )

                    # Hierarchy Columns
                    st.markdown("Hierarchy Columns (Training):")
                    selected_hierarchy_hf = {}
                    for level in config.HIERARCHY_LEVELS:
                        level_key = f'cat_{level.lower()}_col'
                        current_val = st.session_state.get(level_key, "(None)")
                        options_for_level = ["(None)"] + [c for c in available_cols_cat if c != cat_text_col_select]
                        try: idx = options_for_level.index(current_val) if current_val in options_for_level else 0
                        except ValueError: idx = 0
                        selected_hierarchy_hf[level] = st.selectbox(
                            f"{level} Column:", options_for_level, index=idx, key=f"{level_key}_select_main"
                        )

                    submitted_hf_cols = st.form_submit_button("Confirm HF Training Columns")

                    if submitted_hf_cols:
                        active_selections_hf = {lvl: col for lvl, col in selected_hierarchy_hf.items() if col and col != "(None)"}
                        if not cat_text_col_select: st.warning("Select Text Column for Training.")
                        elif not active_selections_hf: st.warning("Select at least one Hierarchy Column for Training.")
                        elif len(list(active_selections_hf.values())) != len(set(list(active_selections_hf.values()))): st.warning("Cannot use same column for multiple hierarchy levels.")
                        else:
                            st.session_state.cat_text_col = cat_text_col_select
                            for level, col in selected_hierarchy_hf.items():
                                st.session_state[f'cat_{level.lower()}_col'] = col
                            st.success("HF training columns confirmed.")
                            # Optionally trigger rerun if needed, but maybe not necessary just for selection confirmation
                            # st.rerun()

            else:
                st.info("Upload labeled training data if using the Hugging Face workflow.")

        else:
            st.info("Training data upload is only required for the 'Hugging Face Model' workflow.")


# === Tab 2: Hierarchy Definition ===
with tab_hierarchy:
    st.header("2. Define Classification Hierarchy")

    if selected_workflow == "LLM Categorization":
        st.markdown("Define the structure the LLM will use for categorization. You can start manually, use AI suggestion, or refine an existing structure.")

        # --- AI Suggestion Section (LLM Workflow) ---
        st.subheader("🤖 AI Hierarchy Suggestion")
        suggestion_possible = (st.session_state.get('uncategorized_df') is not None and
                              st.session_state.get('uncat_text_col'))

        if suggestion_possible:
            data_col_name = st.session_state.uncat_text_col
            slider_enabled = False # Default to disabled
            max_slider_val = config.MIN_LLM_SAMPLE_SIZE
            default_slider_val = config.MIN_LLM_SAMPLE_SIZE
            try:
                 if data_col_name in st.session_state.uncategorized_df.columns:
                    # Calculate counts safely, handling potential non-numeric errors if column was mixed type
                    unique_texts = pd.to_numeric(st.session_state.uncategorized_df[data_col_name], errors='coerce').dropna().astype(str).unique()
                    unique_text_count = len(unique_texts) # Count after cleaning

                    min_samples_needed = config.MIN_LLM_SAMPLE_SIZE
                    max_slider_val = min(config.MAX_LLM_SAMPLE_SIZE, unique_text_count) if unique_text_count >= min_samples_needed else min_samples_needed
                    default_slider_val = min(config.DEFAULT_LLM_SAMPLE_SIZE, max_slider_val)
                    slider_enabled = llm_ready and (unique_text_count >= min_samples_needed)
                    if not slider_enabled and llm_ready: st.warning(f"Need at least {min_samples_needed} unique texts for suggestion.")
                 else: st.warning(f"Column '{data_col_name}' not found.")
            except Exception as e: st.error(f"Error preparing suggestion: {e}")

            sample_size = st.slider(
                f"Samples from '{data_col_name}' for AI analysis:",
                min_value=config.MIN_LLM_SAMPLE_SIZE, max_value=max_slider_val, value=default_slider_val, step=50,
                key="ai_sample_slider_main", help="More samples -> more context -> longer time.", disabled=not slider_enabled
            )

            if st.button("🚀 Generate Suggestion", key="generate_ai_hierarchy_main", type="primary", disabled=not slider_enabled):
                if llm_ready:
                    st.session_state.ai_suggestion_pending = None
                    sample_texts_series = st.session_state.uncategorized_df[data_col_name].dropna().astype(str)
                    unique_texts_list = sample_texts_series.unique().tolist() # Work with unique list
                    if len(unique_texts_list) >= config.MIN_LLM_SAMPLE_SIZE:
                        actual_sample_size = min(len(unique_texts_list), sample_size)
                        # Sample from the unique list
                        sampled_list = pd.Series(unique_texts_list).sample(actual_sample_size, random_state=42).tolist()

                        suggestion = llm_classifier.generate_hierarchy_suggestion(st.session_state.llm_client, sampled_list)
                        if suggestion: st.session_state.ai_suggestion_pending = suggestion; st.success("✅ AI suggestion generated!")
                        else: st.error("❌ Failed to generate AI suggestion.")
                        st.rerun()
                    else: st.warning(f"Need at least {config.MIN_LLM_SAMPLE_SIZE} unique texts.")
                else: st.error("LLM Client not ready.")
        else: st.info("Upload prediction data and select text column in Tab 1.")
        st.divider()

        # --- Hierarchy Editor (LLM Workflow) ---
        st.subheader("✏️ Hierarchy Editor")
        hierarchy_valid = ui_components.display_hierarchy_editor(key_prefix="llm")

    elif selected_workflow == "Hugging Face Model":
        st.info("Hierarchy for HF model is defined by columns selected in Tab 1.")
        st.markdown("This tab mainly defines the hierarchy for the **LLM** workflow.")
        active_hf_cols = {lvl: st.session_state.get(f'cat_{lvl.lower()}_col') for lvl in config.HIERARCHY_LEVELS if st.session_state.get(f'cat_{lvl.lower()}_col') and st.session_state.get(f'cat_{lvl.lower()}_col') != "(None)"}
        if active_hf_cols:
            st.write("**HF Hierarchy Columns:**"); [st.write(f"- **{lvl}:** '{col}'") for lvl, col in active_hf_cols.items()]
        else: st.warning("Select Training Hierarchy columns in Tab 1 for HF.")

# === Tab 3: Run Classification ===
with tab_classify:
    st.header("3. Run Classification")

    # --- Hugging Face Workflow: Training & Loading ---
    if selected_workflow == "Hugging Face Model":
        st.subheader("Train or Load Hugging Face Model")

        hf_training_cols_ready = (st.session_state.get('cat_text_col') and
                                   any(st.session_state.get(f'cat_{level.lower()}_col') and st.session_state.get(f'cat_{level.lower()}_col') != "(None)" for level in config.HIERARCHY_LEVELS))
        hf_training_data_ready = st.session_state.get('categorized_df') is not None
        hf_train_ready = hf_training_data_ready and hf_training_cols_ready

        with st.container(border=True):
            st.markdown("**Option A: Train New HF Model**")
            if not hf_train_ready: st.warning("Upload Training Data & confirm columns in Tab 1.")

            col1_hf, col2_hf, col3_hf = st.columns(3)
            train_button_disabled = not hf_train_ready or st.session_state.hf_model_ready
            with col1_hf:
                hf_model_choice = st.selectbox("Base Model (HF):", ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"], 0, disabled=train_button_disabled, key="hf_model_select")
            with col2_hf:
                hf_num_epochs = st.slider("Epochs (HF):", 1, 10, 3, 1, disabled=train_button_disabled, key="hf_epochs_slider")
            with col3_hf:
                validation_split = st.slider("Validation Split (%):", 5, 50, int(config.DEFAULT_VALIDATION_SPLIT * 100), 5, help="Data held out for validation.", disabled=train_button_disabled, key="hf_val_split_slider")
                validation_split_ratio = validation_split / 100.0

            if st.button("🚀 Start HF Model Training", type="primary", disabled=train_button_disabled):
                if st.session_state.hf_model_ready: st.warning("HF model already active.")
                else:
                    # Ensure columns selected are valid before proceeding
                    hierarchy_cols_map = {level: st.session_state.get(f'cat_{level.lower()}_col') for level in config.HIERARCHY_LEVELS}
                    active_hf_cols = {lvl: col for lvl, col in hierarchy_cols_map.items() if col and col != "(None)"}
                    if not st.session_state.cat_text_col or not active_hf_cols:
                         st.error("Cannot train: HF Training Text or Hierarchy columns not confirmed in Tab 1.")
                    else:
                        prep_texts, prep_labels = hf_classifier.prepare_hierarchical_training_data(
                            st.session_state.categorized_df, st.session_state.cat_text_col, hierarchy_cols_map)
                        if prep_texts is not None and prep_labels is not None:
                            st.session_state.hf_model, st.session_state.hf_tokenizer, st.session_state.hf_label_map, st.session_state.hf_rules = None, None, None, pd.DataFrame()
                            model, tokenizer, label_map, rules = hf_classifier.train_hf_model(
                                all_train_texts=prep_texts, all_train_labels_list=prep_labels,
                                model_choice=hf_model_choice, num_epochs=hf_num_epochs,
                                validation_split_ratio=validation_split_ratio)
                            if model and tokenizer and label_map is not None:
                                st.session_state.hf_model, st.session_state.hf_tokenizer, st.session_state.hf_label_map, st.session_state.hf_rules = model, tokenizer, label_map, rules
                                st.session_state.hf_model_ready = True
                                st.success("✅ HF Model trained!")
                            else: st.error("❌ HF Model training failed.")
                        else: st.error("❌ HF Data prep failed.")

            # Status and Save Button
            if st.session_state.hf_model_ready and st.session_state.hf_model:
                 st.success("HF Model is Trained/Loaded.")
                 if st.button("💾 Save Trained HF Model"):
                     if hf_classifier.save_hf_model_artifacts(st.session_state.hf_model, st.session_state.hf_tokenizer, st.session_state.hf_label_map, st.session_state.hf_rules):
                          st.info(f"HF Model saved to: {os.path.abspath(config.SAVED_HF_MODEL_PATH)}")
                     else: st.error("Failed to save HF model.")
            elif st.session_state.hf_model_ready: st.info("HF Model is loaded.")

        st.markdown("---")
        with st.container(border=True):
            st.markdown("**Option B: Load Existing HF Model**")
            st.caption(f"Path: `{config.SAVED_HF_MODEL_PATH}`")
            if st.button("🔄 Load HF Model", disabled=st.session_state.hf_model_ready):
                if st.session_state.hf_model_ready: st.warning("HF model already active.")
                else:
                    with st.spinner("Loading HF model..."):
                        model, tokenizer, label_map, rules = hf_classifier.load_hf_model_artifacts()
                    if model and tokenizer and label_map is not None:
                        st.session_state.hf_model, st.session_state.hf_tokenizer, st.session_state.hf_label_map, st.session_state.hf_rules = model, tokenizer, label_map, rules
                        st.session_state.hf_model_ready = True
                        st.success("✅ HF Model loaded!")
                    else: st.error("❌ Failed to load HF artifacts.")
            elif st.session_state.hf_model_ready: st.info("HF Model already active.")

        st.markdown("---")
        # --- HF Rule Editor ---
        st.subheader("Review & Edit HF Rules")
        if st.session_state.hf_model_ready and st.session_state.hf_rules is not None and not st.session_state.hf_rules.empty:
             if 'Label' in st.session_state.hf_rules.columns and 'Confidence Threshold' in st.session_state.hf_rules.columns:
                  st.info("Edit thresholds and add comma-sep custom keywords (case-insensitive) to force labels.")
                  if 'Custom Keywords' not in st.session_state.hf_rules.columns: st.session_state.hf_rules['Custom Keywords'] = ''
                  st.session_state.hf_rules['Custom Keywords'] = st.session_state.hf_rules['Custom Keywords'].fillna('')
                  st.session_state.hf_rules['Confidence Threshold'] = pd.to_numeric(st.session_state.hf_rules['Confidence Threshold'], errors='coerce').fillna(config.DEFAULT_HF_THRESHOLD).clip(0.05, 0.95)

                  edited_hf_rules_df = st.data_editor(
                      st.session_state.hf_rules.sort_values(by='Label'),
                      column_config={
                          "Label": st.column_config.TextColumn(disabled=True),
                          "Keywords": st.column_config.TextColumn("Correlated Keywords", disabled=True),
                          "Confidence Threshold": st.column_config.NumberColumn(min_value=0.05, max_value=0.95, step=0.01, format="%.3f"),
                          "Custom Keywords": st.column_config.TextColumn("Custom Keywords (Force Label)", required=False)
                      },
                      use_container_width=True, hide_index=True, num_rows="dynamic", key="hf_rules_editor"
                  )
                  if isinstance(edited_hf_rules_df, pd.DataFrame):
                      if 'Custom Keywords' not in edited_hf_rules_df.columns: edited_hf_rules_df['Custom Keywords'] = ''
                      edited_hf_rules_df['Custom Keywords'] = edited_hf_rules_df['Custom Keywords'].fillna('')
                      if 'Confidence Threshold' not in edited_hf_rules_df.columns: edited_hf_rules_df['Confidence Threshold'] = config.DEFAULT_HF_THRESHOLD
                      edited_hf_rules_df['Confidence Threshold'] = pd.to_numeric(edited_hf_rules_df['Confidence Threshold'], errors='coerce').fillna(config.DEFAULT_HF_THRESHOLD).clip(0.05, 0.95)

                      # Compare after sorting columns for stability
                      cols_order = sorted([c for c in ['Label', 'Keywords', 'Confidence Threshold', 'Custom Keywords'] if c in st.session_state.hf_rules.columns])
                      # Ensure edited df has the same columns before comparison
                      edited_hf_rules_df_compare = edited_hf_rules_df[cols_order].copy()

                      current_df_sorted = st.session_state.hf_rules[cols_order].reset_index(drop=True).astype(str)
                      edited_df_sorted = edited_hf_rules_df_compare.reset_index(drop=True).astype(str)

                      if not current_df_sorted.equals(edited_df_sorted):
                           st.session_state.hf_rules = edited_hf_rules_df.copy()
                           st.success("✅ HF Rules updated.")

             else: st.warning("Cannot display HF rules: Missing required columns.")
        elif st.session_state.hf_model_ready: st.info("HF Model ready, but no rules. Using default threshold.")
        else: st.info("Train or load HF model to view/edit rules.")

        # --- Run HF Classification ---
        st.divider()
        st.header("🚀 Run Hugging Face Classification")
        hf_classify_ready = (st.session_state.hf_model_ready and
                             st.session_state.uncategorized_df is not None and
                             st.session_state.uncat_text_col is not None)
        if st.button("Classify using HF Model", type="primary", disabled=not hf_classify_ready):
             # Ensure prediction column selection is valid
             if st.session_state.uncat_text_col and st.session_state.uncat_text_col in st.session_state.uncategorized_df.columns:
                 texts_to_classify = st.session_state.uncategorized_df[st.session_state.uncat_text_col].fillna("").astype(str).tolist()
                 if texts_to_classify:
                     raw_labels = hf_classifier.classify_texts_with_hf(
                         texts_to_classify, st.session_state.hf_model, st.session_state.hf_tokenizer,
                         st.session_state.hf_label_map, st.session_state.hf_rules
                     )
                     st.session_state.raw_predicted_labels = raw_labels
                     parsed_structs = utils.parse_predicted_labels_to_columns(raw_labels)
                     results_hf_df = st.session_state.uncategorized_df.copy()
                     parsed_df = pd.DataFrame(parsed_structs, index=results_hf_df.index)
                     for col in config.HIERARCHY_LEVELS: results_hf_df[f"HF_{col}"] = parsed_df.get(col)
                     results_hf_df["HF_Raw_Labels"] = [', '.join(map(str, labels)) if labels else None for labels in raw_labels]
                     st.session_state.results_df = results_hf_df
                     st.session_state.app_stage = 'categorized'
                     st.success("HF Classification Complete! View results in Tab 4.")
                     st.rerun()
                 else: st.warning("No text found in the prediction column.")
             elif st.session_state.uncategorized_df is not None:
                 st.error(f"Selected prediction column '{st.session_state.uncat_text_col}' not found in the prediction data.")
             else:
                 st.error("Prediction data not loaded.") # Should be caught by hf_classify_ready, but for safety

        elif not hf_classify_ready: st.warning("Ensure HF model is ready & prediction data/column are set.")


    # --- LLM Workflow: Classification ---
    elif selected_workflow == "LLM Categorization":
        st.subheader("🚀 Run LLM Classification")
        llm_classify_ready = (llm_ready and
                              st.session_state.get('hierarchy_defined', False) and
                              st.session_state.uncategorized_df is not None and
                              st.session_state.uncat_text_col is not None)

        if not st.session_state.get('hierarchy_defined', False): st.warning("Define valid hierarchy in Tab 2.")
        if not llm_ready: st.warning("LLM Client not ready (check sidebar).")
        if st.session_state.uncategorized_df is None or st.session_state.uncat_text_col is None: st.warning("Upload prediction data & select column in Tab 1.")

        if st.button("Classify using LLM", type="primary", disabled=not llm_classify_ready):
             final_hierarchy = utils.build_hierarchy_from_df(st.session_state.hierarchy_df)
             if final_hierarchy and final_hierarchy.get('themes'):
                if st.session_state.uncat_text_col and st.session_state.uncat_text_col in st.session_state.uncategorized_df.columns:
                     results_llm_df = llm_classifier.classify_texts_with_llm(
                         st.session_state.uncategorized_df, st.session_state.uncat_text_col,
                         final_hierarchy, st.session_state.llm_client )
                     if results_llm_df is not None:
                          st.session_state.results_df = results_llm_df
                          st.session_state.app_stage = 'categorized'
                          st.success("LLM Classification Complete! View results in Tab 4.")
                          st.rerun()
                     else: st.error("LLM Classification failed.")
                else:
                     st.error(f"Selected prediction column '{st.session_state.uncat_text_col}' not found in data.")
             else: st.error("Cannot run LLM: Hierarchy invalid/empty.")
        elif not llm_classify_ready: st.info("Complete setup to enable LLM classification.")


# === Tab 4: Results ===
with tab_results:
    st.header("4. View Results")

    if st.session_state.get('results_df') is not None:
        results_df_copy = st.session_state.results_df.copy()
        st.dataframe(results_df_copy, use_container_width=True)

        col_dl1_res, col_dl2_res = st.columns(2)
        with col_dl1_res:
             csv_data = results_df_copy.to_csv(index=False).encode('utf-8')
             st.download_button("📥 Download Results (CSV)", csv_data, 'classification_results.csv', 'text/csv', key='download-csv-main')
        with col_dl2_res:
             excel_data = utils.df_to_excel_bytes(results_df_copy)
             st.download_button("📥 Download Results (Excel)", excel_data, 'classification_results.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='download-excel-main')

        # Display Stats based on workflow
        st.divider()
        if selected_workflow == "Hugging Face Model" and st.session_state.get('raw_predicted_labels') is not None:
             st.subheader("📊 Hugging Face Classification Stats")
             try: utils.display_classification_stats(results_df_copy, st.session_state.raw_predicted_labels)
             except Exception as e_stats: st.error(f"Error displaying HF stats: {e_stats}")

        elif selected_workflow == "LLM Categorization":
             st.subheader("📊 LLM Classification Summary")
             try:
                  total_rows_llm = len(results_df_copy)
                  llm_theme_col, llm_cat_col, llm_reasoning_col = 'LLM_Theme', 'LLM_Category', 'LLM_Reasoning'
                  categorized_rows_count = 0
                  # Ensure columns exist before filtering
                  if llm_theme_col in results_df_copy.columns and llm_cat_col in results_df_copy.columns:
                     categorized_rows_count = results_df_copy[(results_df_copy[llm_theme_col].notna() | results_df_copy[llm_cat_col].notna()) & (results_df_copy[llm_theme_col] != 'Error')].shape[0]

                  error_rows_count = 0
                  if llm_theme_col in results_df_copy.columns:
                      error_idx = set(results_df_copy[results_df_copy[llm_theme_col] == 'Error'].index)
                      if llm_reasoning_col in results_df_copy.columns:
                           # Add index if theme is NA and reasoning contains "Error:"
                           reasoning_error_idx = set(results_df_copy[(results_df_copy[llm_theme_col].isna()) & (results_df_copy[llm_reasoning_col].astype(str).str.contains("Error:", na=False))].index)
                           error_idx.update(reasoning_error_idx)
                      error_rows_count = len(error_idx)
                  elif llm_reasoning_col in results_df_copy.columns: # Check reasoning if theme column is missing
                       error_rows_count = results_df_copy[results_df_copy[llm_reasoning_col].astype(str).str.contains("Error:", na=False)].shape[0]


                  st.metric("Total Rows Processed", f"{total_rows_llm:,}")
                  st.metric("Rows Successfully Categorized", f"{categorized_rows_count:,}")
                  if error_rows_count > 0: st.metric("Rows with Errors", f"{error_rows_count:,}", delta=f"{error_rows_count}", delta_color="inverse")
                  if llm_theme_col in results_df_copy.columns:
                       st.markdown("**Theme Distribution:**"); theme_counts = results_df_copy[llm_theme_col].value_counts().reset_index(); theme_counts.columns = ['Theme', 'Count']; st.dataframe(theme_counts, use_container_width=True)
             except Exception as e_llm_stats: st.error(f"Could not generate LLM summary: {e_llm_stats}")
    else: st.info("Run classification in Tab 3 to view results.")

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption(f"AI Classifier App v1.5") # Increment version

# --- Entry Point ---
if __name__ == "__main__":
    # Init is handled at the top
    pass