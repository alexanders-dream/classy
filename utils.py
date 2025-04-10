# utils.py
"""General utility functions."""

import streamlit as st
import pandas as pd
from io import BytesIO
import traceback
import config # Import your config file
from typing import List, Dict, Any, Tuple, Optional # <-- Ensure List, Tuple, Optional are imported
from collections import defaultdict # <-- Ensure defaultdict is imported

# --- Session State Management ---
def init_session_state():
    """Initializes Streamlit session state variables."""
    # Workflow selection
    if 'selected_workflow' not in st.session_state:
        st.session_state.selected_workflow = "LLM Categorization" # Default workflow

    # --- DataFrames ---
    if 'categorized_df' not in st.session_state:
        st.session_state.categorized_df = None # Training data for HF
    if 'uncategorized_df' not in st.session_state:
        st.session_state.uncategorized_df = None # Data to predict on
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None # Stores final classification output

    # --- Column Selections ---
    if 'cat_text_col' not in st.session_state:
        st.session_state.cat_text_col = None # Text col for HF training
    for level in config.HIERARCHY_LEVELS: # HF hierarchy cols
        if f'cat_{level.lower()}_col' not in st.session_state:
            st.session_state[f'cat_{level.lower()}_col'] = None
    if 'uncat_text_col' not in st.session_state:
        st.session_state.uncat_text_col = None # Text col for prediction (both workflows)

    # --- Hugging Face Model State ---
    if 'hf_model' not in st.session_state:
        st.session_state.hf_model = None
    if 'hf_tokenizer' not in st.session_state:
        st.session_state.hf_tokenizer = None
    if 'hf_label_map' not in st.session_state:
        st.session_state.hf_label_map = None
    if 'hf_rules' not in st.session_state:
        st.session_state.hf_rules = pd.DataFrame(columns=['Label', 'Keywords', 'Confidence Threshold'])
    if 'hf_model_ready' not in st.session_state:
        st.session_state.hf_model_ready = False # Unified flag: True if HF trained or loaded

    # --- LLM State ---
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = None # The initialized LangChain client
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = config.SUPPORTED_PROVIDERS[0] # Default to first provider
    if 'llm_endpoint' not in st.session_state:
        st.session_state.llm_endpoint = "" # Will be set based on provider
    if 'llm_api_key' not in st.session_state:
        st.session_state.llm_api_key = "" # Store API key temporarily
    if 'llm_models' not in st.session_state:
        st.session_state.llm_models = [] # List of available models for selected provider
    if 'llm_selected_model_name' not in st.session_state:
        st.session_state.llm_selected_model_name = None

    # --- Hierarchy State (Used by both, but defined/edited primarily for LLM) ---
    if 'hierarchy_df' not in st.session_state:
        # Correct the column name here during initialization
        st.session_state.hierarchy_df = pd.DataFrame(columns=['Theme', 'Category', 'Segment', 'Subsegment', 'Keywords'])
    if 'hierarchy_defined' not in st.session_state:
        st.session_state.hierarchy_defined = False # Is the hierarchy valid?
    if 'ai_suggestion_pending' not in st.session_state: # For AI suggested hierarchy
        st.session_state.ai_suggestion_pending = None

    # --- App State ---
    if 'app_stage' not in st.session_state:
        st.session_state.app_stage = 'init' # Stages: init, file_uploaded, column_selected, hf_model_ready, hierarchy_defined, categorized
    if 'raw_predicted_labels' not in st.session_state: # Used by HF stats
        st.session_state.raw_predicted_labels = None

    # File upload keys
    if 'categorized_file_key' not in st.session_state:
        st.session_state.categorized_file_key = None
    if 'uncategorized_file_key' not in st.session_state:
        st.session_state.uncategorized_file_key = None


def restart_session():
    """Clears relevant parts of the session state to simulate a restart."""
    st.info("Ending session and clearing state...")
    # Keep only essential settings maybe? Or clear all? Let's clear most things.
    keys_to_clear = [
        'categorized_df', 'uncategorized_df', 'results_df',
        'cat_text_col', 'uncat_text_col',
        'hf_model', 'hf_tokenizer', 'hf_label_map', 'hf_rules', 'hf_model_ready',
        'llm_client', 'llm_models', 'llm_selected_model_name', #'llm_provider', 'llm_endpoint', 'llm_api_key', # Keep API config? Maybe clear key.
        'hierarchy_df', 'hierarchy_defined', 'ai_suggestion_pending',
        'app_stage', 'raw_predicted_labels',
        'categorized_file_key', 'uncategorized_file_key'
    ]
    # Clear hierarchy level columns
    for level in config.HIERARCHY_LEVELS:
        keys_to_clear.append(f'cat_{level.lower()}_col')

    # Add session_initialized to keys to clear so it re-runs init on restart
    keys_to_clear.append('session_initialized')

    for key in keys_to_clear:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception as e:
                 st.warning(f"Could not clear session state key '{key}': {e}")


    # No need to call init_session_state() here, it will be called at the top of app.py on rerun
    st.rerun()

# --- Data Handling ---
@st.cache_data # Cache data loading to avoid reloading on every interaction
def load_data(uploaded_file):
    """Loads data from CSV or Excel, handles common errors."""
    if uploaded_file is None:
        return None
    try:
        file_name = uploaded_file.name
        st.info(f"Loading '{file_name}'...")
        if file_name.endswith('.csv'):
            try:
                # Try UTF-8 first, then latin1
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                st.warning("UTF-8 decoding failed, trying latin1...")
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
            except Exception as e_csv:
                 st.error(f"Error reading CSV file: {e_csv}")
                 return None
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl') # Specify engine
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None

        # Basic cleaning: drop fully empty columns/rows
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        st.success(f"✅ Loaded '{file_name}' ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    except Exception as e:
        st.error(f"Error loading file '{uploaded_file.name}': {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_data # Cache conversion
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Converts DataFrame to Excel bytes using xlsxwriter."""
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='ClassificationResults')
        return output.getvalue()
    except ImportError:
        st.info("`xlsxwriter` not found, falling back to `openpyxl`. Install with `pip install XlsxWriter`.")
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='ClassificationResults')
            return output.getvalue()
        except Exception as e_openpyxl:
            st.error(f"Error generating Excel file with openpyxl: {e_openpyxl}")
            return b""
    except Exception as e_xlsx:
        st.error(f"Error generating Excel file with xlsxwriter: {e_xlsx}")
        return b""

# --- Hierarchy Manipulation ---
def build_hierarchy_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Converts flat hierarchy DataFrame (from editor) back to nested dict.
       Handles potential column name inconsistencies for 'Subsegment'.
    """
    # (Keep the corrected version from the previous response that handles 'Sub-Segment')
    if df is None or df.empty:
        return {'themes': []}

    hierarchy = {'themes': []}
    themes_dict = defaultdict(lambda: {'name': '', 'categories': defaultdict(lambda: {'name': '', 'segments': defaultdict(lambda: {'name': '', 'sub_segments': []})})})

    required_base_cols = ['Theme', 'Category', 'Segment']
    subsegment_key = "Subsegment" # Preferred key
    subsegment_display_key = "Sub-Segment" # Potential display key
    keywords_key = "Keywords"

    df_processed = df.copy()

    actual_subsegment_col = None
    if subsegment_key in df_processed.columns: actual_subsegment_col = subsegment_key
    elif subsegment_display_key in df_processed.columns: actual_subsegment_col = subsegment_display_key
    else: st.error(f"Build Hierarchy Error: Neither '{subsegment_key}' nor '{subsegment_display_key}' found."); return {'themes': []}

    all_expected_cols = required_base_cols + [actual_subsegment_col, keywords_key]

    for col in all_expected_cols:
        if col not in df_processed.columns:
            st.warning(f"Build Hierarchy Warning: Column '{col}' missing. Creating empty."); df_processed[col] = ''
    df_processed = df_processed[all_expected_cols].astype(str).fillna('')

    processed_rows, skipped_rows = 0, 0
    for _, row in df_processed.iterrows():
        theme_name = row['Theme'].strip()
        cat_name = row['Category'].strip()
        seg_name = row['Segment'].strip()
        sub_seg_name = row[actual_subsegment_col].strip()

        if not all([theme_name, cat_name, seg_name, sub_seg_name]): skipped_rows += 1; continue

        keywords = [k.strip() for k in row.get(keywords_key, '').split(',') if k.strip()]

        themes_dict[theme_name]['name'] = theme_name
        categories_dict = themes_dict[theme_name]['categories']
        categories_dict[cat_name]['name'] = cat_name
        segments_dict = categories_dict[cat_name]['segments']
        segments_dict[seg_name]['name'] = seg_name

        sub_segments_list = segments_dict[seg_name]['sub_segments']
        if not any(ss['name'] == sub_seg_name for ss in sub_segments_list):
             sub_segments_list.append({'name': sub_seg_name, 'keywords': keywords})
        processed_rows += 1

    if skipped_rows > 0: st.info(f"Build Hierarchy: Skipped {skipped_rows} rows due to missing path names.")

    final_themes = []
    for theme_name, theme_data in themes_dict.items():
        final_categories = []
        for cat_name, cat_data in theme_data['categories'].items():
            final_segments = []
            for seg_name, seg_data in cat_data['segments'].items():
                 if seg_data['sub_segments']: final_segments.append({'name': seg_data['name'], 'sub_segments': seg_data['sub_segments']})
            if final_segments: final_categories.append({'name': cat_data['name'], 'segments': final_segments})
        if final_categories: final_themes.append({'name': theme_data['name'], 'categories': final_categories})

    final_hierarchy = {'themes': final_themes}
    if not final_themes and processed_rows > 0: st.warning("Build Hierarchy: Processed rows, but result is empty.")
    return final_hierarchy


def flatten_hierarchy(nested_hierarchy: Dict[str, Any]) -> pd.DataFrame:
    """Converts AI-generated nested hierarchy dict to a flat DataFrame for the editor."""
    # (Keep the corrected version from the previous response)
    rows = []
    required_cols = ['Theme', 'Category', 'Segment', 'Subsegment', 'Keywords'] # Use consistent key

    if not nested_hierarchy or 'themes' not in nested_hierarchy:
        return pd.DataFrame(columns=required_cols)

    try:
        for theme in nested_hierarchy.get('themes', []):
            theme_name = theme.get('name', '').strip()
            if not theme_name: continue

            for category in theme.get('categories', []):
                cat_name = category.get('name', '').strip()
                if not cat_name: continue

                for segment in category.get('segments', []):
                    seg_name = segment.get('name', '').strip()
                    if not seg_name: continue

                    if not segment.get('sub_segments'): continue # Skip segments without subsegments
                    else:
                        for sub_segment in segment.get('sub_segments', []):
                            sub_seg_name = sub_segment.get('name', '').strip()
                            if not sub_seg_name: continue

                            keywords_list = [str(k).strip() for k in sub_segment.get('keywords', []) if str(k).strip()]
                            keywords_str = ', '.join(keywords_list)

                            rows.append({
                                'Theme': theme_name,
                                'Category': cat_name,
                                'Segment': seg_name,
                                'Subsegment': sub_seg_name, # Use consistent key
                                'Keywords': keywords_str
                            })
    except Exception as e:
        st.error(f"Error during hierarchy flattening: {e}")
        st.error(traceback.format_exc())
        return pd.DataFrame(columns=required_cols)

    return pd.DataFrame(rows, columns=required_cols)


# --- ADDED function ---
def parse_predicted_labels_to_columns(predicted_labels_list: List[List[str]]) -> List[Dict[str, str | None]]:
    """Parses prefixed labels (e.g., 'Theme: X') into structured dictionaries."""
    structured_results = []
    # Use HIERARCHY_LEVELS from config
    prefixes = {level: f"{level.lower()}:" for level in config.HIERARCHY_LEVELS} # Use lowercase for matching

    for labels in predicted_labels_list:
        row_dict = {key: None for key in config.HIERARCHY_LEVELS} # Initialize with None for each level
        if not labels:
            structured_results.append(row_dict)
            continue

        found_labels = {key: [] for key in config.HIERARCHY_LEVELS}

        for label in labels:
            if not isinstance(label, str): continue # Skip non-string labels
            label_lower = label.lower()
            label_processed = False
            # Check against prefixes in defined hierarchy order
            for level in config.HIERARCHY_LEVELS:
                prefix_lower = prefixes[level]
                if label_lower.startswith(prefix_lower):
                    # Extract value after the prefix "Level: "
                    value = label[len(level) + 2:].strip() # Length of "Level" + ": "
                    if value:
                        found_labels[level].append(value)
                    label_processed = True
                    break # Assume one label belongs to only one hierarchy level type

        # Assign the first found label for each level to the row dictionary
        for level in config.HIERARCHY_LEVELS:
            if found_labels[level]:
                row_dict[level] = found_labels[level][0]

        structured_results.append(row_dict)

    return structured_results
# --- End of added function ---


# --- ADDED function for HF Stats ---
# --- *** NEW Hierarchical Statistics Function *** ---
def display_hierarchical_stats(results_df: pd.DataFrame, prefix: str = ""):
    """
    Calculates and displays hierarchical statistics (Theme, Category, Segment, Subsegment).

    Args:
        results_df: DataFrame containing the classification results.
        prefix: The prefix added to the hierarchy column names (e.g., "HF_", "LLM_").
    """
    #st.subheader(f"📊 {prefix.replace('_',' ')}Hierarchical Statistics")

    if results_df is None or results_df.empty:
        st.warning("No results data available to generate statistics.")
        return

    # Define column names based on prefix
    theme_col = f"{prefix}Theme"
    cat_col = f"{prefix}Category"
    seg_col = f"{prefix}Segment"
    subseg_col = f"{prefix}Subsegment" # Assuming Subsegment is the key in df

    hierarchy_cols = [theme_col, cat_col, seg_col, subseg_col]

    # Check if required columns exist
    missing_cols = [col for col in hierarchy_cols if col not in results_df.columns]
    if missing_cols:
        st.error(f"Cannot generate hierarchical stats. Missing columns: {', '.join(missing_cols)}")
        return

    total_rows = len(results_df)
    st.caption(f"Based on {total_rows:,} processed rows.")

    # 1. Theme Distribution
    st.markdown("#### Theme Distribution")
    theme_counts = results_df[theme_col].value_counts(dropna=True)
    if not theme_counts.empty:
        # Simple Bar Chart for Themes
        st.bar_chart(theme_counts)
        with st.expander("View Theme Counts Table"):
            st.dataframe(theme_counts.reset_index().rename(columns={'index':'Theme', theme_col:'Count'}), use_container_width=True)
    else:
        st.info("No Themes were assigned.")

    st.markdown("---")

    """
    # 2. Category Distribution (Nested under Theme)
    st.markdown("#### Category Distribution (within Themes)")
    # Group by Theme, then count Categories within each Theme
    cat_grouped = results_df.dropna(subset=[theme_col, cat_col]).groupby([theme_col])[cat_col].value_counts()
    if not cat_grouped.empty:
        # Display using expanders for each theme
        unique_themes = results_df[theme_col].dropna().unique()
        for theme in sorted(unique_themes):
            with st.expander(f"Categories within Theme: **{theme}**"):
                theme_cat_counts = cat_grouped.get(theme)
                if theme_cat_counts is not None and not theme_cat_counts.empty:
                    st.dataframe(theme_cat_counts.reset_index(name='Count'), use_container_width=True)
                else:
                    st.caption(f"No Categories assigned under Theme '{theme}'.")
    else:
        st.info("No Category assignments found within Themes.")

    st.markdown("---")

    # 3. Segment Distribution (Nested under Theme & Category)
    st.markdown("#### Segment Distribution (within Categories)")
    seg_grouped = results_df.dropna(subset=[theme_col, cat_col, seg_col]).groupby([theme_col, cat_col])[seg_col].value_counts()
    if not seg_grouped.empty:
        # Further nesting expanders can get deep, maybe just show table or top N
        st.caption("Showing counts grouped by Theme and Category.")
        st.dataframe(seg_grouped.reset_index(name='Count'), use_container_width=True, height=300)
    else:
        st.info("No Segment assignments found within Categories.")
    """

    st.markdown("---")