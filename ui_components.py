import streamlit as st
import pandas as pd
import os
import config # Import your config file
import llm_classifier # Import functions from llm_classifier
from utils import build_hierarchy_from_df, restart_session # Import from utils
from typing import Dict, Any
import traceback


def display_llm_sidebar():
    """Displays the sidebar for LLM provider and model configuration."""
    st.sidebar.header("🤖 LLM Configuration")

    # --- Provider Selection ---
    current_provider_index = config.SUPPORTED_PROVIDERS.index(st.session_state.llm_provider) if st.session_state.llm_provider in config.SUPPORTED_PROVIDERS else 0
    provider = st.sidebar.selectbox(
        "Select AI Provider:",
        options=config.SUPPORTED_PROVIDERS,
        index=current_provider_index,
        key="llm_provider_select",
        help="Choose the LLM service to use."
    )
    # Update session state if provider changes
    if provider != st.session_state.llm_provider:
        st.session_state.llm_provider = provider
        st.session_state.llm_models = [] # Reset models on provider change
        st.session_state.llm_selected_model_name = None
        st.session_state.llm_client = None # Reset client
        # Reset endpoint based on new provider
        if provider == "Groq": st.session_state.llm_endpoint = config.DEFAULT_GROQ_ENDPOINT
        elif provider == "Ollama": st.session_state.llm_endpoint = config.DEFAULT_OLLAMA_ENDPOINT
        # Add other providers...
        else: st.session_state.llm_endpoint = ""
        st.rerun() # Rerun to reflect changes

    # --- API Endpoint ---
    default_endpoint = config.DEFAULT_GROQ_ENDPOINT if provider == "Groq" else config.DEFAULT_OLLAMA_ENDPOINT
    # Set endpoint in state if not already set for the current provider
    if 'llm_endpoint' not in st.session_state or not st.session_state.llm_endpoint:
         st.session_state.llm_endpoint = default_endpoint

    endpoint = st.sidebar.text_input(
        "API Endpoint:",
        value=st.session_state.llm_endpoint,
        key="llm_endpoint_input",
        help="Modify if using a non-default endpoint (e.g., custom Ollama URL)."
    )
    if endpoint != st.session_state.llm_endpoint:
         st.session_state.llm_endpoint = endpoint
         st.session_state.llm_models = [] # Reset models if endpoint changes
         st.session_state.llm_selected_model_name = None
         st.session_state.llm_client = None
         st.rerun()


    # --- API Key (Conditional for Groq) ---
    api_key_needed = provider == "Groq" # Add other providers needing keys here
    api_key_present = False
    if api_key_needed:
        # Load from environment or secrets first
        env_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

        # Use session state to allow user input, prioritizing env/secrets
        if env_api_key and not st.session_state.get("llm_api_key_user_override"):
            st.session_state.llm_api_key = env_api_key
            st.sidebar.success("Groq API Key loaded from environment/secrets.", icon="🔒")
            api_key_present = True
        else:
            # Allow user input if not found or overridden
            user_api_key = st.sidebar.text_input(
                f"{provider} API Key:",
                value=st.session_state.get("llm_api_key", ""),
                type="password",
                key="llm_api_key_input",
                help=f"Required for {provider}. Enter your key here.",
                placeholder=f"Enter your {provider} API key"
            )
            if user_api_key:
                st.session_state.llm_api_key = user_api_key
                st.session_state.llm_api_key_user_override = True # Flag that user provided it
                api_key_present = True
            else:
                st.session_state.llm_api_key = "" # Ensure it's empty if user clears it
                st.session_state.llm_api_key_user_override = False
                st.sidebar.warning(f"{provider} API Key is required to fetch models and run classification.")
                if provider == "Groq": st.sidebar.markdown("[Get Groq API Key](https://console.groq.com/keys)")

    else: # Provider doesn't need a key (like default Ollama)
        st.session_state.llm_api_key = ""
        api_key_present = True # Treat as "present" for logic flow

    # --- Fetch Models ---
    # Fetch if models are empty or if provider/endpoint/key status changed relevantly
    should_fetch = not st.session_state.llm_models
    if provider == "Groq" and not api_key_present:
        should_fetch = False # Don't try fetching Groq models without key
        if st.session_state.llm_models: st.session_state.llm_models = [] # Clear stale models

    if should_fetch and st.session_state.llm_endpoint:
        models = llm_classifier.fetch_available_models(provider, st.session_state.llm_endpoint, st.session_state.llm_api_key if api_key_needed else None)
        if models:
            st.session_state.llm_models = models
            # Try setting a default model if none is selected
            if not st.session_state.get('llm_selected_model_name'):
                default_model = config.DEFAULT_GROQ_MODEL if provider=="Groq" else config.DEFAULT_OLLAMA_MODEL
                if default_model in models:
                    st.session_state.llm_selected_model_name = default_model
                elif models: # Fallback to first available model
                    st.session_state.llm_selected_model_name = models[0]
        else:
            st.sidebar.error(f"Could not fetch models for {provider}.")
            if provider == "Ollama": st.sidebar.markdown("[Download Ollama](https://ollama.com/)")
        # No automatic rerun here, let user select or refresh

    # --- Model Selection Dropdown ---
    st.sidebar.markdown("---") # Separator
    st.sidebar.markdown("**Model Selection**")

    current_model_name = st.session_state.get('llm_selected_model_name')
    current_model_index = 0
    if current_model_name and st.session_state.llm_models and current_model_name in st.session_state.llm_models:
        current_model_index = st.session_state.llm_models.index(current_model_name)

    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        if st.session_state.llm_models:
            selected_model_name = st.selectbox(
                "Select AI Model:",
                options=st.session_state.llm_models,
                index=current_model_index,
                key="llm_model_select",
                help="Choose from models available at the endpoint."
            )
            if selected_model_name != st.session_state.get('llm_selected_model_name'):
                 st.session_state.llm_selected_model_name = selected_model_name
                 st.session_state.llm_client = None # Reset client when model changes
                 st.rerun()

        else:
            st.error("No models available to select.")
            selected_model_name = None

    # Refresh Button
    with col2:
        # Small refresh button using markdown for styling
        st.markdown("<div style='margin-top: 1.8em;'>", unsafe_allow_html=True) # Adjust vertical alignment
        if st.button("🔄", help="Refresh available models list", key="refresh_models_button"):
            if st.session_state.llm_endpoint and (api_key_present if api_key_needed else True):
                 with st.spinner("Fetching models..."):
                     models = llm_classifier.fetch_available_models(
                         st.session_state.llm_provider,
                         st.session_state.llm_endpoint,
                         st.session_state.llm_api_key if api_key_needed else None
                     )
                 if models:
                     st.session_state.llm_models = models
                     # Reset selection if previous model disappears? Or keep it? Let's keep it for now.
                     st.sidebar.success("Models updated!", icon="✅")
                 else:
                     st.sidebar.error("Failed to fetch.", icon="❗")
                     st.session_state.llm_models = [] # Clear if fetch failed
                 st.rerun() # Rerun to update dropdown
            else:
                st.sidebar.warning("Cannot refresh: Check endpoint and API key (if required).", icon="⚠️")
        st.markdown("</div>", unsafe_allow_html=True)


    # --- Initialize LLM Client ---
    client_ready = False
    if st.session_state.llm_selected_model_name and st.session_state.llm_endpoint and (api_key_present if api_key_needed else True):
         # Only initialize if client is None or config changed
         if st.session_state.llm_client is None:
             st.session_state.llm_client = llm_classifier.initialize_llm_client(
                 st.session_state.llm_provider,
                 st.session_state.llm_endpoint,
                 st.session_state.llm_api_key if api_key_needed else None,
                 st.session_state.llm_selected_model_name
             )
         if st.session_state.llm_client:
             client_ready = True


    # --- Display Status and End Session ---
    st.sidebar.markdown("---")
    if client_ready:
        st.sidebar.info(f"Provider: {st.session_state.llm_provider}\nModel: {st.session_state.llm_selected_model_name}")
    elif st.session_state.llm_selected_model_name:
         st.sidebar.error("LLM Client not ready. Check configuration (endpoint, API key, model selection).")
    else:
         st.sidebar.warning("Select a model to initialize the LLM client.")

    if st.sidebar.button("End Session & Clear State", key="end_session_sidebar_button"):
        restart_session() # Call utility function

def flatten_hierarchy(nested_hierarchy: Dict[str, Any]) -> pd.DataFrame:
    """Converts nested hierarchy dict (from AI) to a flat DataFrame for st.data_editor."""
    rows = []
    required_cols = ['Theme', 'Category', 'Segment', 'Sub-Segment', 'Keywords']

    if not nested_hierarchy or 'themes' not in nested_hierarchy:
        st.warning("Flatten Hierarchy: Input structure is empty or missing 'themes'.")
        return pd.DataFrame(columns=required_cols)

    try:
        for theme in nested_hierarchy.get('themes', []):
            theme_name = theme.get('name', '').strip()
            if not theme_name: continue # Skip themes without names

            for category in theme.get('categories', []):
                cat_name = category.get('name', '').strip()
                if not cat_name: continue # Skip categories without names

                for segment in category.get('segments', []):
                    seg_name = segment.get('name', '').strip()
                    if not seg_name: continue # Skip segments without names

                    if not segment.get('sub_segments'):
                         # Option: Add row with None sub-segment if needed, or just skip
                         #st.debug(f"Segment '{seg_name}' has no sub-segments. Skipping for flat view.")
                         continue
                    else:
                        for sub_segment in segment.get('sub_segments', []):
                            sub_seg_name = sub_segment.get('name', '').strip()
                            if not sub_seg_name: continue # Skip sub-segments without names

                            # Join keywords, ensure they are strings
                            keywords_list = [str(k).strip() for k in sub_segment.get('keywords', []) if str(k).strip()]
                            keywords_str = ', '.join(keywords_list)

                            rows.append({
                                'Theme': theme_name,
                                'Category': cat_name,
                                'Segment': seg_name,
                                'Sub-Segment': sub_seg_name,
                                'Keywords': keywords_str
                            })
    except Exception as e:
        st.error(f"Error during hierarchy flattening: {e}")
        st.error(traceback.format_exc())
        return pd.DataFrame(columns=required_cols) # Return empty on error

    if not rows:
        st.warning("Flatten Hierarchy: No valid paths found in the hierarchy structure.")
        return pd.DataFrame(columns=required_cols)

    return pd.DataFrame(rows)

def display_hierarchy_editor(key_prefix="main"):
    """Displays the hierarchy editor and handles AI suggestion logic."""
    st.markdown("Define the Theme -> Category -> Segment -> Subsegment structure. Keywords should be comma-separated.")

    # --- Handle Pending AI Suggestion ---
    if st.session_state.get('ai_suggestion_pending'):
        st.info("🤖 An AI-generated hierarchy suggestion is ready!")
        nested_suggestion = st.session_state.ai_suggestion_pending
        # Flatten suggestion for display and potential application
        df_suggestion = flatten_hierarchy(nested_suggestion)

        st.markdown("**Preview of AI Suggestion:**")
        st.dataframe(df_suggestion, use_container_width=True, height=200)

        col_apply, col_discard = st.columns(2)
        with col_apply:
            if st.button("✅ Apply Suggestion (Replaces Editor)", key=f"{key_prefix}_apply_ai", type="primary"):
                st.session_state.hierarchy_df = df_suggestion # Replace current df
                st.session_state.ai_suggestion_pending = None # Clear pending status
                st.session_state.hierarchy_defined = True # Mark as defined
                st.success("Editor updated with AI suggestion.")
                st.rerun()
        with col_discard:
             if st.button("❌ Discard Suggestion", key=f"{key_prefix}_discard_ai"):
                 st.session_state.ai_suggestion_pending = None # Clear pending status
                 st.rerun()
        st.divider()

    # --- Display Data Editor ---
    st.markdown("**Hierarchy Editor:**")
    # Provide default empty DataFrame if needed
    if 'hierarchy_df' not in st.session_state or st.session_state.hierarchy_df is None:
         st.session_state.hierarchy_df = pd.DataFrame(columns=['Theme', 'Category', 'Segment', 'Subsegment', 'Keywords'])

    edited_df = st.data_editor(
        st.session_state.hierarchy_df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"{key_prefix}_hierarchy_editor_widget",
        hide_index=True,
        column_config={
             "Theme": st.column_config.TextColumn(required=True),
             "Category": st.column_config.TextColumn(required=True),
             "Segment": st.column_config.TextColumn(required=True),
             "Subsegment": st.column_config.TextColumn("Subsegment", required=True), # Corrected key
             "Keywords": st.column_config.TextColumn("Keywords (comma-sep)"),
         }
    )

    # --- Update Session State and Validate ---
    # Use copy to avoid direct mutation issues with caching/reruns
    current_df_copy = st.session_state.hierarchy_df.copy()
    edited_df_copy = edited_df.copy()

    # Convert to string for reliable comparison (handles dtype issues)
    if not current_df_copy.astype(str).equals(edited_df_copy.astype(str)):
        st.session_state.hierarchy_df = edited_df_copy # Update state with the edited version

        # Validate the *edited* structure immediately
        temp_nested = build_hierarchy_from_df(st.session_state.hierarchy_df)
        if temp_nested and temp_nested.get('themes'):
            st.session_state.hierarchy_defined = True
            st.success("Hierarchy changes saved.")
        else:
            st.session_state.hierarchy_defined = False
            st.warning("Hierarchy is empty or invalid after edits. Define at least one full path.")
        st.rerun() # Rerun to reflect saved changes and validation status


    # --- Show Preview of Nested Structure for Validation ---
    st.markdown("**Preview of Current Nested Structure:**")
    current_nested_hierarchy = build_hierarchy_from_df(st.session_state.hierarchy_df)
    if current_nested_hierarchy and current_nested_hierarchy.get('themes'):
        # Simple validation: check if it has themes
        st.json(current_nested_hierarchy, expanded=False)
        st.session_state.hierarchy_defined = True # Mark valid if structure exists
    else:
        st.warning("The hierarchy structure is currently empty or invalid. Use the editor above.")
        st.session_state.hierarchy_defined = False

    return st.session_state.hierarchy_defined # Return validation status