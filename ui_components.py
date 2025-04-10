import streamlit as st
import pandas as pd
import os
import config
import llm_classifier
from utils import build_hierarchy_from_df, restart_session, flatten_hierarchy
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
        help="Choose the LLM service provider."
    )
    # If the provider selection changes, reset related state variables
    if provider != st.session_state.llm_provider:
        st.session_state.llm_provider = provider
        st.session_state.llm_models = []
        st.session_state.llm_selected_model_name = None
        st.session_state.llm_client = None
        # Set the default endpoint for the newly selected provider
        if provider == "Groq":
            st.session_state.llm_endpoint = config.DEFAULT_GROQ_ENDPOINT
        elif provider == "Ollama":
            st.session_state.llm_endpoint = config.DEFAULT_OLLAMA_ENDPOINT
        else:
            st.session_state.llm_endpoint = "" # Handle other potential providers
        st.rerun() # Rerun the app to apply changes

    # --- API Endpoint Input ---
    # Determine the default endpoint based on the selected provider
    default_endpoint = config.DEFAULT_GROQ_ENDPOINT if provider == "Groq" else config.DEFAULT_OLLAMA_ENDPOINT
    # Initialize endpoint in session state if it's missing or empty for the current provider
    if 'llm_endpoint' not in st.session_state or not st.session_state.llm_endpoint:
         st.session_state.llm_endpoint = default_endpoint

    # Display the endpoint input field
    endpoint = st.sidebar.text_input(
        "API Endpoint:",
        value=st.session_state.llm_endpoint,
        key="llm_endpoint_input",
        help="The base URL for the LLM API. Modify for custom Ollama URLs or other endpoints."
    )
    # If the endpoint changes, reset models and client, then rerun
    if endpoint != st.session_state.llm_endpoint:
         st.session_state.llm_endpoint = endpoint
         st.session_state.llm_models = []
         st.session_state.llm_selected_model_name = None
         st.session_state.llm_client = None
         st.rerun()

    # --- API Key Input (Conditional) ---
    # Determine if the selected provider requires an API key
    api_key_needed = provider == "Groq" # Extend this logic for other providers like OpenAI
    api_key_present = False # Flag to track if a key is available for use

    if api_key_needed:
        # Attempt to load the API key from environment variables or Streamlit secrets first
        env_api_key = os.getenv(f"{provider.upper()}_API_KEY") or st.secrets.get(f"{provider.upper()}_API_KEY")

        # Use the key from env/secrets unless the user has manually entered one in the session
        if env_api_key and not st.session_state.get("llm_api_key_user_override"):
            st.session_state.llm_api_key = env_api_key
            st.sidebar.success(f"{provider} API Key loaded from environment/secrets.", icon="🔒")
            api_key_present = True
        else:
            # If no key from env/secrets or user override is active, show the input field
            user_api_key = st.sidebar.text_input(
                f"{provider} API Key:",
                value=st.session_state.get("llm_api_key", ""),
                type="password",
                key="llm_api_key_input",
                help=f"Required for {provider}. Paste your API key here.",
                placeholder=f"Enter your {provider} API key"
            )
            # Update session state based on user input
            if user_api_key:
                st.session_state.llm_api_key = user_api_key
                st.session_state.llm_api_key_user_override = True # Mark that user provided the key
                api_key_present = True
            else:
                # If user clears the input, reset the state
                st.session_state.llm_api_key = ""
                st.session_state.llm_api_key_user_override = False
                st.sidebar.warning(f"{provider} API Key is required.")
                # Provide helpful links for common providers
                if provider == "Groq": st.sidebar.markdown("[Get Groq API Key](https://console.groq.com/keys)")
                # Add links for other providers if needed

    else:
        # If the provider doesn't need an API key (e.g., default Ollama)
        st.session_state.llm_api_key = ""
        api_key_present = True # Consider the key requirement met

    # --- Fetch Available Models ---
    # Determine if models need to be fetched:
    # 1. If the model list is currently empty.
    # 2. (Implicitly handled by resets above) If provider, endpoint, or key changes.
    should_fetch = not st.session_state.llm_models

    # Prevent fetching if a required API key is missing
    if api_key_needed and not api_key_present:
        should_fetch = False
        if st.session_state.llm_models: # Clear any potentially stale models
            st.session_state.llm_models = []

    # Fetch models if conditions are met
    if should_fetch and st.session_state.llm_endpoint:
        with st.spinner(f"Fetching models for {provider}..."):
            models = llm_classifier.fetch_available_models(
                provider,
                st.session_state.llm_endpoint,
                st.session_state.llm_api_key if api_key_needed else None
            )
        if models:
            st.session_state.llm_models = models
            # Attempt to set a default model if none is currently selected
            if not st.session_state.get('llm_selected_model_name'):
                default_model = config.DEFAULT_GROQ_MODEL if provider == "Groq" else config.DEFAULT_OLLAMA_MODEL
                if default_model in models:
                    st.session_state.llm_selected_model_name = default_model
                elif models: # If default isn't available, use the first model in the list
                    st.session_state.llm_selected_model_name = models[0]
        else:
            st.sidebar.error(f"Could not fetch models for {provider}.")
            if provider == "Ollama": st.sidebar.markdown("[Download Ollama](https://ollama.com/)")
        # Avoid st.rerun() here; let the model selection dropdown update naturally

    # --- Model Selection Dropdown ---
    st.sidebar.divider()
    st.sidebar.markdown("**Model Selection**")

    current_model_name = st.session_state.get('llm_selected_model_name')
    current_model_index = 0
    if current_model_name and st.session_state.llm_models and current_model_name in st.session_state.llm_models:
        current_model_index = st.session_state.llm_models.index(current_model_name)

    # Determine the index for the currently selected model, default to 0 if not found
    current_model_index = 0
    if current_model_name and st.session_state.llm_models and current_model_name in st.session_state.llm_models:
        current_model_index = st.session_state.llm_models.index(current_model_name)

    # Layout for model selection and refresh button
    col_select, col_refresh = st.sidebar.columns([4, 1])

    with col_select:
        if st.session_state.llm_models:
            selected_model_name = st.selectbox(
                "Select AI Model:",
                options=st.session_state.llm_models,
                index=current_model_index,
                key="llm_model_select",
                help="Choose the specific model to use for classification and suggestions."
            )
            # If the selected model changes, update state and reset the client
            if selected_model_name != st.session_state.get('llm_selected_model_name'):
                 st.session_state.llm_selected_model_name = selected_model_name
                 st.session_state.llm_client = None
                 st.rerun()
        else:
            st.warning("No models available. Check endpoint/key and refresh.")
            selected_model_name = None

    # Refresh button column
    with col_refresh:
        # Add some top margin to align button better with selectbox
        st.markdown("<div style='margin-top: 1.8em;'></div>", unsafe_allow_html=True)
        if st.button("🔄", help="Refresh available models list", key="refresh_models_button"):
            # Only refresh if endpoint is set and key is present (if needed)
            if st.session_state.llm_endpoint and (api_key_present if api_key_needed else True):
                 with st.spinner("Fetching models..."):
                     models = llm_classifier.fetch_available_models(
                         st.session_state.llm_provider,
                         st.session_state.llm_endpoint,
                         st.session_state.llm_api_key if api_key_needed else None
                     )
                 if models:
                     st.session_state.llm_models = models
                     st.sidebar.success("Models updated!", icon="✅")
                 else:
                     st.sidebar.error("Failed to fetch models.", icon="❗")
                     st.session_state.llm_models = [] # Clear list on failure
                 st.rerun() # Rerun to update the dropdown with new models
            else:
                st.sidebar.warning("Cannot refresh: Check endpoint and API key (if required).", icon="⚠️")

    # --- Initialize LLM Client ---
    client_ready = False
    # Check if all necessary components are available to initialize the client
    can_initialize = (
        st.session_state.llm_selected_model_name and
        st.session_state.llm_endpoint and
        (api_key_present if api_key_needed else True)
    )

    if can_initialize:
         # Initialize only if the client doesn't exist yet (cached) or relevant config changed (handled by resets)
         if st.session_state.llm_client is None:
             with st.spinner("Initializing LLM client..."):
                 st.session_state.llm_client = llm_classifier.initialize_llm_client(
                     st.session_state.llm_provider,
                     st.session_state.llm_endpoint,
                     st.session_state.llm_api_key if api_key_needed else None,
                     st.session_state.llm_selected_model_name
                 )
         # Check if client initialization was successful
         if st.session_state.llm_client:
             client_ready = True

    # --- Display Status and End Session Button ---
    st.sidebar.divider()
    if client_ready:
        st.sidebar.info(f"Provider: {st.session_state.llm_provider}\n\nModel: {st.session_state.llm_selected_model_name}\n\nStatus: Ready ✅")
    elif st.session_state.llm_selected_model_name:
         # If a model is selected but client isn't ready, there's a config issue
         st.sidebar.error("LLM Client not ready. Check endpoint and API key (if required).")
    else:
         # If no model is selected yet
         st.sidebar.warning("Select a model to initialize the LLM client.")

    # Button to clear session state and restart the app flow
    if st.sidebar.button("End Session & Clear State", key="end_session_sidebar_button"):
        restart_session()

# --- Hierarchy Editor Component ---
def display_hierarchy_editor(key_prefix="main"):
    """
    Displays the Streamlit data editor for the hierarchy and handles AI suggestion logic.

    Args:
        key_prefix (str): A prefix for widget keys to avoid collisions if used multiple times.

    Returns:
        bool: True if the hierarchy is considered defined and valid, False otherwise.
    """
    st.markdown("Define the Theme -> Category -> Segment -> Subsegment structure. Keywords should be comma-separated.")

    # --- Handle Pending AI Suggestion ---
    # Check if an AI suggestion was generated and is waiting for user action
    if st.session_state.get('ai_suggestion_pending'):
        st.info("🤖 An AI-generated hierarchy suggestion is ready!")
        nested_suggestion = st.session_state.ai_suggestion_pending
        # Convert the nested suggestion (dict) into a flat DataFrame for preview
        df_suggestion = flatten_hierarchy(nested_suggestion)

        st.markdown("**Preview of AI Suggestion:**")
        st.dataframe(df_suggestion, use_container_width=True, height=200)

        # Buttons to apply or discard the suggestion
        col_apply, col_discard = st.columns(2)
        with col_apply:
            if st.button("✅ Apply Suggestion (Replaces Editor)", key=f"{key_prefix}_apply_ai", type="primary"):
                st.session_state.hierarchy_df = df_suggestion # Overwrite the editor's content
                st.session_state.ai_suggestion_pending = None # Clear the pending flag
                st.session_state.hierarchy_defined = True # Assume applied suggestion is valid initially
                st.success("Editor updated with AI suggestion.")
                st.rerun() # Rerun to show the updated editor
        with col_discard:
             if st.button("❌ Discard Suggestion", key=f"{key_prefix}_discard_ai"):
                 st.session_state.ai_suggestion_pending = None # Clear the pending flag
                 st.rerun() # Rerun to remove the suggestion display
        st.divider()

    # --- Display Data Editor ---
    st.markdown("**Hierarchy Editor:**")
    # Initialize the hierarchy DataFrame in session state if it doesn't exist
    if 'hierarchy_df' not in st.session_state or st.session_state.hierarchy_df is None:
         st.session_state.hierarchy_df = pd.DataFrame(columns=['Theme', 'Category', 'Segment', 'Subsegment', 'Keywords'])

    # Display the data editor widget
    edited_df = st.data_editor(
        st.session_state.hierarchy_df,
        num_rows="dynamic", # Allow adding/deleting rows
        use_container_width=True,
        key=f"{key_prefix}_hierarchy_editor_widget",
        hide_index=True,
        # Configure columns, making key levels required
        column_config={
             "Theme": st.column_config.TextColumn(required=True),
             "Category": st.column_config.TextColumn(required=True),
             "Segment": st.column_config.TextColumn(required=True),
             "Subsegment": st.column_config.TextColumn("Subsegment", required=True), # Standardized name
             "Keywords": st.column_config.TextColumn("Keywords (comma-sep)"),
         }
    )

    # --- Update Session State and Validate on Edit ---
    # Compare the edited DataFrame with the one currently in session state
    # Use copies and string conversion for robust comparison
    current_df_copy = st.session_state.hierarchy_df.copy()
    edited_df_copy = edited_df.copy()

    if not current_df_copy.astype(str).equals(edited_df_copy.astype(str)):
        st.session_state.hierarchy_df = edited_df_copy # Update session state

        # Immediately validate the newly edited structure
        temp_nested = build_hierarchy_from_df(st.session_state.hierarchy_df)
        # A valid structure should have a 'themes' list, even if empty initially
        is_valid = bool(temp_nested and 'themes' in temp_nested)

        if is_valid and temp_nested.get('themes'):
            st.session_state.hierarchy_defined = True
            st.success("Hierarchy changes saved and structure appears valid.")
        elif is_valid: # Structure exists but no themes yet
             st.session_state.hierarchy_defined = False
             st.info("Hierarchy changes saved. Add at least one full path (Theme to Subsegment).")
        else: # build_hierarchy_from_df returned None or invalid dict
            st.session_state.hierarchy_defined = False
            st.warning("Hierarchy structure seems invalid after edits. Check for missing levels.")
        st.rerun() # Rerun to reflect the saved state and validation message

    # --- Show Preview of Nested Structure ---
    st.markdown("**Preview of Current Nested Structure (for validation):**")
    # Rebuild the nested structure from the current DataFrame in state
    current_nested_hierarchy = build_hierarchy_from_df(st.session_state.hierarchy_df)

    if current_nested_hierarchy and current_nested_hierarchy.get('themes'):
        # If themes exist, display the JSON preview and mark as defined
        st.json(current_nested_hierarchy, expanded=False)
        st.session_state.hierarchy_defined = True
    else:
        # If no themes or invalid structure, show warning
        st.warning("The hierarchy structure is currently empty or invalid. Use the editor above to define at least one complete path.")
        st.session_state.hierarchy_defined = False

    # Return the current validation status
    return st.session_state.hierarchy_defined
