import streamlit as st
import pandas as pd
import requests
import json
import time
import traceback
import config # Import your config file
from typing import List, Dict, Optional, Any

# LangChain components
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field, validator, ValidationError

# --- Pydantic Models ---
# Model for AI-generated hierarchy suggestion
class AISuggestionSubSegment(BaseModel):
    name: str = Field(..., description="Name of the sub-segment")
    keywords: List[str] = Field(..., description="List of 3-7 relevant keywords for this sub-segment")

class AISuggestionSegment(BaseModel):
    name: str = Field(..., description="Name of the segment")
    subsegments: List[AISuggestionSubSegment] = Field(..., description="List of subsegments within this segment")

class AISuggestionCategory(BaseModel):
    name: str = Field(..., description="Name of the category")
    segments: List[AISuggestionSegment] = Field(..., description="List of segments within this category")

class AISuggestionTheme(BaseModel):
    name: str = Field(..., description="Name of the theme")
    categories: List[AISuggestionCategory] = Field(..., description="List of categories within this theme")

class AISuggestionHierarchy(BaseModel):
    """Structure for the AI to generate the hierarchy."""
    themes: List[AISuggestionTheme] = Field(..., description="The complete hierarchical structure.")

# Model for single row categorization result
class LLMCategorizationResult(BaseModel):
    """Structure for the LLM to return classification for one text row."""
    theme: Optional[str] = Field(None, description="Assigned theme, or null/None if not applicable.")
    category: Optional[str] = Field(None, description="Assigned category, or null/None if not applicable.")
    segment: Optional[str] = Field(None, description="Assigned segment, or null/None if not applicable.")
    subsegment: Optional[str] = Field(None, description="Assigned sub-segment, or null/None if not applicable.")
    reasoning: Optional[str] = Field(None, description="Brief explanation for the categorization.")


# --- LLM Client Initialization ---
@st.cache_resource # Cache the LLM client resource
def initialize_llm_client(provider: str, endpoint: str, api_key: Optional[str], model_name: str):
    """Initializes and returns the LangChain LLM client."""
    st.info(f"LLM Init: Initializing client for {provider} - Model: {model_name}")
    llm = None # Initialize llm to None
    try:
        if provider == "Groq":
            if not api_key:
                st.error("LLM Init Error: Groq API key is missing.")
                return None
            llm = ChatGroq(
                temperature=config.DEFAULT_LLM_TEMPERATURE,
                groq_api_key=api_key,
                model_name=model_name,
                request_timeout=60, # Longer timeout for potentially complex tasks
            )
        elif provider == "Ollama":
             # Ollama uses base_url, not api_base in newer langchain
            llm = ChatOllama(
                base_url=endpoint,
                model=model_name,
                temperature=config.DEFAULT_LLM_TEMPERATURE,
                request_timeout=120 # Potentially longer timeout for local models
            )
            # Simple check if Ollama endpoint is reachable (optional)
            try:
                requests.get(endpoint, timeout=5) # Quick check if endpoint is responsive
            except requests.exceptions.ConnectionError:
                 st.warning(f"LLM Init Warning: Cannot reach Ollama endpoint at {endpoint}. Ensure Ollama is running.")
                 # Proceeding anyway, as Ollama might start later or be accessible by the LangChain lib differently
            except Exception as e_check:
                 st.warning(f"LLM Init Warning: Error during optional check of Ollama endpoint: {e_check}")

        # Add other providers here (e.g., OpenAI, Gemini)
        # elif provider == "OpenAI":
        #     # from langchain_openai import ChatOpenAI
        #     # llm = ChatOpenAI(...)
        #     st.error("OpenAI provider not yet implemented.")
        #     return None
        else:
            st.error(f"LLM Init Error: Unsupported provider '{provider}'")
            return None

        if llm: # Only test if llm was successfully initialized
            # Simple invocation test to confirm basic connectivity and authentication
            st.info("LLM Init: Testing client connection with a simple request...")
            llm.invoke("Respond with 'OK'") # Use single quotes for clarity
            st.success(f"✅ LLM Client ({provider} - {model_name}) Initialized and Responding.")
            return llm
        else:
             # This case should ideally be caught by the provider checks, but as a safeguard:
             st.error(f"LLM Init Error: Client for {provider} could not be initialized.")
             return None

    except Exception as e:
        st.error(f"🔴 LLM Init Error: Failed to initialize {provider} client: {e}")
        st.error(traceback.format_exc())
        return None

# --- Model Fetching ---
def fetch_available_models(provider: str, endpoint: str, api_key: Optional[str]) -> List[str]:
    """Fetches available model names from the selected provider's endpoint."""
    st.info(f"Fetching models for {provider} from {endpoint}...")
    headers = {}
    models = []

    try:
        if provider == "Groq":
            if not api_key:
                st.error("Cannot fetch Groq models: API key missing.")
                return []
            # Groq uses OpenAI compatible endpoint for models
            url = "https://api.groq.com/openai/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise error for bad status codes
            data = response.json()
            models = sorted([model['id'] for model in data.get('data', []) if 'id' in model])

        elif provider == "Ollama":
            # Ollama uses /api/tags endpoint
            url = f"{endpoint.strip('/')}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Basic filtering heuristic: exclude models likely intended for embeddings based on name/family
            models = sorted([
                model['name'] for model in data.get('models', [])
                if 'embed' not in model.get('details', {}).get('family', '').lower()
            ])

        # Add other providers here
        # elif provider == "OpenAI": ...

        if not models:
             st.warning(f"No models found for {provider} at {endpoint}.")
        else:
             st.success(f"Found {len(models)} models for {provider}.")

        return models

    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not reach endpoint {endpoint}. Is the service running?")
        if provider == "Ollama": st.info("Ensure Ollama is running locally (`ollama serve`)")
        return []
    except requests.exceptions.Timeout:
        st.error(f"Timeout: Request to {endpoint} timed out.")
        return []
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error fetching models: {e.response.status_code} {e.response.reason}")
        try:
            st.error(f"Response body: {e.response.text}") # Show error details from API
            if e.response.status_code == 401 and provider == "Groq":
                 st.error("Authentication Error (401): Please check your Groq API Key.")
        except Exception as inner_e:
            # Avoid crashing if response.text is not available or causes another error
            st.warning(f"Could not display full error response body: {inner_e}")
        return [] # Return empty list on HTTP error
    except Exception as e:
        st.error(f"An unexpected error occurred fetching models: {e}")
        st.error(traceback.format_exc())
        return [] # Return empty list on other exceptions


# --- Hierarchy Formatting for Prompt ---
def format_hierarchy_for_prompt(hierarchy: Dict[str, Any]) -> str:
    """ Formats the nested hierarchy dictionary into a string for LLM prompts."""
    if not hierarchy or 'themes' not in hierarchy or not hierarchy['themes']:
        return "No hierarchy defined."

    prompt_string = "Available Categorization Structure:\n---\n"
    try:
        for theme in hierarchy.get('themes', []):
            prompt_string += f"Theme: {theme.get('name', 'N/A')}\n"
            for category in theme.get('categories', []):
                prompt_string += f"  Category: {category.get('name', 'N/A')}\n"
                for segment in category.get('segments', []):
                    prompt_string += f"    Segment: {segment.get('name', 'N/A')}\n"
                    for sub_segment in segment.get('subsegments', []): # Uses 'subsegments' key from Pydantic model
                        keywords_str = ', '.join(sub_segment.get('keywords', [])) if sub_segment.get('keywords') else 'N/A'
                        prompt_string += f"      Subsegment: {sub_segment.get('name', 'N/A')} (Keywords: {keywords_str})\n"
                prompt_string += "\n" # Space between categories
            prompt_string += "---\n" # Separator between themes
    except Exception as e:
        st.error(f"Error formatting hierarchy for prompt: {e}")
        return "Error: Could not format hierarchy structure."
    return prompt_string


# --- LLM Hierarchy Suggestion ---
def generate_hierarchy_suggestion(llm_client: Any, sample_texts: List[str]) -> Optional[Dict[str, Any]]:
    """Uses the LLM to generate a hierarchy suggestion based on sample text."""
    if not llm_client or not sample_texts:
        st.error("LLM Suggestion: LLM client or sample texts missing.")
        return None

    st.info(f"LLM Suggestion: Analyzing {len(sample_texts)} samples...")
    try:
        # --- LangChain Prompt for Hierarchy Generation ---
        generation_prompt_template = """
        You are an expert data analyst creating structured hierarchies. Analyze the sample text data below to identify themes, categories, segments, and sub-segments.

        **Sample Data:**
        ```
        {sample_data}
        ```

        **Instructions:**
        1. Identify 2-5 main **Themes**.
        2. For each Theme, identify relevant **Categories**.
        3. For each Category, identify relevant **Segments**.
        4. For each Segment, identify specific **Sub-Segments**.
        5. For each **Sub-Segment**, list 3-7 relevant **Keywords** found in or directly inferred from the sample data.
        6. Ensure the hierarchy is logical and covers the data's main topics. Avoid redundancy.
        7. Base the hierarchy *only* on the provided data. Do not invent unrelated topics.
        8. Output the result STRICTLY in the specified JSON format.

        {format_instructions}
        """

        parser = PydanticOutputParser(pydantic_object=AISuggestionHierarchy)
        prompt = PromptTemplate(
            template=generation_prompt_template,
            input_variables=["sample_data"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = LLMChain(llm=llm_client, prompt=prompt)
        sample_data_str = "\n".join([f"- {text}" for text in sample_texts])

        # Invoke LLM
        with st.spinner("🤖 LLM is thinking... Generating hierarchy..."):
            llm_response = chain.invoke({"sample_data": sample_data_str})
            raw_output = llm_response['text']

        # Attempt to parse the output
        st.info("LLM Suggestion: Parsing response...")
        try:
            parsed_hierarchy = parser.parse(raw_output)
            st.success("✅ LLM response parsed successfully.")
            return parsed_hierarchy.model_dump() # Return as a standard dictionary
        except Exception as e_parse:
            st.warning(f"LLM Suggestion: Initial Pydantic parse failed: {e_parse}. Attempting automated fix...")
            try:
                # Use LangChain's fixer which asks the LLM to correct the format
                fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_client)
                parsed_hierarchy = fixing_parser.parse(raw_output)
                st.success("✅ Successfully parsed hierarchy using OutputFixingParser.")
                return parsed_hierarchy.model_dump() # Return as a standard dictionary
            except Exception as e_fix:
                st.error(f"🔴 LLM Suggestion: OutputFixingParser also failed: {e_fix}")
                st.error("Raw AI Response that failed parsing:")
                st.code(raw_output, language='json')
                return None # Return None if fixing fails

    except Exception as e:
        st.error(f"🔴 LLM Suggestion: Unexpected error during generation: {e}")
        st.error(traceback.format_exc())
        return None


# --- LLM Text Classification (Batch Processing) ---
def classify_texts_with_llm(
    df: pd.DataFrame,
    text_column: str,
    hierarchy_dict: Dict[str, Any],
    llm_client: Any,
    batch_size: int = 10, # Number of texts to process in one LLM call
    max_concurrency: int = 5 # Max parallel LLM requests allowed simultaneously
    ) -> Optional[pd.DataFrame]:
    """
    Classifies text in a DataFrame using the LLM, a defined hierarchy, and batch processing.

    Args:
        df: Input DataFrame.
        text_column: Name of the column containing text to classify.
        hierarchy_dict: Nested dictionary defining the classification structure.
        llm_client: Initialized LangChain LLM client.
        batch_size: How many rows to send to the LLM in each batch request.
        max_concurrency: Maximum number of concurrent requests to the LLM provider.

    Returns:
        A DataFrame with added columns for LLM classification results (Theme, Category, etc.)
        or None if a critical error occurs.
    """

    if df is None or df.empty or not text_column or not hierarchy_dict or not llm_client:
        st.error("LLM Classify: Missing inputs (DataFrame, text column, hierarchy, or LLM client).")
        return None

    # --- Setup LangChain for Categorization ---
    categorization_prompt_template = """
    You are an AI assistant specialized in precise text categorization. Classify the text snippet below into the *single most appropriate* path within the provided structure. Only use the Theme, Category, Segment, and Subsegment names explicitly defined in the structure. If no path fits well, you may return null for some or all levels. Provide brief reasoning.

    **Hierarchy Structure:**
    ```
    {hierarchy_structure}
    ```

    **Text Snippet to Categorize:**
    ```
    {text_to_categorize}
    ```

    {format_instructions}
    """

    categorization_parser = PydanticOutputParser(pydantic_object=LLMCategorizationResult)
    # Note: Using OutputFixingParser in batch mode can be slow if many items fail parsing.
    # It's often better to handle parse errors individually after the batch returns.

    categorization_prompt = PromptTemplate(
        template=categorization_prompt_template,
        input_variables=["hierarchy_structure", "text_to_categorize"],
        partial_variables={"format_instructions": categorization_parser.get_format_instructions()}
    )
    categorization_chain = LLMChain(llm=llm_client, prompt=categorization_prompt)

    # --- Prepare Data and Run ---
    df_to_process = df.copy()
    hierarchy_str = format_hierarchy_for_prompt(hierarchy_dict)
    if "Error:" in hierarchy_str or "No hierarchy defined" in hierarchy_str:
        st.error(f"LLM Classify: Invalid hierarchy structure provided for prompt: {hierarchy_str}")
        return None

    # Define mapping from Pydantic fields to DataFrame column names
    output_cols = {
         'theme': 'LLM_Theme',
         'category': 'LLM_Category',
         'segment': 'LLM_Segment',
         'subsegment': 'LLM_Subsegment', # Matches Pydantic field name
         'reasoning': 'LLM_Reasoning'
     }
    # Add output columns to the DataFrame if they don't exist, handling potential conflicts
    for df_col in output_cols.values():
        if df_col not in df_to_process.columns:
            df_to_process[df_col] = pd.NA # Use pandas NA for better type handling
        else:
            st.warning(f"Column '{df_col}' already exists. Output will overwrite it.")

    total_rows = len(df_to_process)
    st.info(f"LLM Classify: Starting batch categorization for {total_rows} rows (Batch Size: {batch_size}, Concurrency: {max_concurrency})...")
    progress_bar = st.progress(0, text="Initializing LLM Batch Categorization...")
    all_results_parsed = []
    error_count = 0
    processed_rows = 0

    # Prepare list of texts to process
    texts_to_classify_list = df_to_process[text_column].fillna("").astype(str).tolist()
    original_indices = df_to_process.index.tolist() # Keep track of original indices

    with st.spinner(f"🤖 LLM is categorizing in batches..."):
        for i in range(0, total_rows, batch_size):
            batch_texts = texts_to_classify_list[i : i + batch_size]
            batch_indices = original_indices[i : i + batch_size]

            if not batch_texts: continue # Skip empty batches

            # Create inputs for the batch call
            batch_inputs = [
                {
                    "hierarchy_structure": hierarchy_str,
                    "text_to_categorize": text
                }
                for text in batch_texts if text.strip() # Only include non-empty texts
            ]
            # Keep track of indices corresponding to non-empty texts in the batch
            valid_indices_in_batch = [idx for idx, text in zip(batch_indices, batch_texts) if text.strip()]

            if not batch_inputs: # If batch only contained empty texts
                 # Add empty results for all original indices in this batch
                 all_results_parsed.extend([{'index': idx, **LLMCategorizationResult().model_dump()} for idx in batch_indices])
                 processed_rows += len(batch_indices)
                 current_progress = processed_rows / total_rows
                 progress_bar.progress(current_progress, text=f"Processed {processed_rows}/{total_rows} rows (Skipped empty batch)")
                 continue

            try:
                # --- Execute Batch Request ---
                batch_responses = categorization_chain.batch(
                    batch_inputs,
                    config={"max_concurrency": max_concurrency}
                )
                # batch_responses is a list of dictionaries, e.g., [{'text': '...'}, {'text': '...'}]

                # --- Process Batch Responses ---
                if len(batch_responses) != len(batch_inputs):
                     st.error(f"Batch Error: Mismatch between input ({len(batch_inputs)}) and output ({len(batch_responses)}) count.")
                     # Add error results for this batch
                     all_results_parsed.extend([{'index': idx, **LLMCategorizationResult(reasoning="Error: Batch response mismatch").model_dump()} for idx in valid_indices_in_batch])
                     error_count += len(valid_indices_in_batch)
                else:
                    for idx, response in zip(valid_indices_in_batch, batch_responses):
                        raw_output = response.get('text', '')
                        try:
                            parsed_result = categorization_parser.parse(raw_output)
                            all_results_parsed.append({'index': idx, **parsed_result.model_dump()})
                        except Exception as e_parse:
                            st.warning(f"LLM Parse Error (Index {idx}): {e_parse}. Raw: '{raw_output[:100]}...'")
                            # Attempting to fix might be slow/unreliable in batch, log as error
                            all_results_parsed.append({'index': idx, **LLMCategorizationResult(reasoning=f"Error: Failed to parse LLM output - {e_parse}").model_dump()})
                            error_count += 1

                # Add empty results for any empty texts that were skipped *within* this batch run
                empty_indices_in_batch = [idx for idx, text in zip(batch_indices, batch_texts) if not text.strip()]
                all_results_parsed.extend([{'index': idx, **LLMCategorizationResult().model_dump()} for idx in empty_indices_in_batch])


            except Exception as e_batch:
                st.error(f"LLM Batch Error (Rows {i+1}-{i+batch_size}): {e_batch}")
                st.error(traceback.format_exc())
                # Add error results for all valid items in the failed batch
                all_results_parsed.extend([{'index': idx, **LLMCategorizationResult(reasoning=f"Error: Batch processing failed - {e_batch}").model_dump()} for idx in valid_indices_in_batch])
                # Add empty results for skipped empty texts
                empty_indices_in_batch = [idx for idx, text in zip(batch_indices, batch_texts) if not text.strip()]
                all_results_parsed.extend([{'index': idx, **LLMCategorizationResult().model_dump()} for idx in empty_indices_in_batch])
                error_count += len(valid_indices_in_batch)


            # Update progress
            processed_rows += len(batch_indices) # Increment by the original batch size
            current_progress = processed_rows / total_rows
            progress_bar.progress(current_progress, text=f"Processed {processed_rows}/{total_rows} rows")

    progress_bar.progress(1.0, text="Assigning results...")

    # --- Consolidate Results ---
    if len(all_results_parsed) == total_rows:
        # Create DataFrame from parsed results, ensuring index matches original
        results_df_final = pd.DataFrame(all_results_parsed).set_index('index')
        # Assign results back to the original DataFrame columns using the index
        for pydantic_field, df_col in output_cols.items():
            if pydantic_field in results_df_final.columns:
                 # Use .loc for robust index-based assignment
                 df_to_process.loc[results_df_final.index, df_col] = results_df_final[pydantic_field]
            else:
                 st.warning(f"LLM Classify: Field '{pydantic_field}' missing in LLM results for column '{df_col}'.")
    else:
        # Correctly indented error message and return
        st.error(f"LLM Classify Error: Number of processed results ({len(all_results_parsed)}) does not match total rows ({total_rows}). Cannot assign results reliably.")
        return None # Return None if result count mismatch

    # Correctly indented final block
    progress_bar.empty()
    st.success(f"✅ LLM Batch Categorization finished! ({error_count} row errors occurred during processing)")
    return df_to_process # Return the processed DataFrame
