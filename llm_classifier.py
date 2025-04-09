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
from langchain_community.chat_models import ChatOllama # Correct import for Ollama
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
    sub_segments: List[AISuggestionSubSegment] = Field(..., description="List of sub-segments within this segment")

class AISuggestionCategory(BaseModel):
    name: str = Field(..., description="Name of the category")
    segments: List[AISuggestionSegment] = Field(..., description="List of segments within this category")

class AISuggestionTheme(BaseModel):
    name: str = Field(..., description="Name of the theme")
    categories: List[AISuggestionCategory] = Field(..., description="List of categories within this theme")

class AISuggestionHierarchy(BaseModel):
    """Structure for the AI to generate the hierarchy."""
    themes: List[AISuggestionTheme] = Field(..., description="The complete hierarchical structure.")
    # Add validators if needed for stricter generation

# Model for single row categorization result
class LLMCategorizationResult(BaseModel):
    """Structure for the LLM to return classification for one text row."""
    theme: Optional[str] = Field(None, description="Assigned theme, or null/None if not applicable.")
    category: Optional[str] = Field(None, description="Assigned category, or null/None if not applicable.")
    segment: Optional[str] = Field(None, description="Assigned segment, or null/None if not applicable.")
    subsegment: Optional[str] = Field(None, description="Assigned sub-segment, or null/None if not applicable.") # Corrected key
    reasoning: Optional[str] = Field(None, description="Brief explanation for the categorization.")


# --- LLM Client Initialization ---
@st.cache_resource # Cache the LLM client resource
def initialize_llm_client(provider: str, endpoint: str, api_key: Optional[str], model_name: str):
    """Initializes and returns the LangChain LLM client."""
    st.info(f"LLM Init: Initializing client for {provider} - Model: {model_name}")
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
                requests.get(endpoint, timeout=5)
            except requests.exceptions.ConnectionError:
                 st.warning(f"LLM Init Warning: Cannot reach Ollama endpoint at {endpoint}. Ensure Ollama is running.")
                 # Return the client anyway, maybe it starts later
            except Exception as e_check:
                 st.warning(f"LLM Init Warning: Error checking Ollama endpoint: {e_check}")


        # Add other providers here (e.g., OpenAI, Gemini)
        # elif provider == "OpenAI":
        #     # from langchain_openai import ChatOpenAI
        #     # llm = ChatOpenAI(...)
        #     st.error("OpenAI provider not yet implemented.")
        #     return None
        else:
            st.error(f"LLM Init Error: Unsupported provider '{provider}'")
            return None

        # Simple invocation test
        st.info("LLM Init: Testing client connection...")
        llm.invoke("Respond with OK")
        st.success(f"✅ LLM Client ({provider} - {model_name}) Initialized Successfully.")
        return llm

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
            url = "https://api.groq.com/openai/v1/models" # Use the correct base for models if different
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
            # Filter for models that seem usable (heuristic: avoid embeddings models if named obviously)
            models = sorted([model['name'] for model in data.get('models', []) if 'embed' not in model.get('details',{}).get('family','').lower()])


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
                 st.error("Check your Groq API Key.")
        except: pass
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred fetching models: {e}")
        st.error(traceback.format_exc())
        return []


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
                    for sub_segment in segment.get('sub_segments', []):
                        keywords_str = ', '.join(sub_segment.get('keywords', [])) if sub_segment.get('keywords') else 'N/A'
                        prompt_string += f"      Subsegment: {sub_segment.get('name', 'N/A')} (Keywords: {keywords_str})\n" # Corrected key
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
            return parsed_hierarchy.model_dump() # Return as dict
        except Exception as e_parse:
            st.warning(f"LLM Suggestion: Initial parse failed: {e_parse}. Attempting to fix...")
            try:
                fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_client)
                parsed_hierarchy = fixing_parser.parse(raw_output)
                st.success("✅ Successfully parsed with OutputFixingParser.")
                return parsed_hierarchy.model_dump() # Return as dict
            except Exception as e_fix:
                st.error(f"🔴 LLM Suggestion: Failed to parse or fix AI output: {e_fix}")
                st.error("Raw AI Response that failed parsing:")
                st.code(raw_output, language='json')
                return None

    except Exception as e:
        st.error(f"🔴 LLM Suggestion: Unexpected error during generation: {e}")
        st.error(traceback.format_exc())
        return None


# --- LLM Text Classification ---
def classify_texts_with_llm(
    df: pd.DataFrame,
    text_column: str,
    hierarchy_dict: Dict[str, Any],
    llm_client: Any
    ) -> Optional[pd.DataFrame]:
    """Classifies text in a DataFrame using the LLM and a defined hierarchy."""

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
    categorization_fixing_parser = OutputFixingParser.from_llm(parser=categorization_parser, llm=llm_client)

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

    # Define output columns
    output_cols = {
        'theme': 'LLM_Theme', 'category': 'LLM_Category', 'segment': 'LLM_Segment',
        'subsegment': 'LLM_Subsegment', 'reasoning': 'LLM_Reasoning' # Corrected key
    }
    # Add columns if they don't exist, handle potential conflicts
    for df_col in output_cols.values():
        if df_col not in df_to_process.columns:
            df_to_process[df_col] = pd.NA # Use pandas NA for better type handling
        else:
            st.warning(f"Column '{df_col}' already exists. Output will overwrite it.")


    total_rows = len(df_to_process)
    st.info(f"LLM Classify: Starting categorization for {total_rows} rows...")
    progress_bar = st.progress(0, text="Initializing LLM Categorization...")
    results = []
    error_count = 0
    rate_limit_delay = 1 # Simple delay in seconds if rate limit errors occur

    with st.spinner("🤖 LLM is categorizing rows..."):
        for i, row in df_to_process.iterrows():
            text_to_categorize = str(row[text_column]) if pd.notna(row[text_column]) else ""
            if not text_to_categorize.strip(): # Skip empty text
                results.append(LLMCategorizationResult().model_dump()) # Append empty result
                continue

            # Simple retry mechanism
            for attempt in range(3): # Try up to 3 times
                try:
                    response = categorization_chain.invoke({
                        "hierarchy_structure": hierarchy_str,
                        "text_to_categorize": text_to_categorize
                    })
                    raw_output = response['text']

                    # Parse the response
                    try:
                        parsed_result = categorization_parser.parse(raw_output)
                    except Exception:
                        # Use fixing parser on parse failure
                        parsed_result = categorization_fixing_parser.parse(raw_output)

                    results.append(parsed_result.model_dump())
                    break # Success, exit retry loop

                except Exception as e_cat:
                    error_count += 1
                    error_msg = str(e_cat)
                    if "rate limit" in error_msg.lower() or "limit" in error_msg.lower() or "429" in error_msg:
                         st.warning(f"Rate limit likely hit (Attempt {attempt+1}/3). Waiting {rate_limit_delay}s...")
                         time.sleep(rate_limit_delay)
                         rate_limit_delay = min(rate_limit_delay * 2, 10) # Exponential backoff up to 10s
                         if attempt == 2: # Last attempt failed
                             st.error(f"LLM Classify Error (Row {i}): Rate limit exceeded after retries.")
                             results.append(LLMCategorizationResult(reasoning="Error: Rate limit exceeded").model_dump())
                    else:
                        st.warning(f"LLM Classify Error (Row {i}, Attempt {attempt+1}/3): {error_msg}. Retrying...")
                        if attempt == 2: # Last attempt failed
                            st.error(f"LLM Classify Error (Row {i}): Failed after retries. Error: {error_msg}")
                            results.append(LLMCategorizationResult(reasoning=f"Error: {error_msg[:150]}").model_dump())
                        time.sleep(0.5) # Short delay before retry for other errors


            # Update progress bar
            current_progress = (i + 1) / total_rows
            progress_bar.progress(current_progress, text=f"Processing row {i + 1}/{total_rows}")

    progress_bar.progress(1.0, text="Assigning results...")

    if len(results) == total_rows:
        results_df = pd.DataFrame(results, index=df_to_process.index)
        # Assign results back to the original DataFrame
        for pydantic_field, df_col in output_cols.items():
            if pydantic_field in results_df.columns:
                df_to_process[df_col] = results_df[pydantic_field]
            else:
                 st.warning(f"LLM Classify: Field '{pydantic_field}' missing in LLM results for column '{df_col}'.")
    else:
        st.error(f"LLM Classify: Number of results ({len(results)}) does not match number of rows ({total_rows}). Cannot assign results.")
        return None


    progress_bar.empty()
    st.success(f"✅ LLM Categorization finished! ({error_count} row errors occurred during processing)")
    return df_to_process