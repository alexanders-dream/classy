
# AI Hierarchical Text Classifier 🏷️

A Streamlit application designed for classifying text data into predefined hierarchical structures using either Large Language Models (LLMs) via API (Groq, Ollama) or fine-tuned Hugging Face Transformer models.

## Overview

This application provides a user-friendly interface to:

1.  **Upload Text Data:** Load text data you want to classify (CSV or Excel).
2.  **Define Hierarchies:**
    *   For LLMs: Manually define a multi-level hierarchy (Theme, Category, Segment, Subsegment) using a data editor, or get AI-generated suggestions based on your data.
    *   For Hugging Face: The hierarchy is inferred from selected columns in labeled training data.
3.  **Classify Text:**
    *   **LLM Workflow:** Send text snippets (in batches) to a configured LLM (Groq Cloud or a local Ollama instance) for classification against the defined hierarchy.
    *   **Hugging Face Workflow:**
        *   Train a new sequence classification model (e.g., DistilBERT, BERT) on your labeled data.
        *   Load a previously trained and saved model.
        *   Edit classification rules (keyword overrides, confidence thresholds).
        *   Classify unlabeled text using the trained/loaded model and rules.
4.  **Review & Download Results:** View the classified data in a table, analyze basic statistics (distribution), and download the results as CSV or Excel.

## Features

*   **Dual Workflow:** Choose between LLM-based or Hugging Face model-based classification.
*   **Supported LLM Providers:**
    *   Groq (Cloud API, requires API key)
    *   Ollama (Local inference, requires Ollama installed and running)
*   **LLM Hierarchy Management:**
    *   Interactive hierarchy editor (Theme -> Category -> Segment -> Subsegment -> Keywords).
    *   AI-powered hierarchy suggestion based on data samples.
    *   Dynamic fetching of available LLM models (Groq, Ollama).
*   **Hugging Face Model Management:**
    *   Fine-tune common Transformer models (DistilBERT, BERT, RoBERTa) for multi-label classification.
    *   Save trained models, tokenizers, label maps, and rules locally.
    *   Load previously saved models for reuse.
*   **Hugging Face Rule Engine:**
    *   Automatic keyword suggestion using Chi-Squared analysis during training.
    *   Manual editing of keyword overrides and confidence thresholds per label.
    *   Classification applies model probability, thresholds, and keyword rules.
*   **User-Friendly Interface:** Built with Streamlit, featuring tabs for a structured workflow, data previews, progress indicators, and download options.
*   **Data Handling:** Supports CSV and Excel file uploads with basic cleaning.
*   **Configuration:** Uses environment variables (`.env` file or Streamlit Secrets) for API keys and endpoints.

## Prerequisites

*   **Python:** Version 3.8 or higher.
*   **Pip:** Python package installer.
*   **Git:** (Optional) For cloning the repository.
*   **Ollama:** (Optional) If using the Ollama workflow, you need Ollama installed and running locally. Download from [ollama.com](https://ollama.com/). You also need to have pulled the desired models (e.g., `ollama pull llama3`).
*   **Groq API Key:** (Optional) If using the Groq workflow, you need an API key from [GroqCloud](https://console.groq.com/keys).

## Installation

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone https://github.com/alexanders-dream/classy.git
    cd alexanders-dream-classy
    ```

2.  **Create and activate a virtual environment (Recommended):**
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system and whether you have CUDA installed, `transformers[torch]` might require additional setup for GPU acceleration. The `accelerate` package is included for potential performance improvements.*

4.  **Set up Environment Variables:**
    Create a file named `.env` in the root directory (`alexanders-dream-classy/`) of the project. Add the following variables as needed:

    ```dotenv
    # Required for Groq workflow
    GROQ_API_KEY="gsk_YOUR_GROQ_API_KEY"

    # Optional: Override default Ollama endpoint if it's not running on http://localhost:11434
    # OLLAMA_ENDPOINT="http://your_ollama_host:11434"
    ```

    *   Replace `"gsk_YOUR_GROQ_API_KEY"` with your actual Groq API key.
    *   Uncomment and set `OLLAMA_ENDPOINT` only if your Ollama service is running on a different address or port.
    *   **Alternatively:** You can use Streamlit Secrets if deploying on Streamlit Community Cloud. See Streamlit documentation for secrets management. The app will check for `st.secrets["GROQ_API_KEY"]` etc.

## Running the Application

1.  Ensure your virtual environment is activated.
2.  Make sure Ollama is running in the background if you plan to use the Ollama workflow (`ollama serve` in your terminal).
3.  Navigate to the project's root directory in your terminal.
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
5.  The application should open in your default web browser.

## Usage Guide

1.  **Sidebar - Workflow & Configuration:**
    *   **Workflow:** Select either "LLM Categorization" or "Hugging Face Model".
    *   **LLM Configuration (if LLM selected):**
        *   Choose the AI Provider (Groq or Ollama).
        *   Verify/Enter the API Endpoint (defaults are provided).
        *   Enter the API Key if required (e.g., for Groq). The app attempts to load from `.env` or Streamlit secrets first.
        *   Fetch and select the desired AI Model from the list.
        *   Click "End Session & Clear State" to reset the application.

2.  **Tab 1: Data Setup:**
    *   **Data to Classify:** Upload the CSV/Excel file containing the text you want to categorize. Preview the data and select the specific column containing the text.
    *   **Training Data (Optional - for HF only):** If using the Hugging Face workflow *and* you want to train a new model, upload a labeled CSV/Excel file. Select the text column and the columns representing each level of your hierarchy (Theme, Category, etc.). Confirm the column selections.

3.  **Tab 2: Hierarchy:**
    *   **LLM Workflow:**
        *   **AI Suggestion:** If data is loaded, optionally generate a hierarchy suggestion based on text samples using the configured LLM. Review and apply or discard the suggestion.
        *   **Hierarchy Editor:** Manually define or edit the Theme -> Category -> Segment -> Subsegment -> Keywords structure. Add, edit, or delete rows. The preview below shows the nested structure being built. The hierarchy must be valid (contain at least one full path) to proceed.
    *   **Hugging Face Workflow:** Displays the hierarchy columns selected from the training data in Tab 1.

4.  **Tab 3: Classification:**
    *   **LLM Workflow:**
        *   Verify LLM configuration, data upload, and hierarchy definition are complete.
        *   Click "Classify using LLM". Texts will be processed in batches.
    *   **Hugging Face Workflow:**
        *   **Option A: Train/Retrain:** If training data is loaded and columns selected, configure training parameters (base model, epochs, validation split) and click "Start HF Training". The trained model becomes active. You can then optionally save it with a name.
        *   **Option B: Load Model:** Select a previously saved model from the dropdown (models saved in `~/.ai_classifier_saved_models/hf_models/`) and click "Load Selected Model". The loaded model becomes active.
        *   **Review & Edit Rules:** If a model is active, view and edit the associated rules (Label, Keywords, Confidence Threshold). Save changes.
        *   **Run Classification:** Verify a model is active and prediction data is ready. Click "Classify using Hugging Face Model".

5.  **Tab 4: Results:**
    *   View the classification results appended to your original data.
    *   Download the results as a CSV or Excel file.
    *   View workflow-specific statistics (Theme distribution for HF, summary counts for LLM).

## File Structure

classy/
├── app.py # Main Streamlit application file, UI logic.
├── config.py # Configuration constants (endpoints, defaults, hierarchy levels).
├── hf_classifier.py # Functions for Hugging Face training, saving, loading, rules, classification.
├── llm_classifier.py # Functions for LLM client setup, model fetching, suggestions, classification (using LangChain).
├── requirements.txt # Python package dependencies.
├── ui_components.py # Reusable Streamlit UI components (LLM sidebar, hierarchy editor).
└── utils.py # Utility functions (data loading, state management, hierarchy manipulation, stats).
└── .env # (You create this) For storing API keys and sensitive config.


## Troubleshooting

*   **Groq Errors:** Ensure your `GROQ_API_KEY` in `.env` or secrets is correct and valid. Check network connectivity.
*   **Ollama Errors:** Make sure the Ollama application is running (`ollama serve`). Verify the `OLLAMA_ENDPOINT` in `.env` (if set) or the sidebar matches where Ollama is listening. Ensure the selected model has been pulled (`ollama list`).
*   **Dependency Issues:** If `pip install` fails, check your Python/Pip version and network connection. Resolve any conflicting packages.
*   **File Loading Errors:** Ensure files are valid CSV or Excel (.xls, .xlsx). Check for encoding issues (UTF-8 and latin1 are attempted for CSV).
*   **HF Model Training Failures:** May require significant RAM/VRAM. Check error messages for specifics. Ensure training data is correctly formatted and columns are selected.
*   **Slow LLM Performance:** Local Ollama performance depends heavily on your hardware. Batch size and concurrency can be adjusted in `llm_classifier.py` (though not exposed in UI currently). Groq performance depends on their service status.

