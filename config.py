import os

# --- File Paths ---
SAVED_HF_MODEL_PATH = "./saved_classification_model"  # Directory for saved HF model artifacts

# --- HF Training Defaults ---
DEFAULT_VALIDATION_SPLIT = 0.15 # Default fraction of training data to use for validation (e.g., 15%)
DEFAULT_HF_THRESHOLD = 0.5

# --- Hierarchy Definition ---
HIERARCHY_LEVELS = ["Theme", "Category", "Segment", "Subsegment"] # Define hierarchy order

# --- API Defaults ---
# Load defaults from environment variables if they exist, otherwise use hardcoded defaults
DEFAULT_OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
DEFAULT_GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1") # Groq uses OpenAI compatible endpoint

# --- Model Defaults ---
# Suggest default models known to work well for each provider
DEFAULT_GROQ_MODEL = "llama3-70b-8192"
DEFAULT_OLLAMA_MODEL = "llama3:latest" # Or another model you have downloaded

# --- Classification Defaults ---
DEFAULT_LLM_TEMPERATURE = 0.1 # Low temp for deterministic classification
DEFAULT_HF_THRESHOLD = 0.5

# --- UI Defaults ---
DEFAULT_LLM_SAMPLE_SIZE = 200
MIN_LLM_SAMPLE_SIZE = 50
MAX_LLM_SAMPLE_SIZE = 1000

# --- Provider List ---
SUPPORTED_PROVIDERS = ["Groq", "Ollama"] # Add "OpenAI", "Gemini" later if needed