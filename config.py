# config.py
"""Configuration constants for the application."""

import os
from pathlib import Path

# --- User Home Directory & Base Save Path ---
# Use pathlib.Path.home() for cross-platform compatibility
USER_HOME = Path.home()
# Define a base directory within the user's home folder for all app models
# Using a hidden folder (starting with '.') is common practice
SAVED_MODELS_BASE_DIR = USER_HOME / ".ai_classifier_saved_models"
# Specific subfolder for Hugging Face models
SAVED_HF_MODELS_BASE_PATH = SAVED_MODELS_BASE_DIR / "hf_models"

# --- HF Training Defaults ---
DEFAULT_VALIDATION_SPLIT = 0.15 # Default fraction for validation
DEFAULT_HF_THRESHOLD = 0.5

# --- Hierarchy Definition ---
HIERARCHY_LEVELS = ["Theme", "Category", "Segment", "Subsegment"] # Define hierarchy order

# --- API Defaults ---
DEFAULT_OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
DEFAULT_GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1")

# --- Model Defaults ---
DEFAULT_GROQ_MODEL = "llama3-70b-8192"
DEFAULT_OLLAMA_MODEL = "llama3:latest"

# --- Classification Defaults ---
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_HF_THRESHOLD = 0.5 # Default confidence threshold for HF predictions

# --- UI Defaults ---
DEFAULT_LLM_SAMPLE_SIZE = 200
MIN_LLM_SAMPLE_SIZE = 50
MAX_LLM_SAMPLE_SIZE = 1000

# --- Provider List ---
SUPPORTED_PROVIDERS = ["Groq", "Ollama"]
