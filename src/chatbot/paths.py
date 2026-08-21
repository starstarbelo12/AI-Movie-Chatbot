"""
Centralized path management for the chatbot project.
Handles all file paths relative to project root.
"""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR = DATA_DIR

# Config paths
CONFIG_DIR = PROJECT_ROOT / "config"

# Static paths
STATIC_DIR = PROJECT_ROOT / "static"

# Model files
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.pkl"
NB_MODEL_PATH = MODELS_DIR / "nb_model.pkl"
WORDS_PATH = MODELS_DIR / "words.pkl"
CLASSES_PATH = MODELS_DIR / "classes.pkl"
DF_CLEANED_PATH = MODELS_DIR / "df_cleaned.pkl"

# Config files
INTENTS_PATH = CONFIG_DIR / "intents.json"

# CSS files
STYLE_CSS_PATH = STATIC_DIR / "style.css"


def ensure_paths_exist():
    """Verify all required directories exist."""
    for path_dir in [MODELS_DIR, CONFIG_DIR, STATIC_DIR]:
        path_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "CONFIG_DIR",
    "STATIC_DIR",
    "HYBRID_MODEL_PATH",
    "NB_MODEL_PATH",
    "WORDS_PATH",
    "CLASSES_PATH",
    "DF_CLEANED_PATH",
    "INTENTS_PATH",
    "STYLE_CSS_PATH",
    "ensure_paths_exist",
]
