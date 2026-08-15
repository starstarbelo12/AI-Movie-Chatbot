"""
Load and store ML models and data.
"""
import json
import pickle

import pandas as pd

from .paths import (
    MLP_MODEL_PATH,
    NB_MODEL_PATH,
    WORDS_PATH,
    CLASSES_PATH,
    DF_CLEANED_PATH,
    INTENTS_PATH,
    ensure_paths_exist,
)

# Ensure all paths exist
ensure_paths_exist()


def load_pickle(file_path, filename):
    """
    Load a pickle file from the specified path.
    
    Args:
        file_path: Full path to pickle file
        filename: Name of file for error messages
        
    Returns:
        Loaded pickle object
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n"
            f"Make sure {filename} is in: {file_path.parent}"
        )

    with open(file_path, "rb") as f:
        return pickle.load(f)


# Load ML Models
mlp_model = load_pickle(MLP_MODEL_PATH, "mlp_model.pkl")
nb_model = load_pickle(NB_MODEL_PATH, "nb_model.pkl")
words = load_pickle(WORDS_PATH, "words.pkl")
classes = load_pickle(CLASSES_PATH, "classes.pkl")

# Load DataFrame
if not DF_CLEANED_PATH.exists():
    raise FileNotFoundError(
        f"Required file not found: {DF_CLEANED_PATH}\n"
        "Make sure df_cleaned.pkl is in the data/models folder."
    )

df_cleaned = pd.read_pickle(DF_CLEANED_PATH)

# Load Intents
if not INTENTS_PATH.exists():
    raise FileNotFoundError(
        f"Required file not found: {INTENTS_PATH}\n"
        "Make sure intents.json is in the config folder."
    )

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)


__all__ = [
    "mlp_model",
    "nb_model",
    "words",
    "classes",
    "df_cleaned",
    "intents",
]
