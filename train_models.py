"""Train the chatbot intent models from config/intents.json."""
import json
import pickle
import re
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parent
INTENTS_PATH = ROOT / "config" / "intents.json"
MODELS_DIR = ROOT / "data" / "models"
STEMMER = PorterStemmer()


def tokenize(text):
    return [STEMMER.stem(word.lower()) for word in re.findall(r"\b\w+\b", text)]


def load_training_data():
    with INTENTS_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    patterns = []
    labels = []
    for intent in config["intents"]:
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            if isinstance(pattern, str) and pattern.strip():
                patterns.append(pattern)
                labels.append(tag)

    if not patterns:
        raise ValueError("No training patterns found in config/intents.json")

    vocabulary = sorted({word for pattern in patterns for word in tokenize(pattern)})
    classes = sorted(set(labels))
    class_to_index = {tag: index for index, tag in enumerate(classes)}
    features = np.zeros((len(patterns), len(vocabulary)), dtype=np.float64)
    vocabulary_index = {word: index for index, word in enumerate(vocabulary)}

    for row_index, pattern in enumerate(patterns):
        for word in set(tokenize(pattern)):
            features[row_index, vocabulary_index[word]] = 1.0

    targets = np.array([class_to_index[tag] for tag in labels])
    return features, targets, vocabulary, classes


def train_and_save():
    features, targets, vocabulary, classes = load_training_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.20,
        random_state=42,
        stratify=targets,
    )

    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=0.0005,
        max_iter=500,
        random_state=42,
    )
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    hybrid_model = VotingClassifier(
        estimators=[("mlp", mlp), ("knn", knn)],
        voting="soft",
        weights=[2, 1],
    )
    hybrid_model.fit(X_train, y_train)

    print("\n=== NAIVE BAYES EVALUATION ===")
    nb_predictions = nb_model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, nb_predictions):.4f}")
    print(classification_report(
        y_test,
        nb_predictions,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
    ))

    print("=== HYBRID EVALUATION ===")
    hybrid_predictions = hybrid_model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, hybrid_predictions):.4f}")
    print(classification_report(
        y_test,
        hybrid_predictions,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
    ))

    # Refit the artifacts on every labeled pattern after evaluation.
    nb_model.fit(features, targets)
    hybrid_model.fit(features, targets)

    artifacts = {
        "nb_model.pkl": nb_model,
        "hybrid_model.pkl": hybrid_model,
        "words.pkl": vocabulary,
        "classes.pkl": np.array(classes),
    }
    for filename, artifact in artifacts.items():
        with (MODELS_DIR / filename).open("wb") as file:
            pickle.dump(artifact, file)

    print(f"Trained {len(classes)} intents from {len(targets)} patterns.")
    print(f"Saved model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    train_and_save()
