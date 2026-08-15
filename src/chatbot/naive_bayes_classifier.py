"""
Naive Bayes model for intent classification.
"""
import numpy as np

from .models import nb_model, classes
from .text_processing import clean_up_sentence


def bow(sentence, vocabulary):
    """Create bag of words vector from sentence."""
    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(vocabulary)

    for sentence_word in sentence_words:
        for i, word in enumerate(vocabulary):
            if word == sentence_word:
                bag[i] = 1

    return np.array(bag, dtype=np.float64)


def predict_intent_nb(sentence, vocabulary, error_threshold=0.10):
    """
    Predict intent class using Naive Bayes model.
    
    Args:
        sentence: User input text
        vocabulary: List of words in vocabulary (from models.words)
        error_threshold: Minimum probability to include prediction
        
    Returns:
        List of (class_name, probability) tuples, sorted by probability
    """
    p = bow(sentence, vocabulary)

    if p.sum() == 0:
        return []

    model_input = np.array([p], dtype=np.float64)
    probabilities = nb_model.predict_proba(model_input)[0]

    results = [
        (i, float(probability))
        for i, probability in enumerate(probabilities)
        if probability >= error_threshold
    ]

    results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        (classes[index], probability)
        for index, probability in results
    ]


__all__ = [
    "predict_intent_nb",
    "bow",
]
