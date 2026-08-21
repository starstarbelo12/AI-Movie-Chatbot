"""
Intent classification using Bag of Words and ML models.
"""
import re
from rapidfuzz import fuzz

from .models import words
from .text_processing import intent_only_keywords
from .hybrid_classifier import predict_intent_hybrid
from .naive_bayes_classifier import predict_intent_nb


def predict_class(
    sentence,
    algorithm="hybrid",
    error_threshold=0.10
):

    # ========================================================
    # HYBRID (MLP + KNN)
    # ========================================================
    
    if algorithm == "hybrid":

        # probabilities = hybrid_model.predict_proba(
        #     np.array([p])
        # )[0]
        return predict_intent_hybrid(sentence, words, error_threshold)

    # ========================================================
    # NAIVE BAYES
    # ========================================================

    elif algorithm == "nb":

        # probabilities = nb_model.predict_proba(
        #     np.array([p])
        # )[0]
        return predict_intent_nb(sentence, words, error_threshold)
       


    else:

        raise ValueError(
            "algorithm must be 'hybrid' or 'nb'"
        )


def strip_title_words(text, matched_norm_title, threshold=85):
    """
    Remove matched movie title words from text before intent classification.
    Helps ensure intent is correctly classified without title interference.
    """
    if not matched_norm_title:
        return text

    pattern = re.compile(re.escape(matched_norm_title), re.IGNORECASE)
    stripped = pattern.sub("", text).strip()

    if stripped:
        return stripped

    title_words = matched_norm_title.split()
    text_words = text.split()
    kept_words = []

    for word in text_words:
        matched = False
        for title_word in title_words:
            if len(title_word) < 3:
                if word.lower() == title_word.lower():
                    matched = True
                    break
                continue

            if fuzz.ratio(
                word.lower(),
                title_word.lower(),
            ) >= threshold:
                matched = True
                break

        if not matched:
            kept_words.append(word)

    return " ".join(kept_words)


__all__ = [
    "predict_class",
    "strip_title_words",
]
