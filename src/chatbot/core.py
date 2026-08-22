"""
Main chatbot orchestration logic.
Imports specialized modules for different tasks.
"""

from .text_processing import correct_spelling
from .movie_matching import find_movie_in_message, get_movie_row
from .intent_classifier import predict_class, strip_title_words
from .response_formatter import generate_response


def chatbot_response(user_message, algorithm="hybrid"):
    """
    Main chatbot orchestration function.
    Coordinates between modules: movie matching, intent classification, and response generation.
    """
    # Input validation
    if not isinstance(user_message, str):
        return {"response": "❌ Please enter a text question.", "intent": "N/A", "scores": {}}

    user_message = user_message.strip()

    if not user_message:
        return {"response": "❌ Please enter a question.", "intent": "N/A", "scores": {}}

    # STEP 1: Find movie in message
    matched_norm_title = find_movie_in_message(user_message)

    # STEP 2: Prepare intent input (correct spelling and strip title)
    corrected_message = correct_spelling(user_message)
    intent_input = strip_title_words(corrected_message, matched_norm_title)

    # STEP 3: Predict intent
    if not intent_input.strip():
        predictions = [("search_movie", 1.0)] if matched_norm_title else []
    else:
        predictions = predict_class(
            intent_input,
            algorithm=algorithm,
            error_threshold=0.10,
        )

    # STEP 4: Extract intent tag
    if not predictions:
        if matched_norm_title is not None:
            tag = "search_movie"
        else:
            return {
                "response": (
                    "🤔 I'm not sure I understood that. "
                    "Try asking about a movie's genre, budget, rating, "
                    "runtime, plot, language, release date, or collection."
                ),
                "intent": "N/A",
                "scores": {},
            }
    else:
        tag = predictions[0][0]

    # STEP 5: Handle greeting/goodbye
    if tag in {"greeting", "goodbye"}:
        return {
            "response": generate_response(tag),
            "intent": tag,
            "scores": dict(predictions),
        }

    # STEP 6: Other intents require a movie
    if matched_norm_title is None:
        return {
            "response": (
                "❌ I couldn't identify the movie title. "
                "Please include the movie name."
            ),
            "intent": tag,
            "scores": dict(predictions),
        }

    # STEP 7: Get movie record
    row = get_movie_row(matched_norm_title)

    if row is None:
        return {
            "response": (
                "❌ I found a possible movie title, "
                "but couldn't retrieve its database record."
            ),
            "intent": tag,
            "scores": dict(predictions),
        }

    # STEP 8: Generate response using formatter
    return {
        "response": generate_response(tag, row),
        "intent": tag,
        "scores": dict(predictions),
    }


__all__ = [
    "chatbot_response",
]
