from pathlib import Path
import json
import pickle
import random
import re
import unicodedata

import numpy as np
import pandas as pd
import nltk
import wordninja

from rapidfuzz import fuzz, process
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD SAVED MODELS / DATA
# ============================================================

def load_pickle(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Make sure {filename} is in the same folder as chatbot.py."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


mlp_model = load_pickle("mlp_model.pkl")
nb_model = load_pickle("nb_model.pkl")
words = load_pickle("words.pkl")
classes = load_pickle("classes.pkl")

df_path = BASE_DIR / "df_cleaned.pkl"
if not df_path.exists():
    raise FileNotFoundError(
        f"Required file not found: {df_path}\n"
        "Make sure df_cleaned.pkl is in the same folder as chatbot.py."
    )

df_cleaned = pd.read_pickle(df_path)


intents_path = BASE_DIR / "intents.json"
if not intents_path.exists():
    raise FileNotFoundError(
        f"Required file not found: {intents_path}\n"
        "Make sure intents.json is in the same folder as chatbot.py."
    )

with open(intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)


# ============================================================
# NLTK
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

stemmer = PorterStemmer()


# ============================================================
# SPELL CORRECTION
# ============================================================

spell = SpellChecker()


def remove_accents(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(c) != "Mn"
    )


# Teach the spell checker movie-title vocabulary so real title words
# are not incorrectly changed.
domain_words = set()

for title in df_cleaned["title"].astype(str).str.lower():
    clean_title = remove_accents(title)
    title_words = re.findall(r"\b\w+\b", clean_title)
    domain_words.update(title_words)

spell.word_frequency.load_words(domain_words)


common_typo_overrides = {
    "si": "is",
    "teh": "the",
    "hw": "how",
    "waht": "what",
    "wat": "what",
    "wut": "what",
    "wich": "which",
    "whic": "which",
    "hows": "how's",
    "wats": "what's",
    "whts": "what's",
}


def segment_word(word):
    pieces = wordninja.split(word)

    if len(pieces) <= 1:
        return None

    def is_valid_piece(piece):
        return (
            piece in domain_words
            or (len(piece) >= 3 and piece in spell)
        )

    if all(is_valid_piece(piece) for piece in pieces):
        return " ".join(pieces)

    return None


def correct_spelling(text):
    text = remove_accents(text)
    corrected_words = []

    for word in text.lower().split():

        if word in common_typo_overrides:
            corrected_words.append(common_typo_overrides[word])
            continue

        if word in spell:
            corrected_words.append(word)
            continue

        segmented = segment_word(word)

        if segmented is not None:
            corrected_words.append(segmented)
            continue

        correction = spell.correction(word)
        corrected_words.append(correction if correction else word)

    return " ".join(corrected_words)


# ============================================================
# MOVIE TITLE MATCHING
# EXACT -> RAPIDFUZZ -> TF-IDF
# ============================================================

all_titles_lower = (
    df_cleaned["title"]
    .astype(str)
    .str.lower()
    .drop_duplicates()
    .tolist()
)

title_word_lists = sorted(
    [(title, title.split()) for title in all_titles_lower],
    key=lambda x: len(x[1]),
    reverse=True,
)

requirement_words = {
    "what", "is", "the", "rating", "of", "genre", "budget", "revenue",
    "summary", "summarize", "plot", "story", "storyline",
    "collection", "does", "have", "how", "much", "long", "runtime",
    "duration", "rt", "language", "languages", "spoken",
    "released", "release", "date", "when", "was", "for", "on",
    "about", "tell", "me", "a", "an", "movie", "film",
    "call", "called", "named", "know", "want", "full", "i",
    "to", "total", "amount", "details", "detail", "information",
    "info", "find", "search", "look", "up", "check",
    "give", "show", "please",
}

title_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
)

title_vectors = title_vectorizer.fit_transform(all_titles_lower)


def normalize_text(text):
    text = remove_accents(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_exact_movie(user_message):
    cleaned_input = normalize_text(user_message)
    message_words = cleaned_input.split()

    for title, title_words in title_word_lists:

        title_len = len(title_words)

        if title_len == 0 or title_len > len(message_words):
            continue

        for i in range(len(message_words) - title_len + 1):

            possible_title = message_words[i:i + title_len]

            if possible_title != title_words:
                continue

            remaining_words = (
                message_words[:i] +
                message_words[i + title_len:]
            )

            if all(word in requirement_words for word in remaining_words):
                return title

    return None


def find_fuzzy_movie(
    user_message,
    fuzzy_cutoff=65,
    min_title_len=4,
):
    cleaned_input = normalize_text(user_message)

    candidate_words = [
        word
        for word in cleaned_input.split()
        if word not in requirement_words
    ]

    candidate = " ".join(candidate_words).strip()

    if len(candidate) < 3:
        return None

    potential_titles = [
        title
        for title in all_titles_lower
        if len(title) >= min_title_len
    ]

    result = process.extractOne(
        candidate,
        potential_titles,
        scorer=fuzz.token_set_ratio,
    )

    if result:
        matched_title, score, _ = result

        if score >= fuzzy_cutoff:
            return matched_title

    return None


def find_vector_movie(
    user_message,
    vector_cutoff=0.45,
    min_title_len=4,
):
    cleaned_input = normalize_text(user_message)

    candidate_words = [
        word
        for word in cleaned_input.split()
        if word not in requirement_words
    ]

    candidate = " ".join(candidate_words).strip()

    if len(candidate) < 3:
        return None

    candidate_vector = title_vectorizer.transform([candidate])

    similarities = cosine_similarity(
        candidate_vector,
        title_vectors,
    )[0]

    best_index = int(np.argmax(similarities))
    best_title = all_titles_lower[best_index]
    best_score = float(similarities[best_index])

    if (
        best_score >= vector_cutoff
        and len(best_title) >= min_title_len
    ):
        return best_title

    return None


def find_movie_in_message(user_message):
    exact_match = find_exact_movie(user_message)

    if exact_match is not None:
        return exact_match

    fuzzy_match = find_fuzzy_movie(user_message)

    if fuzzy_match is not None:
        return fuzzy_match

    vector_match = find_vector_movie(user_message)

    if vector_match is not None:
        return vector_match

    return None


def get_movie_row(user_message):
    matched_title = find_movie_in_message(user_message)

    if matched_title is None:
        return None

    matches = df_cleaned[
        df_cleaned["title"].astype(str).str.lower() == matched_title.lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


# ============================================================
# BAG OF WORDS + INTENT PREDICTION
# ============================================================

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    return [
        stemmer.stem(word.lower())
        for word in sentence_words
    ]


def bow(sentence, vocabulary):
    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(vocabulary)

    for sentence_word in sentence_words:
        for i, word in enumerate(vocabulary):
            if word == sentence_word:
                bag[i] = 1

    return np.array(bag, dtype=np.float64)


def predict_class(
    sentence,
    algorithm="mlp",
    error_threshold=0.10,
):
    p = bow(sentence, words)

    if p.sum() == 0:
        return []

    model_input = np.array([p], dtype=np.float64)

    if algorithm == "mlp":
        probabilities = mlp_model.predict_proba(model_input)[0]

    elif algorithm == "nb":
        probabilities = nb_model.predict_proba(model_input)[0]

    else:
        raise ValueError(
            "Invalid algorithm. Use 'mlp' or 'nb'."
        )

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


# ============================================================
# REMOVE MOVIE TITLE WORDS BEFORE INTENT CLASSIFICATION
# ============================================================

def strip_title_words(text, matched_title, threshold=85):
    if not matched_title:
        return text

    title_words = matched_title.split()
    text_words = text.split()
    kept_words = []

    for word in text_words:

        matched = False

        for title_word in title_words:

            # Avoid matching very short title words against arbitrary words.
            if len(title_word) < 3:
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


# ============================================================
# GET INTENT RESPONSE
# ============================================================

def get_intent_response(tag):
    for intent in intents.get("intents", []):
        if intent.get("tag") == tag:
            responses = intent.get("responses", [])

            if responses:
                return random.choice(responses)

            return None

    return None


# ============================================================
# FINAL CHATBOT
# ============================================================

def chatbot_response(user_message, algorithm="mlp"):
    if not isinstance(user_message, str):
        return "❌ Please enter a text question."

    user_message = user_message.strip()

    if not user_message:
        return "❌ Please enter a question."

    # --------------------------------------------------------
    # STEP 1: SPELL CORRECTION
    # --------------------------------------------------------
    corrected_message = correct_spelling(user_message)

    # --------------------------------------------------------
    # STEP 2: FIND MOVIE FIRST
    # --------------------------------------------------------
    matched_title = find_movie_in_message(corrected_message)

    # --------------------------------------------------------
    # STEP 3: REMOVE MOVIE TITLE FROM INTENT INPUT
    # --------------------------------------------------------
    intent_input = strip_title_words(
        corrected_message,
        matched_title,
    )

    # --------------------------------------------------------
    # STEP 4: PREDICT INTENT USING SELECTED MODEL
    # --------------------------------------------------------
    predictions = predict_class(
        intent_input,
        algorithm=algorithm,
        error_threshold=0.10,
    )

    # --------------------------------------------------------
    # STEP 5: HANDLE NO PREDICTION
    # --------------------------------------------------------
    if not predictions:
        # A bare movie title is naturally interpreted as a
        # request for general movie information.
        if matched_title is not None:
            tag = "search_movie"
        else:
            return (
                "🤔 I'm not sure I understood that. "
                "Try asking about a movie's genre, budget, rating, "
                "runtime, plot, language, release date, or collection."
            )
    else:
        tag = predictions[0][0]

    # --------------------------------------------------------
    # STEP 6: GREETING / GOODBYE
    # THESE ARE CLASSIFIED BY THE MODEL, NOT A RULE OVERRIDE.
    # --------------------------------------------------------
    if tag in {"greeting", "goodbye"}:
        response = get_intent_response(tag)

        if response:
            return response

        if tag == "greeting":
            return (
                "Hello movie fan! Welcome to the Movie Information Desk. "
                "What film can I look up for you today?"
            )

        return "Goodbye! Enjoy your movie night!"

    # --------------------------------------------------------
    # STEP 7: OTHER INTENTS REQUIRE A MOVIE
    # --------------------------------------------------------
    if matched_title is None:
        return (
            "❌ I couldn't identify the movie title. "
            "Please include the movie name."
        )

    # --------------------------------------------------------
    # STEP 8: GET MOVIE RECORD
    # --------------------------------------------------------
    row = get_movie_row(corrected_message)

    if row is None:
        return (
            "❌ I found a possible movie title, "
            "but couldn't retrieve its database record."
        )

    # --------------------------------------------------------
    # STEP 9: GENERATE RESPONSE
    # --------------------------------------------------------

    if tag == "search_movie":
        return (
            f"🎬 **Movie:** {row['title']}\n"
            f"🔹 Collection: {row['belongs_to_collection']}\n"
            f"🔹 Genres: {row['genres']}\n"
            f"🔹 Languages: {row['spoken_languages']}\n"
            f"🔹 Runtime: {int(row['runtime'])} minutes\n"
            f"🔹 Rating: {row['vote_average']}/10\n"
            f"🔹 Budget: ${row['budget']:,.0f}\n"
            f"🔹 Revenue: ${row['revenue']:,.0f}\n"
            f"🔹 Tagline: \"{row['tagline']}\"\n"
            f"📝 Summary: {row['overview']}"
        )

    if tag == "ask_genre":
        return (
            f"🎭 **{row['title']}** belongs to: "
            f"{row['genres']}."
        )

    if tag == "ask_runtime":
        return (
            f"⏱️ **{row['title']}** runs for "
            f"{int(row['runtime'])} minutes."
        )

    if tag == "ask_rating":
        return (
            f"⭐ **{row['title']}** has a rating of "
            f"{row['vote_average']}/10."
        )

    if tag == "ask_collection":
        if row["belongs_to_collection"] == "Unknown":
            return (
                f"📦 **{row['title']}** is not part "
                f"of a known collection."
            )

        return (
            f"📦 **{row['title']}** is part of "
            f"{row['belongs_to_collection']}."
        )

    if tag == "ask_revenue":
        return (
            f"💰 **{row['title']}** earned "
            f"${row['revenue']:,.0f}."
        )

    if tag == "ask_budget":
        return (
            f"🎥 **{row['title']}** had a production budget of "
            f"${row['budget']:,.0f}."
        )

    if tag == "ask_summary":
        return (
            f"📝 **{row['title']}** — "
            f"{row['overview']}"
        )

    if tag == "ask_language":
        return (
            f"🗣️ **{row['title']}** is spoken in: "
            f"{row['spoken_languages']}."
        )

    if tag == "ask_release_date":
        return (
            f"📅 **{row['title']}** was released on "
            f"{row['release_date'].strftime('%B %d, %Y')}."
        )

    return "🤔 Sorry, I didn't quite catch that."


__all__ = [
    "chatbot_response",
    "predict_class",
    "find_movie_in_message",
]
