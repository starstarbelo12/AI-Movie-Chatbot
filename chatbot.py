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
# SPELL CORRECTION & ACCENT REMOVAL
# ============================================================

spell = SpellChecker()


def remove_accents(text):
    """
    Normalizes accented characters to standard ASCII letters.
    E.g., 'Pokémon' -> 'Pokemon', 'Amélie' -> 'Amelie'.
    """
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

# Pure intent keywords stripped when extracting title candidates
intent_only_keywords = {
    "what", "is", "rating", "genre", "genres", "budget", "revenue",
    "summary", "summarize", "plot", "story", "storyline",
    "collection", "runtime", "duration", "rt", "time", "length",
    "language", "languages", "spoken", "released", "release",
    "date", "when", "was", "tell", "me", "details", "detail",
    "information", "info", "find", "search", "look", "up",
    "check", "give", "show", "please", "cost", "made", "earned",
    "score", "how", "much", "many", "long", "which"
}

# Full requirement word list used for exact sentence matching validation
requirement_words = intent_only_keywords.union({
    "the", "of", "for", "on", "about", "a", "an", "movie", "film",
    "call", "called", "named", "know", "want", "full", "i", "to",
    "total", "amount", "does", "have"
})


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

        # Protect valid dictionary words, title words, and intent words from segmentation
        if word in spell or word in domain_words or word in requirement_words:
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
# ACCENT-STRIPPED MOVIE TITLE MATCHING INDEX
# ============================================================

# Map normalized accent-free title string to original title string
title_norm_to_original = {}

for raw_title in df_cleaned["title"].astype(str).drop_duplicates():
    norm = remove_accents(raw_title.lower())
    if norm not in title_norm_to_original:
        title_norm_to_original[norm] = raw_title

# Store accent-normalized titles for indexing and matching
all_titles_lower = list(title_norm_to_original.keys())

title_word_lists = sorted(
    [(norm_title, norm_title.split()) for norm_title in all_titles_lower],
    key=lambda x: len(x[1]),
    reverse=True,
)

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


def find_exact_movie(cleaned_input):
    """
    Finds exact title matches in user input against normalized title index.
    Validates that any non-title words in the message belong to requirement_words.
    """
    message_words = cleaned_input.split()
    msg_len = len(message_words)

    for norm_title, title_words in title_word_lists:
        title_len = len(title_words)

        if title_len == 0 or title_len > msg_len:
            continue

        for i in range(msg_len - title_len + 1):
            possible_title = message_words[i:i + title_len]

            if possible_title == title_words:
                remaining_words = message_words[:i] + message_words[i + title_len:]
                if all(word in requirement_words for word in remaining_words):
                    return norm_title

    return None


def find_token_overlap_movie(cleaned_input):
    """
    Ranks movie titles using stemmed token F1-score balancing against normalized titles.
    """
    words_list = cleaned_input.split()
    candidate_words = [w for w in words_list if w not in intent_only_keywords]

    if not candidate_words:
        return None

    candidate_stemmed = [stemmer.stem(w) for w in candidate_words]
    cand_len = len(candidate_stemmed)

    best_title = None
    max_overlap = 0
    best_f1 = 0.0

    for norm_title, title_words in title_word_lists:
        title_stemmed = [stemmer.stem(w) for w in title_words]
        title_len = len(title_words)

        overlap_words = set(candidate_stemmed).intersection(set(title_stemmed))
        overlap_count = len(overlap_words)

        if overlap_count > 0:
            f1 = (2.0 * overlap_count) / (cand_len + title_len)

            if title_len == 1 and cand_len > 1:
                f1 *= 0.1

            if (overlap_count > max_overlap) or (
                overlap_count == max_overlap and f1 > best_f1
            ):
                max_overlap = overlap_count
                best_f1 = f1
                best_title = norm_title

    min_required = 1 if cand_len == 1 else 2
    if max_overlap >= min_required and best_f1 >= 0.25:
        return best_title

    return None


def find_fuzzy_movie(
    cleaned_input,
    fuzzy_cutoff=65,
):
    """
    Fuzzy matches candidate keywords against stored normalized movie titles.
    """
    if not cleaned_input:
        return None

    words_list = cleaned_input.split()
    candidate_words = [
        word for word in words_list if word not in intent_only_keywords
    ]
    candidate = " ".join(candidate_words).strip()

    if not candidate:
        candidate = cleaned_input

    results = process.extract(
        candidate,
        all_titles_lower,
        scorer=fuzz.token_set_ratio,
        limit=5,
    )

    for matched_title, score, _ in results:
        if len(matched_title.split()) == 1 and len(candidate.split()) > 1:
            continue

        effective_cutoff = 85 if len(candidate) <= 3 else fuzzy_cutoff
        if score >= effective_cutoff:
            return matched_title

    return None


def find_vector_movie(
    cleaned_input,
    vector_cutoff=0.45,
):
    """
    TF-IDF character n-gram cosine similarity fallback matching on normalized titles.
    """
    candidate_words = [
        word
        for word in cleaned_input.split()
        if word not in intent_only_keywords
    ]

    candidate = " ".join(candidate_words).strip()

    if len(candidate) < 2:
        return None

    candidate_vector = title_vectorizer.transform([candidate])

    similarities = cosine_similarity(
        candidate_vector,
        title_vectors,
    )[0]

    top_indices = np.argsort(similarities)[::-1][:5]

    for idx in top_indices:
        best_title = all_titles_lower[idx]
        best_score = float(similarities[idx])

        if best_score < vector_cutoff:
            break

        if len(best_title.split()) == 1 and len(candidate.split()) > 1:
            continue

        return best_title

    return None


def match_pipeline(cleaned_text):
    """Executes exact -> token overlap -> fuzzy -> vector matching cascade."""
    exact_match = find_exact_movie(cleaned_text)
    if exact_match is not None:
        return exact_match

    overlap_match = find_token_overlap_movie(cleaned_text)
    if overlap_match is not None:
        return overlap_match

    fuzzy_match = find_fuzzy_movie(cleaned_text)
    if fuzzy_match is not None:
        return fuzzy_match

    vector_match = find_vector_movie(cleaned_text)
    if vector_match is not None:
        return vector_match

    return None


def find_movie_in_message(user_message):
    """
    Double-Track Matching Engine:
    Track 1: Match raw input first against accent-normalized titles.
    Track 2: Fall back to spell-corrected text if raw input fails.
    """
    raw_normalized = normalize_text(user_message)
    matched = match_pipeline(raw_normalized)
    if matched is not None:
        return matched

    corrected_message = correct_spelling(user_message)
    corrected_normalized = normalize_text(corrected_message)
    return match_pipeline(corrected_normalized)


def get_movie_row(matched_norm_title):
    """
    Retrieves the DataFrame record for a matched normalized movie title.
    Maps back to the original title and resolves duplicates by popularity/revenue/votes.
    """
    if not matched_norm_title:
        return None

    original_title = title_norm_to_original.get(matched_norm_title)
    if not original_title:
        return None

    matches = df_cleaned[
        df_cleaned["title"].astype(str).str.lower() == original_title.lower()
    ]

    if matches.empty:
        # Fallback search if exact original case string isn't found directly
        matches = df_cleaned[
            df_cleaned["title"].astype(str).apply(lambda x: remove_accents(x.lower())) == matched_norm_title
        ]

    if matches.empty:
        return None

    sort_cols = [
        col for col in ["popularity", "vote_count", "revenue", "budget"]
        if col in matches.columns
    ]
    if sort_cols:
        matches = matches.sort_values(by=sort_cols, ascending=False)

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

def strip_title_words(text, matched_norm_title, threshold=85):
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
    # STEP 1: FIND MOVIE (DUAL-TRACK: RAW & CORRECTED)
    # --------------------------------------------------------
    matched_norm_title = find_movie_in_message(user_message)

    # --------------------------------------------------------
    # STEP 2: PREPARE INTENT INPUT
    # --------------------------------------------------------
    corrected_message = correct_spelling(user_message)
    intent_input = strip_title_words(
        corrected_message,
        matched_norm_title,
    )

    if not intent_input.strip():
        intent_input = corrected_message

    # --------------------------------------------------------
    # STEP 3: PREDICT INTENT USING SELECTED MODEL
    # --------------------------------------------------------
    predictions = predict_class(
        intent_input,
        algorithm=algorithm,
        error_threshold=0.10,
    )

    # --------------------------------------------------------
    # STEP 4: HANDLE NO PREDICTION
    # --------------------------------------------------------
    if not predictions:
        if matched_norm_title is not None:
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
    # STEP 5: GREETING / GOODBYE
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
    # STEP 6: OTHER INTENTS REQUIRE A MOVIE
    # --------------------------------------------------------
    if matched_norm_title is None:
        return (
            "❌ I couldn't identify the movie title. "
            "Please include the movie name."
        )

    # --------------------------------------------------------
    # STEP 7: GET MOVIE RECORD (WITH AMBIGUITY RESOLUTION)
    # --------------------------------------------------------
    row = get_movie_row(matched_norm_title)

    if row is None:
        return (
            "❌ I found a possible movie title, "
            "but couldn't retrieve its database record."
        )

    # --------------------------------------------------------
    # STEP 8: SAFE DATA FIELD FORMATTING
    # --------------------------------------------------------
    try:
        rel_date_str = pd.to_datetime(row["release_date"]).strftime('%B %d, %Y')
    except Exception:
        rel_date_str = str(row.get("release_date", "Unknown"))

    budget_val = row.get("budget", 0)
    revenue_val = row.get("revenue", 0)
    runtime_val = row.get("runtime", 0)

    budget_str = f"${budget_val:,.0f}" if pd.notnull(budget_val) else "N/A"
    revenue_str = f"${revenue_val:,.0f}" if pd.notnull(revenue_val) else "N/A"
    runtime_str = f"{int(runtime_val)} minutes" if pd.notnull(runtime_val) else "N/A"

    # --------------------------------------------------------
    # STEP 9: GENERATE RESPONSE
    # --------------------------------------------------------

    if tag == "search_movie":
        return (
            f"🎬 **Movie:** {row['title']}\n"
            f"🔹 Collection: {row.get('belongs_to_collection', 'Unknown')}\n"
            f"🔹 Genres: {row.get('genres', 'N/A')}\n"
            f"🔹 Languages: {row.get('spoken_languages', 'N/A')}\n"
            f"🔹 Runtime: {runtime_str}\n"
            f"🔹 Rating: {row.get('vote_average', 'N/A')}/10\n"
            f"🔹 Budget: {budget_str}\n"
            f"🔹 Revenue: {revenue_str}\n"
            f"🔹 Tagline: \"{row.get('tagline', '')}\"\n"
            f"📝 Summary: {row.get('overview', 'N/A')}"
        )

    if tag == "ask_genre":
        return (
            f"🎭 **{row['title']}** belongs to: "
            f"{row.get('genres', 'N/A')}."
        )

    if tag == "ask_runtime":
        return (
            f"⏱️ **{row['title']}** runs for "
            f"{runtime_str}."
        )

    if tag == "ask_rating":
        return (
            f"⭐ **{row['title']}** has a rating of "
            f"{row.get('vote_average', 'N/A')}/10."
        )

    if tag == "ask_collection":
        col = row.get("belongs_to_collection", "Unknown")
        if col == "Unknown" or pd.isna(col):
            return (
                f"📦 **{row['title']}** is not part "
                f"of a known collection."
            )

        return (
            f"📦 **{row['title']}** is part of "
            f"{col}."
        )

    if tag == "ask_revenue":
        return (
            f"💰 **{row['title']}** earned "
            f"{revenue_str}."
        )

    if tag == "ask_budget":
        return (
            f"🎥 **{row['title']}** had a production budget of "
            f"{budget_str}."
        )

    if tag == "ask_summary":
        return (
            f"📝 **{row['title']}** — "
            f"{row.get('overview', 'N/A')}"
        )

    if tag == "ask_language":
        return (
            f"🗣️ **{row['title']}** is spoken in: "
            f"{row.get('spoken_languages', 'N/A')}."
        )

    if tag == "ask_release_date":
        return (
            f"📅 **{row['title']}** was released on "
            f"{rel_date_str}."
        )

    return "🤔 Sorry, I didn't quite catch that."


__all__ = [
    "chatbot_response",
    "predict_class",
    "find_movie_in_message",
]