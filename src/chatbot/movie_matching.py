"""
Movie title matching using multiple algorithms.
"""
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import df_cleaned
from .text_processing import (
    remove_accents,
    normalize_text,
    correct_spelling,
    intent_only_keywords,
    requirement_words,
)

stemmer = PorterStemmer()

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
    words_list = cleaned_input.split()
    candidate_words = [w for w in words_list if w not in intent_only_keywords]

    if not candidate_words:
        return None

    candidate_str = " ".join(candidate_words)
    candidate_stemmed = [stemmer.stem(w) for w in candidate_words]
    cand_len = len(candidate_stemmed)

    best_title = None
    best_score = -1.0

    for norm_title, title_words in title_word_lists:
        title_stemmed = [stemmer.stem(w) for w in title_words]
        title_len = len(title_words)

        overlap_words = set(candidate_stemmed).intersection(set(title_stemmed))
        overlap_count = len(overlap_words)

        if overlap_count == 0:
            continue

        # 1. Base Token F1-score
        f1 = (2.0 * overlap_count) / (cand_len + title_len)

        # 2. Query Coverage: Percentage of query tokens matched
        cand_coverage = overlap_count / cand_len
        score = (cand_coverage * 0.7) + (f1 * 0.3)

        # 3. Calculate length ratio to prevent sub-phrase hijacking
        len_ratio = min(cand_len, title_len) / max(cand_len, title_len)

        # Check contiguous phrase alignment
        norm_title_str = " ".join(title_words)
        is_contiguous = norm_title_str in candidate_str or candidate_str in norm_title_str

        # Apply boost ONLY if candidate length is proportional to the query.
        # Prevents short titles (e.g., "The Jewel") from hijacking longer queries 
        # that happen to contain those words (e.g., "Pokémon: Arceus and the Jewel of Life").
        if is_contiguous and overlap_count == title_len:
            if len_ratio >= 0.6:
                score += 0.3  # Boost for matching proportional title length
            else:
                score *= 0.8  # Penalize embedded short-phrase false positives

        if score > best_score:
            best_score = score
            best_title = norm_title

    if best_title is not None and best_score >= 0.55:
        return best_title

    return None


def find_fuzzy_movie(
    cleaned_input,
    fuzzy_cutoff=80,
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
    vector_cutoff=0.60,
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


__all__ = [
    "find_movie_in_message",
    "get_movie_row",
    "title_norm_to_original",
]
