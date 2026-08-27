"""
Movie title matching using multiple algorithms.
"""
import re
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
    # norm = remove_accents(raw_title.lower())
    norm = normalize_text(raw_title)
    if norm not in title_norm_to_original:
        title_norm_to_original[norm] = raw_title

# Store accent-normalized titles for indexing and matching
all_titles_lower = list(title_norm_to_original.keys())

title_word_lists = sorted(
    [(norm_title, norm_title.split()) for norm_title in all_titles_lower],
    key=lambda x: len(x[1]),
    reverse=True,
)

comparison_title_candidates = sorted(
    all_titles_lower,
    key=len,
    reverse=True
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


def token_overlap_score(candidate, norm_title):
    candidate_words = candidate.split()
    title_words = norm_title.split()
    if not candidate_words or not title_words:
        return 0.0

    candidate_stemmed = [stemmer.stem(word) for word in candidate_words]
    title_stemmed = [stemmer.stem(word) for word in title_words]
    overlap_count = len(
        set(candidate_stemmed).intersection(set(title_stemmed))
    )
    if overlap_count == 0:
        return 0.0

    candidate_length = len(candidate_stemmed)
    title_length = len(title_words)
    f1 = (2.0 * overlap_count) / (candidate_length + title_length)
    candidate_coverage = overlap_count / candidate_length
    score = (candidate_coverage * 0.7) + (f1 * 0.3)

    length_ratio = min(candidate_length, title_length) / max(
        candidate_length,
        title_length,
    )
    title_text = " ".join(title_words)
    is_contiguous = title_text in candidate or candidate in title_text
    if is_contiguous and overlap_count == title_length:
        if length_ratio >= 0.6:
            score += 0.3
        else:
            score *= 0.8

    return score


def find_token_overlap_movie(cleaned_input):
    words_list = cleaned_input.split()
    candidate_words = [w for w in words_list if w not in intent_only_keywords]

    if not candidate_words:
        return None

    candidate_str = " ".join(candidate_words)
    best_title = None
    best_score = -1.0

    for norm_title, _ in title_word_lists:
        score = token_overlap_score(candidate_str, norm_title)
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
        scorer=fuzz.ratio,
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


def find_movie_match(user_message):
    """Return the movie selected by the existing matching pipeline."""
    matched_title = find_movie_in_message(user_message)
    if matched_title is None:
        return None

    candidate = " ".join(
        word
        for word in normalize_text(user_message).split()
        if word not in requirement_words and word != "s"
    ).strip()

    exact = find_exact_movie(user_message)
    if exact == matched_title:
        method = "exact"
        selected_confidence = 1.0
    else:
        compact_words = [
            word
            for word in normalize_text(user_message).split()
            if word not in requirement_words and word != "s"
        ]
        compact = find_compact_title(compact_words)
        if compact == matched_title:
            method = "compact"
            selected_confidence = 1.0
        else:
            overlap = find_token_overlap_movie(normalize_text(user_message))
            if overlap == matched_title:
                method = "token overlap"
                selected_confidence = token_overlap_score(
                    candidate,
                    matched_title,
                )
            else:
                fuzzy = find_fuzzy_movie(user_message)
                if fuzzy == matched_title:
                    method = "fuzzy"
                    selected_confidence = fuzz.ratio(
                        candidate,
                        matched_title,
                    ) / 100
                else:
                    method = "vector"
                    vector_scores = cosine_similarity(
                        title_vectorizer.transform([candidate]),
                        title_vectors,
                    )[0]
                    selected_confidence = float(
                        vector_scores[all_titles_lower.index(matched_title)]
                    )

    if method == "token overlap":
        scored_candidates = sorted(
            (
                token_overlap_score(candidate, title),
                title,
            )
            for title in all_titles_lower
            if token_overlap_score(candidate, title) > 0
        )
        candidates = [
            {"title": title, "confidence": score}
            for score, title in reversed(scored_candidates[-3:])
        ]
    else:
        candidates = [
            {"title": title, "confidence": score / 100}
            for title, score, _ in process.extract(
                candidate,
                all_titles_lower,
                scorer=fuzz.ratio,
                limit=3,
            )
        ] if candidate else []

    candidates = [
        {"title": matched_title, "confidence": selected_confidence}
    ] + [
        item for item in candidates if item["title"] != matched_title
    ][:2]

    return {
        "title": matched_title,
        "method": method,
        "match_score": selected_confidence,
        "candidates": candidates,
    }

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


# A normalized title is used for matching, while this map retains the exact
# database value needed to retrieve a row.  This also fixes hyphen/accent
# variants such as "Spider-Man" and "spider man".
title_norm_to_original = {}
for original_title in df_cleaned["title"].astype(str).dropna():
    title_norm_to_original.setdefault(normalize_text(original_title), original_title)

all_titles_lower = list(title_norm_to_original)
title_word_lists = sorted(
    [(title, title.split()) for title in all_titles_lower],
    key=lambda item: len(item[1]),
    reverse=True,
)

comparison_title_candidates = sorted(
    all_titles_lower,
    key=len,
    reverse=True
)

requirement_words = {
    "what", "is", "the", "rating", "ratings", "score", "scores", "of",
    "genre", "budget", "revenue", "earnings", "gross", "box", "office",
    "summary", "summarize", "plot", "story", "storyline", "collection",
    "does", "have", "how", "much", "runtime", "duration", "language",
    "languages", "spoken", "released", "release", "date", "when", "was",
    "about", "tell", "me", "a", "an", "movie", "movies", "film", "films",
    "call", "called", "named", "know", "want", "full", "i", "to", "for",
    "on", "total", "amount", "details", "detail", "information", "info",
    "find", "search", "look", "up", "check", "give", "show", "please",
    "top", "bottom", "highest", "lowest", "most", "least", "best", "worst",
    "biggest", "smallest", "longest", "shortest", "cheapest", "expensive",
    "popular", "popularity", "votes", "vote", "count", "list", "display",
    "and", "or","compare", "compared", "comparing", "comparison", 
    "difference", "differences", "versus", "vs", "part", "series",
    "enough", "cover", "covers", "covered", "back", "make", "made",
    "cost", "costs", "earn", "earns", "earning", "earned",
    "profit", "profits", "profitable", "exceed", "exceeds", "exceeded"
}

title_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
title_vectors = title_vectorizer.fit_transform(all_titles_lower)


def find_exact_movie(user_message):
    message_words = normalize_text(user_message).split()
    for title, title_words in title_word_lists:
        title_len = len(title_words)
        if not title_len or title_len > len(message_words):
            continue
        for index in range(len(message_words) - title_len + 1):
            if message_words[index:index + title_len] != title_words:
                continue
            remaining = message_words[:index] + message_words[index + title_len:]
            if all(word in requirement_words or word.isdigit() for word in remaining):
                return title
    return None


# def find_fuzzy_movie(user_message, fuzzy_cutoff=65, min_title_len=4):
#     candidate = " ".join(
#         word for word in normalize_text(user_message).split()
#         if word not in requirement_words and not word.isdigit()
#     )
#     if len(candidate) < 3:
#         return None
#     result = process.extractOne(candidate, all_titles_lower, scorer=fuzz.token_set_ratio)
#     if result and result[1] >= fuzzy_cutoff and len(result[0]) >= min_title_len:
#         return result[0]
#     return None

def find_fuzzy_movie(user_message, fuzzy_cutoff=65, min_title_len=4):
    candidate = " ".join(
        word for word in normalize_text(user_message).split()
        if word not in requirement_words and not word.isdigit()
    )
    if len(candidate) < 3:
        return None
    result = process.extractOne(candidate, all_titles_lower, scorer=fuzz.ratio)
    if result and result[1] >= fuzzy_cutoff and len(result[0]) >= min_title_len:
        return result[0]
    return None


# def find_vector_movie(user_message, vector_cutoff=0.45, min_title_len=4):
#     candidate = " ".join(
#         word for word in normalize_text(user_message).split()
#         if word not in requirement_words and not word.isdigit()
#     )
#     if len(candidate) < 3:
#         return None
#     similarities = cosine_similarity(title_vectorizer.transform([candidate]), title_vectors)[0]
#     best_index = np.argmax(similarities)
#     best_title = all_titles_lower[best_index]
#     return best_title if similarities[best_index] >= vector_cutoff and len(best_title) >= min_title_len else None

def find_vector_movie(user_message, vector_cutoff=0.45, min_title_len=4):
    candidate = " ".join(
        word for word in normalize_text(user_message).split()
        if word not in requirement_words and not word.isdigit()
    )
    if len(candidate) < 3:
        return None
    similarities = cosine_similarity(title_vectorizer.transform([candidate]), title_vectors)[0]
    best_index = np.argmax(similarities)
    best_title = all_titles_lower[best_index]
    return best_title if similarities[best_index] >= vector_cutoff and len(best_title) >= min_title_len else None

def find_movie_in_message(user_message):
    """
    Compatibility wrapper for chatbot_response().

    Matching order:
    1. Exact title match
    2. Fuzzy title match
    3. Character n-gram/vector match

    Returns the normalized movie title or None.
    """

    # 1. Exact match first
    exact = find_exact_movie(user_message)
    if exact is not None:
        return exact

    compact_words = [
        word
        for word in normalize_text(user_message).split()
        if word not in requirement_words and word != "s"
    ]
    compact = find_compact_title(compact_words)
    if compact is not None:
        return compact

    overlap = find_token_overlap_movie(normalize_text(user_message))
    if overlap is not None:
        return overlap

    # 2. Fuzzy match for typos / small wording differences
    fuzzy = find_fuzzy_movie(user_message)
    if fuzzy is not None:
        return fuzzy

    # 3. Vector fallback for cases such as:
    #    "toystory" -> "toy story"
    #    "jumaj" -> "jumanji"
    vector = find_vector_movie(user_message)
    if vector is not None:
        return vector

    return None

# Comparison queries need a looser vocabulary than single-title lookup:
# words like "compare", "vs", "better" show up constantly and should never
# be treated as part of a title candidate.
comparison_stopwords = requirement_words | {
    "compare", "compared", "comparing", "comparison", "versus", "vs",
    "against", "better", "worse", "which", "than", "or", "and",
    "between", "both", "two",
}

_stopwords_en = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
    "shan", "shouldn", "wasn", "weren", "won", "wouldn",
})

comparison_fillers = _stopwords_en | requirement_words | {
    "compare", "compared", "comparing", "comparison", "versus","vs",
    "bigger", "smaller", "higher", "lower", "greater", "two",
    "better", "worse", "which", "than", "or", "and",
    "between", "both", "two", "difference", "differences", "differ", "differs", "differing",
    "differentiate", "differentiating",
    "contrast", "contrasting", "contrasted",
    "distinguish", "distinguishing", "distinction", "distinctions",
    "disparity", "dissimilarity","make", "back", "cost", "costs", "earn", "earns", "earning",
    "exceed", "exceeds", "exceeded", "cover", "covers", "covered",
    "enough", "profit", "profits", "profitable", "revenues", "money",
    "grossed"
}

def strip_articles(words):
    return [word for word in words if word not in ("the", "a", "an")]

_title_contents = []
_content_to_title = {}
for _title in all_titles_lower:
    _content = " ".join(strip_articles(_title.split()))
    if not _content:
        continue
    _title_contents.append(_content)
    _content_to_title.setdefault(_content, _title)

def _length_aware_score(candidate, title_content, **kwargs):
    base = fuzz.token_set_ratio(candidate, title_content)
    diff = abs(len(candidate.split()) - len(title_content.split()))
    return max(base - diff * 8, 0)

def find_best_fuzzy_title(candidate_words, fuzzy_cutoff=70, min_title_len=4):
    candidate_words = strip_articles(candidate_words)
    if not candidate_words:
        return None
    candidate = " ".join(candidate_words)
    if len(candidate) < 3:
        return None
    result = process.extractOne(
        candidate, _title_contents, scorer=_length_aware_score, score_cutoff=fuzzy_cutoff
    )
    if not result:
        return None
    matched_title = _content_to_title.get(result[0])
    if matched_title and len(matched_title) >= min_title_len:
        return matched_title
    return None

comparison_connectors = re.compile(
    r"\b(?:and|or|vs|versus|against|between|compared to|compare to|to)\b"
)

def find_prefix_containment_title(candidate_words, min_title_len=4):
    candidate = normalize_text(" ".join(candidate_words))

    if len(candidate) < 3:
        return None

    containing_titles = [
        title for title in all_titles_lower
        if len(title) >= min_title_len
        and re.search(
            rf"(?<!\w){re.escape(candidate)}(?!\w)",
            title
        )
    ]

    if not containing_titles:
        return None

    # Use the same normalized format as all_titles_lower
    normalized_titles = (
        df_cleaned["title"]
        .astype(str)
        .map(normalize_text)
    )

    candidates_df = df_cleaned[
        normalized_titles.isin(containing_titles)
    ]

    # Prevent .iloc[0] from crashing when there are no matches
    if candidates_df.empty:
        return None

    # Only sort using columns that exist
    sort_columns = [
        col for col in ["popularity", "vote_count"]
        if col in candidates_df.columns
    ]

    if sort_columns:
        candidates_df = candidates_df.sort_values(
            by=sort_columns,
            ascending=False
        )

    return normalize_text(candidates_df.iloc[0]["title"])

def find_compact_title(candidate_words, min_title_len=4):
    """Match titles when users omit spaces, such as ``ironman 3``."""
    compact_stopwords = {"the", "a", "an", "of"}
    candidate = re.sub(
        r"[^a-z0-9]",
        "",
        "".join(word for word in candidate_words if word not in compact_stopwords),
    )
    if len(candidate) < 3:
        return None

    for title in all_titles_lower:
        if len(title) < min_title_len:
            continue
        compact_title = re.sub(
            r"[^a-z0-9]",
            "",
            "".join(word for word in title.split() if word not in compact_stopwords),
        )
        singular_title = "".join(
            word[:-1] if word.endswith("s") and len(word) > 3 else word
            for word in title.split()
            if word not in compact_stopwords
        )
        if (
            compact_title == candidate
            or singular_title == candidate
            or (
                len(candidate_words) >= 2
                and len(candidate) >= 8
                and candidate in singular_title
            )
        ):
            return title

    return None

def find_movies_in_message(user_message, max_movies=15, fuzzy_cutoff=70, min_title_len=4):
    cleaned = normalize_text(user_message)
    matches = []

    remaining_text = cleaned
    for title in comparison_title_candidates:
        if all(word in requirement_words for word in title.split()):
+            continue
        match = re.search(rf"(?<!\w){re.escape(title)}(?!\w)", remaining_text)
        if match and title not in matches:
            matches.append(title)
            remaining_text = remaining_text[:match.start()] + " " + remaining_text[match.end():]
            if len(matches) == max_movies:
                return matches

    for segment in comparison_connectors.split(remaining_text):
        segment_words = [
            word for word in segment.split()
            if word not in comparison_fillers
        ]
        if not segment_words or len(" ".join(segment_words)) < 3:
            continue

        matched_title = (
            find_prefix_containment_title(segment_words, min_title_len)
            or find_compact_title(segment_words, min_title_len)
            or find_best_fuzzy_title(segment_words, fuzzy_cutoff, min_title_len)
        )
        if matched_title and matched_title not in matches:
            matches.append(matched_title)
            if len(matches) == max_movies:
                break

    return matches


__all__ = [
    "find_movie_in_message",
    "find_movie_match",
    "find_movies_in_message",
    "get_movie_row",
    "title_norm_to_original",
]
