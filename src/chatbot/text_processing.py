"""
Text processing: spell correction, accent removal, normalization.
"""
import re
import unicodedata

import nltk
import wordninja
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer

from .models import df_cleaned

# ============================================================
# NLTK & STEMMER
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

stemmer = PorterStemmer()


# ============================================================
# SPELL CHECKING & ACCENT REMOVAL
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


# Teach the spell checker movie-title vocabulary
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
    "score", "how", "much", "many", "long", "which", "year", "years",
    "production", "country", "countries", "nation", "national",
    "origin", "originate", "foreign", "domestic", "filmed", "filming",
    "location", "behind", "involved", "come", "from", "list", "display",
    "who", "studio", "company", "companies", "produced", "distributed",
    "financed", "backed", "house", "created", "studios", "were",
    "worked", "credits", "owns", "distributor","votes", "vote", 
    "count", "ratings", "reviews", "review", "people",
    "rated", "voted", "number", "users", "viewers", "did", "get",
    "popular", "based", "audience", "critic", "scored", "widely",
    "popularity", "trending", "well", "known", "famous", "hyped",
    "buzz", "hype", "index", "rank", "attention", "getting",
    "interest", "public", "demand", "sought", "after", "current",
    "level", "buzzworthy", "hit", "liked", "engagement", "talked",
    "metric", "trend", "it", "this", "overall", "total"

}

INTENT_TARGETS = list(intent_only_keywords)

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

        # 1. Keep static overrides minimal (only for compound/franchise words)
        if word in common_typo_overrides:
            corrected_words.append(common_typo_overrides[word])
            continue

        # 2. Preserve valid dictionary words, title words, and requirement words
        if word in spell or word in domain_words or word in requirement_words:
            corrected_words.append(word)
            continue

        # 3. Dynamic Fuzzy Intent Matching
        # Catch typos for intent keywords (e.g., 'buget', 'ratin', 'revnue')
        if len(word) >= 3:
            intent_match = process.extractOne(
                word,
                INTENT_TARGETS,
                scorer=fuzz.ratio,
                score_cutoff=75  # Auto-correct to intent keyword if similarity >= 75%
            )
            if intent_match:
                matched_word, score, _ = intent_match
                corrected_words.append(matched_word)
                continue

        # 4. Word Segmentation (wordninja)
        segmented = segment_word(word)
        if segmented is not None:
            corrected_words.append(segmented)
            continue

        # 5. General spell checker fallback
        correction = spell.correction(word)
        corrected_words.append(correction if correction else word)

    return " ".join(corrected_words)


def normalize_text(text):
    text = remove_accents(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    return [
        stemmer.stem(word.lower())
        for word in sentence_words
    ]


__all__ = [
    "remove_accents",
    "correct_spelling",
    "normalize_text",
    "clean_up_sentence",
    "stemmer",
    "intent_only_keywords",
    "requirement_words",
    "domain_words",
]
