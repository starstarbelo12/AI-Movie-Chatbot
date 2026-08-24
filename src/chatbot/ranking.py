MIN_RELIABLE_BUDGET = 500_000

RANKING_CONFIG = {
    "rank_rating": {"column": "vote_average", "label": "rating", "format": lambda value: f"{value:.1f}/10"},
    "rank_revenue": {"column": "revenue", "label": "revenue", "format": lambda value: f"${value:,.0f}"},
    "rank_budget": {"column": "budget", "label": "budget", "format": lambda value: f"${value:,.0f}"},
    "rank_popularity": {"column": "popularity", "label": "popularity", "format": lambda value: f"{value:.1f}"},
    "rank_vote_count": {"column": "vote_count", "label": "vote count", "format": lambda value: f"{int(value):,}"},
    "rank_runtime": {"column": "runtime", "label": "runtime", "format": lambda value: f"{int(value)} minutes"},
}

LOWEST_WORDS = {"lowest", "low", "least","worse", "worst", "smallest", "shortest", "fewest", "cheapest", "bottom"}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

def has_reliable_budget(value):
    try:
        return float(value) >= MIN_RELIABLE_BUDGET
    except (TypeError, ValueError):
        return False


def get_rank_direction(text):
    return "ascending" if set(normalize_text(text).split()) & LOWEST_WORDS else "descending"


def get_rank_count(text, default=5):
    """Return an explicitly requested count, defaulting to 5 and never exceeding 10."""
    normalized = normalize_text(text)
    count_patterns = (
        r"\b(?:top|bottom|first|last|show|list|display|give)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:top|bottom|highest|lowest|most|least|best|worst|biggest|smallest|longest|shortest|cheapest)\b",
    )
    for pattern in count_patterns:
        match = re.search(pattern, normalized)
        if match:
            requested = NUMBER_WORDS.get(match.group(1), None)
            return max(1, min(requested if requested is not None else int(match.group(1)), 10))
    return default


# Exact legitimate genres found in movies_metadata.csv
KNOWN_GENRES = {
    "action", "adventure", "animation", "comedy", "crime",
    "documentary", "drama", "family", "fantasy", "foreign",
    "history", "horror", "music", "mystery", "romance",
    "science fiction", "tv movie", "thriller", "war", "western"
}

# Aliases to help users search flexibly
GENRE_ALIASES = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "animated": "animation",
    "romantic": "romance",
    "musical": "music"
}

def get_genre_filtered_df(user_message):
    """
    Extracts genre from user query and returns (filtered_df, genre_display_name).
    If no genre is found, returns (df_cleaned, None).
    """
    normalized = normalize_text(user_message)

    # Sort by length descending to catch multi-word genres (e.g., "science fiction") first
    all_search_terms = sorted(list(KNOWN_GENRES) + list(GENRE_ALIASES.keys()), key=len, reverse=True)

    for term in all_search_terms:
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            # Map the alias back to the exact dataset spelling if needed
            target_genre = GENRE_ALIASES.get(term, term)

            # Filter df_cleaned for rows where the genres column contains the target_genre
            filtered_df = df_cleaned[
                df_cleaned["genres"].astype(str).str.lower().str.contains(target_genre, na=False)
            ]

            if not filtered_df.empty:
                return filtered_df, target_genre.title()

    # Fallback to the full dataset if no genre detected
    return df_cleaned, None


def rank_movies(tag, user_message):
    config = RANKING_CONFIG[tag]
    column = config["column"]

    # 1. Filter by genre if present, otherwise default to full dataset
    target_df, detected_genre = get_genre_filtered_df(user_message)

    ranked = target_df[["title", column]].dropna(subset=[column]).copy()


    # Popularity must be at least 0.1
    if "popularity" in target_df.columns:
        ranked = ranked[target_df.loc[ranked.index, "popularity"] >= 0.1]

    # Vote count must be at least 10
    if "vote_count" in target_df.columns:
        ranked = ranked[target_df.loc[ranked.index, "vote_count"] >= 10]

    if column == "budget":
        ranked = ranked[ranked[column].map(has_reliable_budget)]
    elif column == "revenue":
        ranked = ranked[ranked[column] > 0]
    elif column == "vote_average":
        # Drop unrated movies (0.0) so they don't skew the "worst rating" results
        ranked = ranked[ranked[column] > 0.1]

    ranked = ranked.drop_duplicates(subset=["title"]).sort_values(
        by=column,
        ascending=(get_rank_direction(user_message) == "ascending"),
        kind="stable",
    ).head(get_rank_count(user_message))

    if ranked.empty:
        return "❌ I couldn't find enough reliable movie data for that ranking."

    direction = "lowest" if get_rank_direction(user_message) == "ascending" else "highest"

    # 2. Format title depending on whether a genre was detected
    genre_label = f" {detected_genre}" if detected_genre else ""
    lines = [f"🏆 **{direction.title()} {config['label']}{genre_label} movies:**"]

    for position, (_, movie) in enumerate(ranked.iterrows(), start=1):
        lines.append(f"{position}. **{movie['title']}** — {config['format'](movie[column])}")

    return "\n".join(lines)