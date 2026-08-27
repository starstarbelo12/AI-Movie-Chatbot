import re
from .models import df_cleaned
from .text_processing import normalize_text

MIN_RELIABLE_BUDGET = 500_000
MAX_RANK_RESULTS = 10

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
    """Return the requested result count, capped at the supported maximum."""
    normalized = normalize_text(text)
    count_patterns = (
        r"\b(?:top|bottom|first|last|show|list|display|give)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:top|bottom|highest|lowest|most|least|best|worst|biggest|smallest|longest|shortest|cheapest)\b",
    )
    for pattern in count_patterns:
        match = re.search(pattern, normalized)
        if match:
            requested = NUMBER_WORDS.get(match.group(1), None)
            return max(
                1,
                min(
                    requested if requested is not None else int(match.group(1)),
                    MAX_RANK_RESULTS,
                ),
            )
    return default


def requested_rank_count(text):
    """Return an explicitly requested count before applying the result cap."""
    normalized = normalize_text(text)
    count_patterns = (
        r"\b(?:top|bottom|first|last|show|list|display|give)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:top|bottom|highest|lowest|most|least|best|worst|biggest|smallest|longest|shortest|cheapest)\b",
    )
    for pattern in count_patterns:
        match = re.search(pattern, normalized)
        if match:
            value = match.group(1)
            return NUMBER_WORDS[value] if value in NUMBER_WORDS else int(value)
    return None


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
    Detect a genre from the user's query and return only movies
    belonging to that genre.

    Returns:
        (filtered_df, genre_display_name)

    If no genre is detected, returns:
        (df_cleaned, None)
    """

    normalized_query = normalize_text(user_message)

    # Normalize aliases so "sci-fi", "sci fi", and "scifi"
    # can all map to the same dataset genre.
    normalized_aliases = {
        normalize_text(alias): genre
        for alias, genre in GENRE_ALIASES.items()
    }

    # Search longer phrases first, e.g. "science fiction"
    search_terms = sorted(
        set(
            [normalize_text(g) for g in KNOWN_GENRES]
            + list(normalized_aliases.keys())
        ),
        key=len,
        reverse=True,
    )

    detected_genre = None

    for term in search_terms:
        if re.search(
            rf"\b{re.escape(term)}\b",
            normalized_query,
        ):
            detected_genre = normalized_aliases.get(
                term,
                term,
            )
            break

    # No genre in the query
    if detected_genre is None:
        return df_cleaned, None

    # Normalize the genre that will be searched in the dataset
    normalized_target_genre = normalize_text(
        detected_genre
    )

    # Normalize the entire genres column.
    #
    # This works whether the dataset contains:
    #   "Science Fiction, Adventure"
    #   "['Science Fiction', 'Adventure']"
    #   '[{"name": "Science Fiction"}, {"name": "Adventure"}]'
    normalized_genres = (
        df_cleaned["genres"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )

    # Match the normalized genre against the normalized
    # dataset genre text.
    genre_mask = normalized_genres.str.contains(
        re.escape(normalized_target_genre),
        na=False,
        regex=True,
    )

    filtered_df = df_cleaned[genre_mask].copy()

    # If a genre was detected but nothing matched, do NOT
    # silently return the entire dataset. That would cause
    # exactly the problem seen in your screenshot.
    if filtered_df.empty:
        return filtered_df, detected_genre.title()

    return filtered_df, detected_genre.title()

def rank_movies(tag, user_message):
    config = RANKING_CONFIG[tag]
    column = config["column"]

    # A ranking must contain at least two results. Reject explicit requests
    # such as "top 1" or "bottom 1" instead of presenting a one-item list.
    requested_count = requested_rank_count(user_message)
    if requested_count is not None and requested_count < 2:
        return (
            "❌ Ranking requires at least two movies. "
            "Please request top 2 or more results."
        )

    # ========================================================
    # 1. Filter by genre if present
    # ========================================================

    target_df, detected_genre = get_genre_filtered_df(
        user_message
    )

    # ========================================================
    # 2. Keep the columns needed for ranking
    # ========================================================

    rank_columns = [
        "title",
        column,
    ]

    # Include genres so we can display them when a genre
    # was requested.
    if "genres" in target_df.columns:
        rank_columns.append("genres")

    ranked = (
        target_df[rank_columns]
        .dropna(subset=[column])
        .copy()
    )

    # ========================================================
    # 3. Reliability filters
    # ========================================================

    # Popularity must be at least 0.1
    if "popularity" in target_df.columns:
        ranked = ranked[
            target_df.loc[
                ranked.index,
                "popularity"
            ] >= 0.1
        ]

    # Vote count must be at least 10
    if "vote_count" in target_df.columns:
        ranked = ranked[
            target_df.loc[
                ranked.index,
                "vote_count"
            ] >= 10
        ]

    # ========================================================
    # 4. Ranking-specific filters
    # ========================================================

    if column == "budget":
        ranked = ranked[
            ranked[column].map(
                has_reliable_budget
            )
        ]

    elif column == "revenue":
        ranked = ranked[
            ranked[column] > 0
        ]

    elif column == "vote_average":
        # Remove unrated movies
        ranked = ranked[
            ranked[column] > 0.1
        ]

    # A genre or reliability filter may leave fewer than two usable records.
    if len(ranked) < 2:
        return (
            "❌ Ranking requires at least two movies with reliable data. "
            "Please try another category or request a broader ranking."
        )

    # ========================================================
    # 5. Sort and limit
    # ========================================================

    ranked = (
        ranked
        .drop_duplicates(
            subset=["title"]
        )
        .sort_values(
            by=column,
            ascending=(
                get_rank_direction(user_message)
                == "ascending"
            ),
            kind="stable",
        )
        .head(
            get_rank_count(user_message)
        )
    )

    if ranked.empty:
        return (
            "❌ I couldn't find enough reliable "
            "movie data for that ranking."
        )

    # ========================================================
    # 6. Ranking heading
    # ========================================================

    direction = (
        "lowest"
        if get_rank_direction(user_message)
        == "ascending"
        else "highest"
    )

    genre_label = (
        f" {detected_genre}"
        if detected_genre
        else ""
    )

    limit_notice = (
        f"\n\n_Note: showing {MAX_RANK_RESULTS} results maximum; "
        "this chatbot does not support more than 10 ranking results._"
        if requested_count and requested_count > MAX_RANK_RESULTS
        else ""
    )

    lines = [
        f"🏆 **{direction.title()} "
        f"{config['label']}"
        f"{genre_label} movies:**"
    ]

    # ========================================================
    # 7. Display ranked movies
    # ========================================================

    for position, (_, movie) in enumerate(
        ranked.iterrows(),
        start=1,
    ):
        line = (
            f"{position}. **{movie['title']}** — "
            f"{config['format'](movie[column])}"
        )

        # Only display genre when the USER requested a genre.
        if (
            detected_genre
            and "genres" in ranked.columns
        ):
            movie_genres = str(
                movie["genres"]
            ).strip()

            if (
                movie_genres
                and movie_genres.lower()
                not in {"nan", "unknown"}
            ):
                line += (
                    f" — Genre: {movie_genres}"
                )

        lines.append(line)

    return "\n".join(lines) + limit_notice
