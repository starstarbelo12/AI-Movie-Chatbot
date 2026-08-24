"""
Main chatbot orchestration logic.
Imports specialized modules for different tasks.
"""

from .text_processing import (
    correct_spelling,
    normalize_text,
)

from .movie_matching import (
    find_movie_in_message,
    get_movie_row,
    find_movies_in_message,
)

from .intent_classifier import (
    predict_class,
    strip_title_words,
)

from .response_formatter import generate_response

from .ranking import (
    RANKING_CONFIG,
    rank_movies,
    has_reliable_budget,
)

from .comparison import (
    compare_movies,
    compare_budget_and_revenue,
)


# ============================================================
# RANKING / COMPARISON INTENT DETECTION
# ============================================================

def detect_supported_intent(text, comparison_titles=None):
    """
    Deterministically routes ranking, comparison, and budget
    requests before normal ML intent classification.
    """

    normalized = normalize_text(text)
    words = set(normalized.split())
    comparison_titles = comparison_titles or []

    # --------------------------------------------------------
    # 1. Financial comparison
    # --------------------------------------------------------
    financial_comparison = any(
        phrase in normalized
        for phrase in (
            "make back",
            "made back",
            "more than it cost",
            "more than its budget",
            "revenue exceed",
            "earn more than",
            "earn enough to cover",
            "profitable",
            "make a profit",
            "budget and revenue",
            "budget to revenue",
        )
    )

    if financial_comparison:
        return "compare_budget_revenue"

    # --------------------------------------------------------
    # 2. General movie comparison
    # --------------------------------------------------------
    comparison_words = {
        "compare",
        "compared",
        "comparison",
        "higher",
        "lower",
        "bigger",
        "smaller",
        "more",
        "less",
        "better",
        "worse",
    }

    if len(comparison_titles) >= 2 and words & comparison_words:
        return "compare_movies"

    # --------------------------------------------------------
    # 3. Ranking intents
    # --------------------------------------------------------
    ranking_words = {
        "top",
        "bottom",
        "highest",
        "lowest",
        "most",
        "least",
        "best",
        "worst",
        "biggest",
        "smallest",
        "longest",
        "shortest",
        "fewest",
        "cheapest",
    }

    if words & ranking_words:

        # Vote count
        if {
            "vote",
            "votes",
            "voted",
            "voting",
            "count",
            "counts",
        } & words:
            return "rank_vote_count"

        # Budget
        if {
            "budget",
            "budgets",
            "cost",
            "costs",
            "expensive",
            "spent",
            "production",
            "produce",
            "making",
        } & words:
            return "rank_budget"

        # Revenue
        if {
            "revenue",
            "revenues",
            "earn",
            "earned",
            "earnings",
            "gross",
            "box",
            "office",
        } & words:
            return "rank_revenue"

        # Popularity
        if {
            "popularity",
            "popular",
            "trending",
            "famous",
        } & words:
            return "rank_popularity"

        # Runtime
        if {
            "runtime",
            "duration",
            "longest",
            "shortest",
        } & words:
            return "rank_runtime"

        # Rating
        if {
            "rating",
            "ratings",
            "rated",
            "score",
            "scores",
            "stars",
        } & words:
            return "rank_rating"

        # Smart fallback:
        # "top romance movies", "best horror movies", etc.
        # defaults to highest-rated movies.
        return "rank_rating"

    # --------------------------------------------------------
    # 4. Standalone budget query
    # --------------------------------------------------------
    if {
        "budget",
        "budgets",
        "cost",
        "costs",
        "spent",
        "funding",
    } & words:
        return "ask_budget"

    return None


# ============================================================
# MULTI-ATTRIBUTE DETECTION
# ============================================================
# Allows one message to ask for multiple facts about the SAME
# movie.
#
# Examples:
#   "toy story genre rating budget"
#   "genre and collection of avatar"
#   "Titanic rating and popularity"
#
# The title should already be stripped before this function
# is called.
# ============================================================

ASK_TAG_KEYWORDS = [
    (
        "ask_genre",
        {"genre", "genres", "category"},
        [
            "type of movie",
            "kind of movie",
            "categorize this movie",
        ],
    ),

    (
        "ask_collection",
        {"collection", "franchise"},
        [
            "part of a series",
            "belongs to a collection",
            "part of a collection",
        ],
    ),

    (
        "ask_runtime",
        {"runtime", "duration", "minutes"},
        [
            "run time",
            "how long is",
            "length of the movie",
        ],
    ),

    (
        "ask_rating",
        {
            "rating",
            "ratings",
            "rated",
            "score",
            "scores",
            "stars",
        },
        [
            "vote average",
            "user score",
        ],
    ),

    (
        "ask_revenue",
        {
            "revenue",
            "revenues",
            "earnings",
            "earned",
            "gross",
        },
        [
            "box office",
            "made at the box office",
            "how much money did it make",
        ],
    ),

    (
        "ask_budget",
        {
            "budget",
            "budgets",
            "cost",
            "costs",
            "spent",
            "funding",
        },
        [
            "how much did it cost",
            "production cost",
            "production budget",
        ],
    ),

    (
        "ask_summary",
        {
            "summary",
            "overview",
            "plot",
            "synopsis",
            "storyline",
        },
        [
            "what is it about",
            "what's it about",
            "tell me the plot",
        ],
    ),

    (
        "ask_language",
        {"language", "languages"},
        [
            "spoken in",
            "what language",
        ],
    ),

    (
        "ask_release_date",
        {"released"},
        [
            "release date",
            "when was it released",
            "when did it come out",
            "release year",
        ],
    ),

    (
        "ask_production_companies",
        {"studio", "studios"},
        [
            "production company",
            "production companies",
            "produced by",
            "made by",
            "which studio",
            "what studio",
        ],
    ),

    (
        "ask_production_countries",
        {"country", "countries"},
        [
            "production country",
            "production countries",
            "filmed in",
            "shot in",
            "made in",
            "which country",
            "what country",
        ],
    ),

    (
        "ask_vote_count",
        {
            "votes",
            "voted",
            "voting",
            "count",
            "counts",
        },
        [
            "vote count",
            "number of votes",
            "how many votes",
            "how many people voted",
        ],
    ),

    (
        "ask_popularity",
        {
            "popularity",
            "popular",
            "trending",
            "famous",
        },
        [],
    ),
]


def detect_requested_attributes(text):
    """
    Finds all ask_* attributes requested in a title-stripped
    question.

    Example:
        "genre rating budget"
    returns:
        ["ask_genre", "ask_rating", "ask_budget"]
    """

    normalized = normalize_text(text)
    tokens = set(normalized.split())

    found = []

    for tag, token_set, phrases in ASK_TAG_KEYWORDS:
        hit = (
            bool(tokens & token_set)
            or any(
                phrase in normalized
                for phrase in phrases
            )
        )

        if hit:
            found.append(tag)

    return found


# ============================================================
# MULTI-ATTRIBUTE RESPONSE FORMATTER
# ============================================================

def format_intent_response(tag, row):
    """
    Formats one ask_* response for a movie row.

    This is also used repeatedly when the user asks for
    multiple attributes about the same movie.
    """

    if tag == "search_movie":
        return (
            f"🎬 **Movie:** {row['title']}\n"
            f"🔹 Collection: {row['belongs_to_collection']}\n"
            f"🔹 Genres: {row['genres']}\n"
            f"🔹 Languages: {row['spoken_languages']}\n"
            f"🔹 Runtime: {int(row['runtime'])} minutes\n"
            f"🔹 Rating: {row['vote_average']}/10 "
            f"({int(row['vote_count']):,} votes)\n"
            f"🔹 Popularity: {row['popularity']:.1f}\n"
            f"🔹 Budget: ${row['budget']:,.0f}\n"
            f"🔹 Revenue: ${row['revenue']:,.0f}\n"
            f"🔹 Production: {row['production_companies']}\n"
            f"🔹 Country: {row['production_countries']}\n"
            f"🔹 Tagline: \"{row['tagline']}\"\n"
            f"📝 Summary: {row['overview']}"
        )

    elif tag == "ask_genre":
        return (
            f"🎭 **{row['title']}** belongs to {row['genres']}."
        )

    elif tag == "ask_runtime":
        return (
            f"⏱️ **{row['title']}** runs for "
            f"{int(row['runtime'])} minutes."
        )

    elif tag == "ask_rating":
        return (
            f"⭐ **{row['title']}** has a rating of "
            f"{row['vote_average']}/10."
        )

    elif tag == "ask_collection":
        if row["belongs_to_collection"] == "Unknown":
            return (
                f"📦 **{row['title']}** is not part of "
                f"a known collection."
            )

        return (
            f"📦 **{row['title']}** is part of "
            f"{row['belongs_to_collection']}."
        )

    elif tag == "ask_revenue":
        return (
            f"💰 **{row['title']}** earned "
            f"${row['revenue']:,.0f}."
        )

    elif tag == "ask_budget":
        if not has_reliable_budget(row["budget"]):
            return (
                f"⚠️ **{row['title']}** has no reliable "
                f"budget recorded in this dataset."
            )

        return (
            f"🎥 **{row['title']}** had a budget of "
            f"${row['budget']:,.0f}."
        )

    elif tag == "ask_summary":
        return (
            f"📝 **{row['title']}** — {row['overview']}"
        )

    elif tag == "ask_language":
        return (
            f"🗣️ **{row['title']}** is available in: "
            f"{row['spoken_languages']}."
        )

    elif tag == "ask_release_date":
        return (
            f"📅 **{row['title']}** was released on "
            f"{row['release_date'].strftime('%B %d, %Y')}."
        )

    elif tag == "ask_production_companies":
        if row["production_companies"] == "Unknown":
            return (
                f"🏢 **{row['title']}** has no listed "
                f"production company."
            )

        return (
            f"🏢 **{row['title']}** was produced by "
            f"{row['production_companies']}."
        )

    elif tag == "ask_production_countries":
        if row["production_countries"] == "Unknown":
            return (
                f"🌍 **{row['title']}** has no listed "
                f"production country."
            )

        return (
            f"🌍 **{row['title']}** was produced in "
            f"{row['production_countries']}."
        )

    elif tag == "ask_vote_count":
        return (
            f"🗳️ **{row['title']}** has "
            f"{int(row['vote_count']):,} votes."
        )

    elif tag == "ask_popularity":
        return (
            f"🔥 **{row['title']}** has a popularity "
            f"score of {row['popularity']:.1f}."
        )

    return None


# ============================================================
# MAIN CHATBOT
# ============================================================

def chatbot_response(user_message, algorithm="hybrid"):
    """
    Main chatbot orchestration function.

    Supports:
        - normal single-movie questions
        - movie rankings
        - movie comparisons
        - budget/revenue comparisons
        - multiple ask_* attributes for one movie

    Always returns the dictionary format expected by Streamlit:
        {
            "response": str,
            "intent": str,
            "scores": dict
        }
    """

    # --------------------------------------------------------
    # STEP 0: Input validation
    # --------------------------------------------------------

    if not isinstance(user_message, str):
        return {
            "response": "❌ Please enter a text question.",
            "intent": "N/A",
            "scores": {},
        }

    user_message = user_message.strip()

    if not user_message:
        return {
            "response": "❌ Please enter a question.",
            "intent": "N/A",
            "scores": {},
        }

    # --------------------------------------------------------
    # STEP 1: Correct spelling
    # --------------------------------------------------------

    corrected_message = correct_spelling(
        user_message
    )

    # --------------------------------------------------------
    # STEP 2: Find multiple movie titles
    #
    # This is mainly for comparison queries.
    # --------------------------------------------------------

    comparison_titles = find_movies_in_message(
        corrected_message,
        max_movies=15,
    )

    # --------------------------------------------------------
    # STEP 3: Detect ranking / comparison intent
    #
    # These deterministic rules run before ML so ranking and
    # comparison requests are not misunderstood by the model.
    # --------------------------------------------------------

    forced_tag = detect_supported_intent(
        corrected_message,
        comparison_titles,
    )

    # --------------------------------------------------------
    # STEP 4: Ranking
    #
    # Rankings do NOT require a movie title.
    # --------------------------------------------------------

    if forced_tag in RANKING_CONFIG:
        ranking_response = rank_movies(
            forced_tag,
            corrected_message,
        )

        return {
            "response": ranking_response,
            "intent": forced_tag,
            "scores": {},
        }

    # --------------------------------------------------------
    # STEP 5: Find normal single movie title
    # --------------------------------------------------------

    matched_title = find_movie_in_message(
        corrected_message
    )

    # --------------------------------------------------------
    # STEP 6: Remove movie titles before ML intent detection
    #
    # This prevents words from movie titles from being
    # interpreted as intent keywords.
    # --------------------------------------------------------

    titles_to_strip = list(
        dict.fromkeys(
            comparison_titles
            + (
                [matched_title]
                if matched_title
                else []
            )
        )
    )

    intent_input = corrected_message

    for title in titles_to_strip:
        intent_input = strip_title_words(
            intent_input,
            title,
        )

    # --------------------------------------------------------
    # STEP 7: Predict normal intent
    # --------------------------------------------------------

    if not intent_input.strip():
        predictions = (
            [("search_movie", 1.0)]
            if matched_title
            else []
        )
    else:
        predictions = predict_class(
            intent_input,
            algorithm=algorithm,
            error_threshold=0.10,
        )

    # --------------------------------------------------------
    # STEP 8: Determine final intent
    # --------------------------------------------------------

    if forced_tag is not None:
        tag = forced_tag

    elif predictions:
        tag = predictions[0][0]

    elif matched_title is not None:
        tag = "search_movie"

    else:
        tag = None

    # --------------------------------------------------------
    # STEP 9: No recognized intent
    # --------------------------------------------------------

    if tag is None:
        return {
            "response": (
                f"🤔 I'm not sure I understood that. "
                f"(Model: {algorithm.upper()})\n"
                f"Try asking about a movie's genre, budget, "
                f"rating, runtime, plot, language, release date, "
                f"popularity, vote count, rankings, or movie "
                f"comparisons."
            ),
            "intent": "N/A",
            "scores": dict(predictions),
        }

    # --------------------------------------------------------
    # STEP 10: Greeting / goodbye
    #
    # Preserve the existing GitHub chatbot behavior.
    # --------------------------------------------------------

    if tag in {"greeting", "goodbye"}:
        return {
            "response": generate_response(tag),
            "intent": tag,
            "scores": dict(predictions),
        }

    # --------------------------------------------------------
    # STEP 11: Financial comparison
    # --------------------------------------------------------

    if tag == "compare_budget_revenue":

        if comparison_titles:
            comparison_response = (
                compare_budget_and_revenue(
                    comparison_titles,
                    corrected_message,
                )
            )

            return {
                "response": comparison_response,
                "intent": tag,
                "scores": dict(predictions),
            }

        return {
            "response": (
                "❌ Please include a movie title, "
                "or two movie titles if you want me "
                "to compare both movies."
            ),
            "intent": tag,
            "scores": dict(predictions),
        }

    # --------------------------------------------------------
    # STEP 12: General movie comparison
    # --------------------------------------------------------

    if tag == "compare_movies":

        if len(comparison_titles) < 2:
            return {
                "response": (
                    "❌ Please include two movie titles "
                    "so I can compare them."
                ),
                "intent": tag,
                "scores": dict(predictions),
            }

        comparison_response = compare_movies(
            comparison_titles,
            corrected_message,
        )

        return {
            "response": comparison_response,
            "intent": tag,
            "scores": dict(predictions),
        }

    # --------------------------------------------------------
    # STEP 13: All remaining movie intents require a movie
    # --------------------------------------------------------

    if matched_title is None:
        return {
            "response": (
                "❌ I couldn't identify the movie title. "
                "Please include the movie name."
            ),
            "intent": tag,
            "scores": dict(predictions),
        }

    # --------------------------------------------------------
    # STEP 14: Retrieve movie row
    #
    # Use your existing get_movie_row() rather than accessing
    # df_cleaned directly here.
    # --------------------------------------------------------

    row = get_movie_row(matched_title)

    if row is None:
        return {
            "response": (
                "❌ I found a possible movie title, "
                "but couldn't retrieve its database record."
            ),
            "intent": tag,
            "scores": dict(predictions),
        }

    # --------------------------------------------------------
    # STEP 15: Full movie information
    # --------------------------------------------------------

    if tag == "search_movie":
        return {
            "response": format_intent_response(
                "search_movie",
                row,
            ),
            "intent": tag,
            "scores": dict(predictions),
        }

    # --------------------------------------------------------
    # STEP 16: Multiple attributes for ONE movie
    #
    # Examples:
    #   "Titanic genre rating"
    #   "Avatar budget revenue popularity"
    # --------------------------------------------------------

    requested_tags = detect_requested_attributes(
        intent_input
    )

    if len(requested_tags) >= 2:

        response_lines = [
            format_intent_response(
                requested_tag,
                row,
            )
            for requested_tag in requested_tags
        ]

        response_lines = [
            line
            for line in response_lines
            if line
        ]

        if response_lines:
            return {
                "response": "\n".join(
                    response_lines
                ),
                "intent": "multi_attribute",
                "scores": dict(predictions),
            }

    # --------------------------------------------------------
    # STEP 17: Normal single-attribute response
    #
    # Keep the existing response_formatter so your original
    # chatbot behavior remains intact.
    # --------------------------------------------------------

    response = generate_response(
        tag,
        row,
    )

    # Safety fallback in case the existing formatter doesn't
    # have a response for the tag.
    if not response:
        response = format_intent_response(
            tag,
            row,
        )

    return {
        "response": (
            response
            or "🤔 Sorry, I didn't quite catch that."
        ),
        "intent": tag,
        "scores": dict(predictions),
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "chatbot_response",
]