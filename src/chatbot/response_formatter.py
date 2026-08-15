"""
Response formatting based on intents and movie data.
"""
import random
import pandas as pd

from .models import intents


def get_intent_response(tag):
    """Get a random response template for the given intent tag."""
    for intent in intents.get("intents", []):
        if intent.get("tag") == tag:
            responses = intent.get("responses", [])

            if responses:
                return random.choice(responses)

            return None

    return None


def format_release_date(row):
    """Format release date from movie row."""
    try:
        rel_date_str = pd.to_datetime(row["release_date"]).strftime('%B %d, %Y')
    except Exception:
        rel_date_str = str(row.get("release_date", "Unknown"))
    
    return rel_date_str


def format_currency(value):
    """Format a numeric value as currency."""
    return f"${value:,.0f}" if pd.notnull(value) else "N/A"


def format_runtime(value):
    """Format a numeric value as runtime."""
    return f"{int(value)} minutes" if pd.notnull(value) else "N/A"


def generate_response(tag, row=None):
    """
    Generate response based on intent tag and movie data.
    
    Args:
        tag: Intent classification tag
        row: Movie DataFrame row (required for all non-greeting/goodbye intents)
        
    Returns:
        Formatted response string
    """
    # Greeting/Goodbye responses
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

    # From here, all responses require movie data
    if row is None:
        return "❌ No movie data available."

    # Format commonly used fields
    rel_date_str = format_release_date(row)
    budget_str = format_currency(row.get("budget", 0))
    revenue_str = format_currency(row.get("revenue", 0))
    runtime_str = format_runtime(row.get("runtime", 0))

    # Tag-specific responses
    if tag == "search_movie":
        return (
            f"🎬 **Movie:** {row['title']}\n\n"
            f"🔹 Collection: {row.get('belongs_to_collection', 'Unknown')}\n\n"
            f"🔹 Genres: {row.get('genres', 'N/A')}\n\n"
            f"🔹 Languages: {row.get('spoken_languages', 'N/A')}\n\n"
            f"🔹 Runtime: {runtime_str}\n\n"
            f"🔹 Rating: {row.get('vote_average', 'N/A')}/10\n\n"
            f"🔹 Budget: {budget_str}\n\n"
            f"🔹 Revenue: {revenue_str}\n\n"
            f"🔹 Tagline: \"{row.get('tagline', '')}\"\n\n"
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
    "generate_response",
    "get_intent_response",
    "format_release_date",
    "format_currency",
    "format_runtime",
]
