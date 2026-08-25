# ============================================================
# COMPARISON HELPERS
# ============================================================

from .models import df_cleaned
from .text_processing import normalize_text
from .movie_matching import title_norm_to_original
from .ranking import has_reliable_budget

def get_movie_rows(movie_titles):
    rows = []
    for title in movie_titles:
        title_key = normalize_text(title)
        original_title = title_norm_to_original.get(title_key)
        if original_title is None:
            continue
        matches = df_cleaned[df_cleaned["title"].astype(str).map(normalize_text) == title_key]
        if matches.empty:
            continue
        sort_cols = [c for c in ["popularity", "vote_count", "revenue", "budget"] if c in matches.columns]
        if sort_cols:
            matches = matches.sort_values(by=sort_cols, ascending=False)
        rows.append(matches.iloc[0])
    return rows

def requested_comparison_metrics(user_message):
    # Using raw lowercase instead of normalize_text to prevent NLP stemmers from breaking the keywords
    text = user_message.lower()
    metrics = []

    if any(word in text for word in ["budget", "cost", "spent", "expensive", "cheap"]):
        metrics.append("budget")
    if any(word in text for word in ["revenue", "earn", "gross", "money", "income", "profit"]):
        metrics.append("revenue")
    if any(word in text for word in ["rating", "score", "rate", "star"]):
        metrics.append("rating")
    if any(word in text for word in ["popular", "trend", "famous"]):
        metrics.append("popularity")
    if any(word in text for word in ["runtime", "duration", "long", "short"]):
        metrics.append("runtime")
    # Added "vot" just in case a stemmer shortened it before it even got here
    if any(word in text for word in ["vote", "vot"]):
        metrics.append("vote_count")

    return metrics

def compare_movies(movie_titles, user_message):
    if len(movie_titles) > 5:
        return "⚠️ You can only compare 5 movies total."

    rows = get_movie_rows(movie_titles)

    if len(rows) < 2:
        return "❌ I could not retrieve at least two valid movie records."

    # 1. Fetch metrics based on user keywords
    metrics = requested_comparison_metrics(user_message)

    # 2. If NO specific metrics were detected, fallback to the default 4
    if not metrics:
        metrics = ["rating", "budget", "revenue", "popularity"]

    label_map = {
        "budget": ("Budget", "budget", lambda x: f"${x:,.0f}"),
        "revenue": ("Revenue", "revenue", lambda x: f"${x:,.0f}"),
        "rating": ("Rating", "vote_average", lambda x: f"{x:.1f}/10"),
        "popularity": ("Popularity", "popularity", lambda x: f"{x:.1f}"),
        "runtime": ("Runtime", "runtime", lambda x: f"{int(x)} minutes"),
        "vote_count": ("Vote Count", "vote_count", lambda x: f"{int(x):,}")
    }

    # Check if the user specifically asked for lower/lowest
    text_words = user_message.lower().replace("?", "").replace(",", " ").split()
    lowest_words = {"lowest", "low", "lower", "least", "worse", "worst", "smallest", "shortest", "fewest", "cheapest","less","lesser"}
    find_lowest = bool(set(text_words) & lowest_words)

    titles_header = " vs ".join([r['title'] for r in rows])
    lines = [f"⚖️ **Comparison: {titles_header}**"]

    for metric in metrics:
        label, column, formatter = label_map[metric]
        lines.append(f"\n### {label}\n")

        for r in rows:
            lines.append(f"- **{r['title']}**: {formatter(r[column])}")
            lines.append("")

        # Calculate lowest or highest dynamically
        if find_lowest:
            target_val = min([r[column] for r in rows])
            direction_label = "Lower/Lowest"
        else:
            target_val = max([r[column] for r in rows])
            direction_label = "Higher/Highest"
        winners = [r['title'] for r in rows if r[column] == target_val]
        winner_str = "All movies are equal" if len(winners) == len(rows) else ", ".join(winners)

        lines.append(f"**{direction_label} {label.lower()}:** {winner_str}")

    # Only show financial profit if BOTH budget and revenue are being compared
    if "budget" in metrics and "revenue" in metrics:
        for row in rows:
            if not has_reliable_budget(row["budget"]):
                lines.append(f"\n⚠️ **{row['title']}** has no reported budget in this dataset.")
                continue
            profit_difference = row["revenue"] - row["budget"]
            lines.append(f"\n💵 **{row['title']} financial difference:** ${profit_difference:,.0f} (Revenue − Budget)")

    return "\n".join(lines)


def compare_budget_and_revenue(movie_titles, user_message):
    if len(movie_titles) > 5:
        return "⚠️ You can only compare 5 movies total."

    rows = get_movie_rows(movie_titles)

    if not rows:
        return "❌ I couldn't find the movie record. Please include a movie title."

    if len(rows) == 1:
        row = rows[0]
        if not has_reliable_budget(row["budget"]):
            return (f"⚠️ **{row['title']}** has no reported budget in this dataset, so profitability can't be determined.\n"
                    f"• Revenue: ${row['revenue']:,.0f}")
        difference = row["revenue"] - row["budget"]
        result = "✅ Revenue is greater than budget." if difference > 0 else "❌ Revenue is not greater than budget."
        return (f"💰 **{row['title']}**\n"
                f"• Budget: ${row['budget']:,.0f}\n"
                f"• Revenue: ${row['revenue']:,.0f}\n"
                f"• Revenue − Budget: ${difference:,.0f}\n"
                f"• {result}")

    titles_header = " vs ".join([r['title'] for r in rows])
    lines = [f"💰 **Budget and Revenue Comparison:** {titles_header}", "", "### Budget", ""]

    for r in rows:
        lines.append(f"- **{r['title']}**: ${r['budget']:,.0f}")
        lines.append("")

    lines.extend(["", "### Revenue", ""])
    for r in rows:
        lines.append(f"- **{r['title']}**: ${r['revenue']:,.0f}")
        lines.append("")

    max_budget = max([r["budget"] for r in rows])
    b_winners = [r["title"] for r in rows if r["budget"] == max_budget]
    b_winner = "All movies are equal" if len(b_winners) == len(rows) else ", ".join(b_winners)

    max_revenue = max([r["revenue"] for r in rows])
    r_winners = [r["title"] for r in rows if r["revenue"] == max_revenue]
    r_winner = "All movies are equal" if len(r_winners) == len(rows) else ", ".join(r_winners)

    lines.extend(["", f"**Higher budget:** {b_winner}", f"**Higher revenue:** {r_winner}"])

    return "\n".join(lines)
