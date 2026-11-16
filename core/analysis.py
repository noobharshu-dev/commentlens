"""
core/analysis.py
----------------
Aggregation, filtering, and search utilities for classified comment lists.

Functions
---------
aggregate_stats(comments)                          -> dict
sentiment_timeline(comments)                       -> pd.DataFrame
theme_counts(comments, top_n=8)                    -> list[tuple[str, int]]
language_breakdown(comments)                       -> dict
filter_palette(comments, palette, show_toxic=False) -> list[dict]
search_rows(rows, query)                           -> list[dict]
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNCLASSIFIED_STATUSES = {"failed", "error"}

_TOP_LIKED_THRESHOLD = 10  # comments with >= this many likes are "top liked"


def _pct(part: int, total: int) -> float:
    """Return percentage rounded to 2 dp, or 0.0 if total is zero."""
    return round(part / total * 100, 2) if total else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_stats(comments: list[dict]) -> dict:
    """
    Compute summary statistics across all classified comments.

    Returns a dict with absolute counts and percentage equivalents
    (keys suffixed with ``_pct``), plus:
        sentiment_score : float  – mean sentiment_score rescaled to 0-100
        sentiment_label : str    – "Mostly negative" / "Mixed" / "Mostly positive"
    """
    total = len(comments)
    if total == 0:
        return {
            "total": 0,
            "sentiment_score": 0.0,
            "sentiment_label": "Mixed",
        }

    positive_count = sum(1 for c in comments if str(c.get("primary_sentiment", "")).lower() == "positive")
    negative_count = sum(1 for c in comments if str(c.get("primary_sentiment", "")).lower() == "negative")
    neutral_count = sum(1 for c in comments if str(c.get("primary_sentiment", "")).lower() == "neutral")
    mixed_count = sum(1 for c in comments if str(c.get("primary_sentiment", "")).lower() == "mixed")
    questions_count = sum(1 for c in comments if c.get("is_question"))
    suggestions_count = sum(1 for c in comments if c.get("is_suggestion"))
    bug_reports_count = sum(1 for c in comments if c.get("is_bug_report"))
    praise_count = sum(1 for c in comments if c.get("is_praise"))
    duplicates_count = sum(1 for c in comments if c.get("is_duplicate"))
    replies_count = sum(1 for c in comments if c.get("is_reply"))
    spam_count = sum(1 for c in comments if c.get("is_spam_or_bot"))
    toxic_count = sum(1 for c in comments if c.get("is_rude_or_toxic"))
    top_liked_count = sum(
        1 for c in comments if c.get("likes", 0) >= _TOP_LIKED_THRESHOLD
    )
    unclassified_count = sum(
        1 for c in comments
        if c.get("classification_status", "pending") in _UNCLASSIFIED_STATUSES
    )

    # Sentiment score: mean of per-comment sentiment_score, rescaled 0-100.
    scores = [
        float(c["sentiment_score"])
        for c in comments
        if isinstance(c.get("sentiment_score"), (int, float))
    ]
    raw_mean = statistics.mean(scores) if scores else 0.0
    # Scores are -1.0 to 1.0; rescale to 0–100.
    sentiment_score = round((raw_mean + 1.0) / 2.0 * 100, 1)

    if sentiment_score < 40:
        sentiment_label = "Mostly negative"
    elif sentiment_score >= 60:
        sentiment_label = "Mostly positive"
    else:
        sentiment_label = "Mixed"

    def row(count: int, name: str) -> tuple[str, int, str, float]:
        return name, count, f"{name}_pct", _pct(count, total)

    pairs = [
        row(positive_count, "positive_count"),
        row(negative_count, "negative_count"),
        row(neutral_count, "neutral_count"),
        row(mixed_count, "mixed_count"),
        row(questions_count, "questions_count"),
        row(suggestions_count, "suggestions_count"),
        row(bug_reports_count, "bug_reports_count"),
        row(praise_count, "praise_count"),
        row(duplicates_count, "duplicates_count"),
        row(replies_count, "replies_count"),
        row(spam_count, "spam_count"),
        row(toxic_count, "toxic_count"),
        row(top_liked_count, "top_liked_count"),
        row(unclassified_count, "unclassified_count"),
    ]

    result: dict = {"total": total}
    for name, count, pct_key, pct_val in pairs:
        result[name] = count
        result[pct_key] = pct_val

    result["sentiment_score"] = sentiment_score
    result["sentiment_label"] = sentiment_label

    return result


def sentiment_timeline(comments: list[dict]) -> pd.DataFrame:
    """
    Build a per-day positive/negative comment count DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        date     (datetime64[ns])
        positive (int)
        negative (int)

    One row per calendar date present in the data, sorted ascending.
    """
    rows = []
    for c in comments:
        date_str = c.get("published_at", "")
        sentiment = str(c.get("primary_sentiment", "")).lower()
        if not date_str or sentiment not in {"positive", "negative"}:
            continue
        rows.append({"date": date_str, "sentiment": sentiment})

    if not rows:
        return pd.DataFrame(columns=["date", "positive", "negative"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    pivot = (
        df.groupby(["date", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure both columns exist.
    for col in ("positive", "negative"):
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[["date", "positive", "negative"]].sort_values("date")
    return pivot.reset_index(drop=True)


def theme_counts(comments: list[dict], top_n: int = 8) -> list[tuple[str, int]]:
    """
    Count ``theme`` values for Suggestion and Criticism comments.

    Returns
    -------
    List of (theme, count) tuples, sorted descending by count, limited to top_n.
    """
    counter: Counter[str] = Counter()
    for c in comments:
        if c.get("is_suggestion") or c.get("is_criticism"):
            theme = c.get("theme", "").strip()
            if theme:
                counter[theme] += 1

    return counter.most_common(top_n)


def language_breakdown(comments: list[dict]) -> dict[str, int]:
    """
    Count comments per detected language.

    Returns
    -------
    dict mapping language label -> count, sorted descending.
    """
    counter: Counter[str] = Counter()
    for c in comments:
        lang = c.get("language", "").strip()
        if lang:
            counter[lang] += 1

    return dict(counter.most_common())


def filter_palette(
    comments: list[dict],
    palette: str,
    show_toxic: bool = False,
) -> list[dict]:
    """
    Filter *comments* according to the named *palette*.

    Parameters
    ----------
    comments   : Full list of comment dicts.
    palette    : One of the named palette strings (see below).
    show_toxic : When palette is "All", include toxic comments if True.

    Valid palettes
    --------------
    "All", "Positive", "Negative", "Neutral", "Mixed",
    "Questions", "Suggestions", "Bug reports", "Praise",
    "Duplicates", "Replies", "Spam", "Rude or toxic",
    "Top liked", "Unclassified"
    """
    p = palette.strip()

    def _is_safe(c: dict) -> bool:
        return show_toxic or not c.get("is_rude_or_toxic", False)

    if p == "All":
        return [c for c in comments if _is_safe(c)]

    if p == "Positive":
        return [c for c in comments if str(c.get("primary_sentiment", "")).lower() == "positive" and _is_safe(c)]

    if p == "Negative":
        return [c for c in comments if str(c.get("primary_sentiment", "")).lower() == "negative" and _is_safe(c)]

    if p == "Neutral":
        return [c for c in comments if str(c.get("primary_sentiment", "")).lower() == "neutral" and _is_safe(c)]

    if p == "Mixed":
        return [c for c in comments if str(c.get("primary_sentiment", "")).lower() == "mixed" and _is_safe(c)]

    if p == "Questions":
        return [c for c in comments if c.get("is_question") and _is_safe(c)]

    if p == "Suggestions":
        return [c for c in comments if c.get("is_suggestion") and _is_safe(c)]

    if p == "Bug reports":
        return [c for c in comments if c.get("is_bug_report") and _is_safe(c)]

    if p == "Praise":
        return [c for c in comments if c.get("is_praise") and _is_safe(c)]

    if p == "Duplicates":
        return [c for c in comments if c.get("is_duplicate") and _is_safe(c)]

    if p == "Replies":
        return [c for c in comments if c.get("is_reply") and _is_safe(c)]

    if p == "Spam":
        return [c for c in comments if c.get("is_spam_or_bot")]

    if p == "Rude or toxic":
        return [c for c in comments if c.get("is_rude_or_toxic")]

    if p == "Top liked":
        # FR-24: "Top liked | All, sorted by likes descending"
        safe_comments = [c for c in comments if _is_safe(c)]
        return sorted(safe_comments, key=lambda c: c.get("likes", 0), reverse=True)

    if p == "Unclassified":
        return [
            c for c in comments
            if c.get("classification_status", "pending") in _UNCLASSIFIED_STATUSES
        ]

    raise ValueError(
        f"Unknown palette {palette!r}. "
        "Valid palettes: All, Positive, Negative, Neutral, Mixed, "
        "Questions, Suggestions, Bug reports, Praise, Duplicates, "
        "Replies, Spam, Rude or toxic, Top liked, Unclassified."
    )


def search_rows(rows: list[dict], query: str) -> list[dict]:
    """
    Return rows where *query* appears (case-insensitive) in
    ``original_text`` or ``ai_summary``.
    """
    if not query:
        return list(rows)

    q = query.lower()
    return [
        r for r in rows
        if q in r.get("original_text", "").lower()
        or q in r.get("ai_summary", "").lower()
    ]
