"""
core/export.py
--------------
CSV export helpers for CommentLens.

Functions
---------
export_csv(comments, palette=None)                    -> bytes  (UTF-8 BOM)
export_filename(video_id, palette=None, date_str=None) -> str
"""

from __future__ import annotations

import io
from datetime import date
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column order per MRD FR-28
# ---------------------------------------------------------------------------

_COLUMNS = [
    "comment_id",
    "youtube_id",
    "parent_id",
    "is_reply",
    "author",
    "published_at",
    "original_text",
    "likes",
    "reply_count",
    # Cleaning fields
    "is_emoji_only",
    "is_duplicate",
    "duplicate_count",
    "classification_status",
    # Classification fields
    "primary_sentiment",
    "sentiment_score",
    "primary_category",
    "is_question",
    "is_praise",
    "is_criticism",
    "is_suggestion",
    "is_bug_report",
    "is_rude_or_toxic",
    "is_spam_or_bot",
    "is_off_topic",
    "theme",
    "language",
    "priority",
    "ai_summary",
    "suggested_reply",
]


def export_csv(
    comments: list[dict],
    palette: Optional[str] = None,
) -> bytes:
    """
    Convert a list of comment dicts to a UTF-8 BOM CSV byte string.

    Columns are ordered per MRD FR-28.  Any column not present in a comment
    dict is filled with an empty string.

    Parameters
    ----------
    comments : List of classified comment dicts.
    palette  : Optional palette label — currently only used for naming; not
               filtered here (caller is responsible for pre-filtering).

    Returns
    -------
    bytes — UTF-8 BOM encoded CSV, suitable for Streamlit st.download_button.
    """
    if not comments:
        # Return an empty CSV with the correct header.
        df = pd.DataFrame(columns=_COLUMNS)
    else:
        df = pd.DataFrame(comments)

        # Add any missing columns as empty strings.
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # Reorder to canonical column list; drop any extra columns.
        df = df[_COLUMNS]

    buffer = io.BytesIO()
    # utf-8-sig adds the BOM so Excel opens the file correctly.
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


def export_filename(
    video_id: str,
    palette: Optional[str] = None,
    date_str: Optional[str] = None,
) -> str:
    """
    Build a canonical export filename.

    Format
    ------
    ``commentlens_{video_id}_{palette}_{date}.csv``
    ``commentlens_{video_id}_all_{date}.csv``  (when palette is None or "All")

    Parameters
    ----------
    video_id  : YouTube video ID string.
    palette   : Optional palette label string.
    date_str  : Optional ISO date string (defaults to today).

    Returns
    -------
    Filename string (no directory component).
    """
    if date_str is None:
        date_str = date.today().isoformat()

    # Normalise palette to a safe filename component.
    if not palette or palette.strip().lower() == "all":
        palette_part = "all"
    else:
        # Lowercase, replace spaces / slashes with underscore, strip special chars.
        safe = palette.strip().lower()
        safe = safe.replace(" ", "_").replace("/", "_")
        # Keep only alphanumeric and underscores.
        safe = "".join(ch for ch in safe if ch.isalnum() or ch == "_")
        palette_part = safe or "all"

    return f"commentlens_{video_id}_{palette_part}_{date_str}.csv"
