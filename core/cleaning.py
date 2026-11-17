"""
core/cleaning.py
----------------
Pre-processing utilities for raw YouTube comment lists.

Functions
---------
clean_comments(raw_list)          -> list[dict]
flag_duplicates(comments)         -> list[dict]   (mutates in place, returns list)
make_batches(comments, size=40)   -> generator[list[dict]]
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Generator

# ---------------------------------------------------------------------------
# Emoji-only detection regex
# Matches strings that contain only emoji/symbol codepoints and whitespace.
# ---------------------------------------------------------------------------
_EMOJI_RE = re.compile(
    r"^[\U0001F000-\U0001FFFF\u2600-\u26FF\u2700-\u27BF\uFE00-\uFE0F\u200D\u20E3\s]+$"
)


def _is_emoji_only(text: str) -> bool:
    """Return True if *text* consists solely of emoji/symbols and whitespace."""
    return bool(_EMOJI_RE.match(text))


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_comments(raw_list: list[dict]) -> list[dict]:
    """
    Sanitise a raw list of comment dicts returned by youtube.fetch_comments().

    Steps
    -----
    1. Drop entries whose ``original_text`` is empty after stripping.
    2. Assign a 0-indexed integer ``comment_id``.
    3. Set ``is_emoji_only`` flag.
    4. Inject default fields expected by the classifier:
       - ``is_duplicate``          = False
       - ``duplicate_count``       = 1
       - ``classification_status`` = "pending"

    Returns
    -------
    New list of cleaned dicts (original dicts are shallow-copied).
    """
    cleaned: list[dict] = []
    counter = 0

    for raw in raw_list:
        text: str = raw.get("original_text") or ""
        if not text.strip():
            continue  # drop empty

        entry = dict(raw)  # shallow copy
        entry["original_text"] = text.strip()
        entry["comment_id"] = counter
        entry["is_emoji_only"] = _is_emoji_only(entry["original_text"])
        entry["is_duplicate"] = False
        entry["duplicate_count"] = 1
        entry["classification_status"] = "pending"

        cleaned.append(entry)
        counter += 1

    return cleaned


def flag_duplicates(comments: list[dict]) -> list[dict]:
    """
    Detect duplicate comments by normalised text.

    A comment is flagged when its normalised text appears 2 or more times
    across the full list.  When flagged:
    - ``is_duplicate``    is set to ``True``
    - ``duplicate_count`` is set to the total occurrence count

    The *first* occurrence of a text is flagged the same as subsequent ones;
    all copies carry the same ``duplicate_count``.

    Mutates the dicts in *comments* in place and returns the same list.
    """
    # Count normalised text occurrences.
    norm_counts: Counter[str] = Counter(
        _normalize(c.get("original_text") or "") for c in comments
    )

    for comment in comments:
        norm = _normalize(comment.get("original_text") or "")
        count = norm_counts[norm]
        if count >= 2:
            comment["is_duplicate"] = True
            comment["duplicate_count"] = count

    return comments


def make_batches(
    comments: list[dict], size: int = 40
) -> Generator[list[dict], None, None]:
    """
    Yield successive sub-lists of *size* from *comments*.

    The last batch may be smaller than *size*.

    Parameters
    ----------
    comments : Full list of comment dicts.
    size     : Maximum number of comments per batch.

    Yields
    ------
    list[dict]
    """
    if size < 1:
        raise ValueError(f"Batch size must be >= 1, got {size!r}.")

    for start in range(0, len(comments), size):
        yield comments[start : start + size]
