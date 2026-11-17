"""
core/cache.py
-------------
Local JSON cache for CommentLens analysis results.

Functions
---------
load_cache(video_id)                                                -> dict | None
save_cache(video_id, video_info, limit, comments, summary, themes) -> None
clear_cache()                                                       -> int   (files deleted)

Cache directory: ./cache/  (relative to the working directory when the app runs)
Cache filename : ./cache/{video_id}.json

Cache file schema
-----------------
{
  "video":       { ... video_info dict ... },
  "limit":       1000,
  "analyzed_at": "2024-01-15T10:30:00.123456",
  "comments":    [ ... list of comment dicts ... ],
  "summary":     "Plain text summary.",
  "themes":      { "theme label": count, ... }
}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_cache_dir() -> None:
    """Create the cache directory if it does not exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(video_id: str) -> Path:
    return _CACHE_DIR / f"{video_id}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cache(video_id: str) -> Optional[dict]:
    """
    Read a cached analysis for *video_id*.

    Returns
    -------
    dict on success, None if the file does not exist or is corrupt.
    """
    path = _cache_path(video_id)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Cache hit for video_id=%r (%s).", video_id, path)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read cache for video_id=%r: %s", video_id, exc)
        return None


def save_cache(
    video_id: str,
    video_info: dict,
    limit: int,
    comments: list[dict],
    summary: str,
    themes: list[tuple[str, int]] | dict,
) -> None:
    """
    Persist an analysis result to the local cache.

    Parameters
    ----------
    video_id   : YouTube video ID string.
    video_info : dict with title, channel, views, comment_count.
    limit      : The fetch limit that was used.
    comments   : Full list of classified comment dicts.
    summary    : Plain text summary string.
    themes     : Either a list of (theme, count) tuples or a {theme: count} dict.
    """
    _ensure_cache_dir()

    # Normalise themes to a plain dict for JSON serialisation.
    if isinstance(themes, dict):
        themes_dict: dict[str, int] = themes
    else:
        themes_dict = {theme: count for theme, count in themes}

    payload = {
        "video": video_info,
        "limit": limit,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "comments": comments,
        "summary": summary,
        "themes": themes_dict,
    }

    path = _cache_path(video_id)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("Cache saved for video_id=%r (%s).", video_id, path)
    except OSError as exc:
        logger.error("Failed to write cache for video_id=%r: %s", video_id, exc)
        raise


def clear_cache() -> int:
    """
    Delete all JSON files inside the cache directory.

    Returns
    -------
    int — number of files deleted.
    """
    if not _CACHE_DIR.exists():
        return 0

    deleted = 0
    for json_file in _CACHE_DIR.glob("*.json"):
        try:
            json_file.unlink()
            deleted += 1
            logger.info("Deleted cache file: %s", json_file)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", json_file, exc)

    logger.info("clear_cache: deleted %d file(s).", deleted)
    return deleted
