"""
core/youtube.py
---------------
YouTube Data API v3 helpers for CommentLens.

Functions
---------
extract_video_id(url_or_id) -> str
get_video_info(video_id, api_key=None) -> dict
fetch_comments(video_id, api_key=None, limit=1000, on_progress=None) -> list[dict]

Custom exceptions
-----------------
CommentsDisabledError
"""

from __future__ import annotations

import os
import re
import urllib.parse
from typing import Callable, Optional

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from core.diagnostics import session_diagnostics

load_dotenv()

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CommentsDisabledError(Exception):
    """Raised when a video has comments disabled."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_PATTERNS = [
    # https://www.youtube.com/watch?v=VIDEO_ID
    re.compile(r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})"),
    # https://youtu.be/VIDEO_ID
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    # https://www.youtube.com/shorts/VIDEO_ID
    re.compile(r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"),
    # https://www.youtube.com/live/VIDEO_ID
    re.compile(r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})"),
    # https://www.youtube.com/embed/VIDEO_ID
    re.compile(r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"),
    # https://www.youtube.com/v/VIDEO_ID
    re.compile(r"(?:youtube\.com/v/)([A-Za-z0-9_-]{11})"),
]


def _handle_http_error(exc: HttpError, video_id: str = "") -> None:
    """Parse Google API HttpError into human-readable, safe exceptions without leaking credentials."""
    status = getattr(exc.resp, "status", None)
    content = getattr(exc, "content", b"")
    reason = ""
    message = ""
    try:
        import json
        detail = json.loads(content.decode("utf-8", errors="ignore"))
        err_obj = detail.get("error", {})
        errors = err_obj.get("errors", [])
        message = err_obj.get("message", "")
        if errors:
            reason = errors[0].get("reason", "")
    except Exception:
        pass

    if status == 403 and reason == "commentsDisabled":
        raise CommentsDisabledError("Comments are turned off for this video. Try another video.") from exc

    if status == 403 and reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
        raise ValueError(
            "YouTube Data API daily quota exceeded. Google Cloud provides 10,000 units/day for free. "
            "Please wait for your quota to reset at midnight Pacific Time or try the sample demo video."
        ) from exc

    if status == 400 or "API key not valid" in message or reason == "keyInvalid":
        raise ValueError(
            "The YouTube API key provided is not valid. A valid Google Cloud YouTube Data API v3 key "
            "starts with 'AIzaSy' and is 39 characters long. Please check your key in the sidebar settings "
            "or Google Cloud Console (APIs & Services > Credentials)."
        ) from exc

    if status == 404:
        raise ValueError(
            f"No video found for ID '{video_id}'. Please ensure the video link is public and correct."
        ) from exc

    # Sanitize any key= parameters from raw message so keys are never leaked in UI
    clean_msg = re.sub(r"key=[^&\s'\"]+", "key=REDACTED", str(exc))
    raise ValueError(f"YouTube API request failed: {clean_msg}") from exc


def _build_client(api_key: Optional[str] = None):
    """Build a YouTube API client using the provided or env-sourced API key."""
    key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
    if not key or key.strip().startswith("your_"):
        raise ValueError(
            "YouTube API key not found or not configured. Set YOUTUBE_API_KEY in the sidebar settings or .env file."
        )
    key_clean = key.strip().strip("'\"")
    if not key_clean.startswith("AIza") or len(key_clean) < 30:
        raise ValueError(
            f"The YouTube API key '{key_clean[:4]}...' appears invalid or truncated. "
            "A valid Google Cloud YouTube Data API v3 key is 39 characters long and starts with 'AIzaSy'. "
            "Please copy the full key from Google Cloud Console (APIs & Services > Credentials)."
        )
    return build("youtube", "v3", developerKey=key_clean)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_video_id(url_or_id: str) -> str:
    """
    Parse a YouTube URL or bare video ID and return the 11-character video ID.

    Accepts:
        - Bare 11-char video IDs
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID

    Raises:
        ValueError: if the input cannot be parsed as a valid YouTube video ID.
    """
    url_or_id = url_or_id.strip()

    # If it already looks like a bare video ID return it immediately.
    if _VIDEO_ID_RE.match(url_or_id):
        return url_or_id

    # Try URL patterns.
    for pattern in _PATTERNS:
        match = pattern.search(url_or_id)
        if match:
            return match.group(1)

    # Last attempt: parse query string from whatever URL was given.
    parsed = urllib.parse.urlparse(url_or_id)
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs:
        vid = qs["v"][0]
        if _VIDEO_ID_RE.match(vid):
            return vid

    raise ValueError(
        "This does not look like a YouTube video link. Try one like youtube.com/watch?v=..."
    )


def get_video_info(video_id: str, api_key: Optional[str] = None) -> dict:
    """
    Fetch basic video metadata from the YouTube Data API v3.

    Returns
    -------
    dict with keys:
        title        : str
        channel      : str
        views        : int
        comment_count: int
    """
    session_diagnostics.info("FETCH", f"Fetching metadata for YouTube video: {video_id}")
    client = _build_client(api_key)
    try:
        response = (
            client.videos()
            .list(part="snippet,statistics", id=video_id)
            .execute()
        )
    except HttpError as exc:
        _handle_http_error(exc, video_id)

    items = response.get("items", [])
    if not items:
        session_diagnostics.error("FETCH", f"No video found for ID {video_id!r}")
        raise ValueError(
            f"No video found for ID {video_id!r}. Check the ID or ensure the video is public."
        )

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    info = {
        "title": snippet.get("title", "Unknown Title"),
        "channel": snippet.get("channelTitle", "Unknown Channel"),
        "views": int(stats.get("viewCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
    }
    session_diagnostics.success(
        "FETCH",
        f"Retrieved video info: '{info['title']}' ({info['views']:,} views, {info['comment_count']:,} comments)",
    )
    return info


def fetch_comments(
    video_id: str,
    api_key: Optional[str] = None,
    limit: int = 1000,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """
    Fetch up to *limit* comments (top-level + replies) for *video_id*.

    Parameters
    ----------
    video_id    : YouTube video ID string.
    api_key     : Optional API key override (falls back to YOUTUBE_API_KEY env var).
    limit       : Maximum total comments to return (top-level + replies combined).
    on_progress : Optional callback(fetched_so_far, total_estimate).

    Returns
    -------
    List of dicts with keys:
        youtube_id      : str   – unique comment ID from YouTube
        parent_id       : str | None – None for top-level comments
        is_reply        : bool
        author          : str
        published_at    : str   – "YYYY-MM-DD"
        original_text   : str
        likes           : int
        reply_count     : int   – 0 for replies

    Raises
    ------
    CommentsDisabledError : if the video has comments disabled.
    """
    session_diagnostics.info("FETCH", f"Starting comment fetch for video {video_id} (limit={limit})...")
    client = _build_client(api_key)
    comments: list[dict] = []

    def _parse_top_level(item: dict) -> dict:
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        raw_date = snippet.get("publishedAt", "")[:10]  # "YYYY-MM-DD"
        return {
            "youtube_id": item["snippet"]["topLevelComment"]["id"],
            "parent_id": None,
            "is_reply": False,
            "author": snippet.get("authorDisplayName", ""),
            "published_at": raw_date,
            "original_text": snippet.get("textOriginal", snippet.get("textDisplay", "")),
            "likes": int(snippet.get("likeCount", 0)),
            "reply_count": int(item["snippet"].get("totalReplyCount", 0)),
        }

    def _parse_reply(reply: dict, parent_id: str) -> dict:
        snippet = reply["snippet"]
        raw_date = snippet.get("publishedAt", "")[:10]
        return {
            "youtube_id": reply["id"],
            "parent_id": parent_id,
            "is_reply": True,
            "author": snippet.get("authorDisplayName", ""),
            "published_at": raw_date,
            "original_text": snippet.get("textOriginal", snippet.get("textDisplay", "")),
            "likes": int(snippet.get("likeCount", 0)),
            "reply_count": 0,
        }

    page_token: Optional[str] = None

    try:
        while len(comments) < limit:
            max_results = min(100, limit - len(comments))

            request_kwargs: dict = dict(
                part="snippet,replies",
                videoId=video_id,
                maxResults=max_results,
                order="relevance",
                textFormat="plainText",
            )
            if page_token:
                request_kwargs["pageToken"] = page_token

            response = client.commentThreads().list(**request_kwargs).execute()

            for item in response.get("items", []):
                if len(comments) >= limit:
                    break

                top = _parse_top_level(item)
                comments.append(top)

                # Inline replies (up to 5 from API)
                replies_data = item.get("replies", {}).get("comments", [])
                for reply_item in replies_data:
                    if len(comments) >= limit:
                        break
                    comments.append(_parse_reply(reply_item, top["youtube_id"]))

            if on_progress:
                on_progress(len(comments), limit)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    except HttpError as exc:
        _handle_http_error(exc, video_id)

    session_diagnostics.success(
        "FETCH",
        f"Comment fetch complete. Total retrieved: {len(comments):,} comments.",
    )
    return comments
