"""
Unit and integration tests for CommentLens.
Verifies all functional requirements and business logic from MRD.md.
"""

import io
import json
import os
import sys
from pathlib import Path

# Ensure commentlens directory is in sys.path
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import pytest
import pandas as pd

from core.youtube import extract_video_id
from core.cleaning import clean_comments, flag_duplicates, make_batches
from core.classifier import validate_results, _DEFAULT_RESULT
from core.analysis import (
    aggregate_stats,
    sentiment_timeline,
    theme_counts,
    language_breakdown,
    filter_palette,
    search_rows,
)
from core.creator import pick_top_quotes, build_creator_report, _extract_citations
from core.export import export_csv, export_filename
from core.cache import save_cache, load_cache, clear_cache


# ---------------------------------------------------------------------------
# 1. YouTube URL extraction tests (FR-01)
# ---------------------------------------------------------------------------

def test_extract_video_id_valid_formats():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?feature=shared&v=dQw4w9WgXcQ&t=30") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid_format():
    with pytest.raises(ValueError) as excinfo:
        extract_video_id("https://example.com/not-youtube")
    assert "This does not look like a YouTube video link" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. Cleaning & duplicates tests (FR-07, FR-08, FR-09, FR-10)
# ---------------------------------------------------------------------------

def test_clean_comments_drops_empty_and_detects_emoji():
    raw = [
        {"original_text": "Great video!"},
        {"original_text": "   "},
        {"original_text": "🔥👏"},
        {"original_text": ""},
    ]
    cleaned = clean_comments(raw)
    assert len(cleaned) == 2
    assert cleaned[0]["comment_id"] == 0
    assert cleaned[0]["original_text"] == "Great video!"
    assert cleaned[0]["is_emoji_only"] is False

    assert cleaned[1]["comment_id"] == 1
    assert cleaned[1]["original_text"] == "🔥👏"
    assert cleaned[1]["is_emoji_only"] is True


def test_flag_duplicates():
    raw = [
        {"original_text": "First comment"},
        {"original_text": "first comment  "},
        {"original_text": "Unique comment"},
    ]
    cleaned = clean_comments(raw)
    flagged = flag_duplicates(cleaned)
    assert flagged[0]["is_duplicate"] is True
    assert flagged[0]["duplicate_count"] == 2
    assert flagged[1]["is_duplicate"] is True
    assert flagged[1]["duplicate_count"] == 2
    assert flagged[2]["is_duplicate"] is False
    assert flagged[2]["duplicate_count"] == 1


def test_make_batches():
    items = [{"comment_id": i} for i in range(95)]
    batches = list(make_batches(items, size=40))
    assert len(batches) == 3
    assert len(batches[0]) == 40
    assert len(batches[1]) == 40
    assert len(batches[2]) == 15


# ---------------------------------------------------------------------------
# 3. Classifier validation tests (FR-12, FR-13, FR-16)
# ---------------------------------------------------------------------------

def test_validate_results_enforces_schema_and_rules():
    batch = [
        {"comment_id": 0, "likes": 50, "original_text": "App crashes"},
        {"comment_id": 1, "likes": 2, "original_text": "Hate you"},
        {"comment_id": 2, "likes": 0, "original_text": "Missing result"},
    ]
    raw_results = [
        {
            "comment_id": 0,
            "primary_sentiment": "negative",
            "sentiment_score": -0.85,
            "primary_category": "bug report",
            "is_bug_report": True,
            "theme": "software bugs",
            "language": "english",
            "priority": "low",
            "ai_summary": "App crash report",
            "suggested_reply": "Fix is on the way!",
        },
        {
            "comment_id": 1,
            "primary_sentiment": "negative",
            "sentiment_score": -0.99,
            "primary_category": "rude or toxic",
            "is_rude_or_toxic": True,
            "theme": "",
            "language": "english",
            "priority": "low",
            "ai_summary": "Toxic hate comment",
            "suggested_reply": "Should be wiped",
        },
    ]

    validated = validate_results(raw_results, batch)
    assert len(validated) == 3

    # Comment 0: sentiment capitalized, bug report sets priority High
    assert validated[0]["primary_sentiment"] == "Negative"
    assert validated[0]["primary_category"] == "Bug report"
    assert validated[0]["priority"] == "High"
    assert validated[0]["suggested_reply"] == "Fix is on the way!"

    # Comment 1: toxic comment forces suggested_reply to empty string
    assert validated[1]["is_rude_or_toxic"] is True
    assert validated[1]["suggested_reply"] == ""

    # Comment 2: missing result fills defaults safely
    assert validated[2]["primary_sentiment"] == "Neutral"
    assert validated[2]["primary_category"] == "Other"
    assert validated[2]["priority"] == "Low"


# ---------------------------------------------------------------------------
# 4. Analysis and stats tests (FR-18, FR-19, FR-24)
# ---------------------------------------------------------------------------

def test_aggregate_stats_and_filter_palette():
    comments = [
        {
            "comment_id": 0,
            "likes": 15,
            "primary_sentiment": "Positive",
            "sentiment_score": 0.8,
            "primary_category": "Praise",
            "is_praise": True,
            "is_question": False,
            "is_suggestion": False,
            "is_bug_report": False,
            "is_duplicate": False,
            "is_reply": False,
            "is_rude_or_toxic": False,
            "is_spam_or_bot": False,
            "published_at": "2026-11-01",
            "classification_status": "ok",
            "theme": "camera",
            "language": "English",
        },
        {
            "comment_id": 1,
            "likes": 5,
            "primary_sentiment": "Negative",
            "sentiment_score": -0.6,
            "primary_category": "Criticism",
            "is_praise": False,
            "is_question": True,
            "is_suggestion": False,
            "is_bug_report": False,
            "is_duplicate": True,
            "is_reply": False,
            "is_rude_or_toxic": False,
            "is_spam_or_bot": False,
            "published_at": "2026-11-01",
            "classification_status": "ok",
            "theme": "battery",
            "language": "Hindi",
        },
        {
            "comment_id": 2,
            "likes": 0,
            "primary_sentiment": "Negative",
            "sentiment_score": -0.9,
            "primary_category": "Rude or toxic",
            "is_praise": False,
            "is_question": False,
            "is_suggestion": False,
            "is_bug_report": False,
            "is_duplicate": False,
            "is_reply": False,
            "is_rude_or_toxic": True,
            "is_spam_or_bot": False,
            "published_at": "2026-11-02",
            "classification_status": "ok",
            "theme": "",
            "language": "English",
        },
    ]

    stats = aggregate_stats(comments)
    assert stats["total"] == 3
    assert stats["positive_count"] == 1
    assert stats["negative_count"] == 2
    assert stats["questions_count"] == 1
    assert stats["duplicates_count"] == 1
    assert stats["toxic_count"] == 1

    # Overlap test: Comment 1 is Negative and a Question (FR-13, FR-24)
    negatives = filter_palette(comments, "Negative", show_toxic=False)
    questions = filter_palette(comments, "Questions", show_toxic=False)
    assert any(c["comment_id"] == 1 for c in negatives)
    assert any(c["comment_id"] == 1 for c in questions)

    # Toxic toggle test (FR-26)
    all_clean = filter_palette(comments, "All", show_toxic=False)
    all_with_toxic = filter_palette(comments, "All", show_toxic=True)
    assert len(all_clean) == 2
    assert len(all_with_toxic) == 3

    # Top liked (FR-24)
    top_liked = filter_palette(comments, "Top liked", show_toxic=False)
    assert top_liked[0]["comment_id"] == 0  # 15 likes


def test_sentiment_timeline():
    comments = [
        {"published_at": "2026-11-01", "primary_sentiment": "Positive"},
        {"published_at": "2026-11-01", "primary_sentiment": "Negative"},
        {"published_at": "2026-11-02", "primary_sentiment": "Positive"},
    ]
    df = sentiment_timeline(comments)
    assert len(df) == 2
    assert "date" in df.columns
    assert "positive" in df.columns
    assert "negative" in df.columns


# ---------------------------------------------------------------------------
# 5. Creator tools tests (FR-30, FR-31, FR-32)
# ---------------------------------------------------------------------------

def test_extract_citations():
    text = "The viewers loved the camera [c12] and requested Pixel comparisons [c34, c56]."
    cites = _extract_citations(text)
    assert cites == [12, 34, 56]


def test_pick_top_quotes_excludes_toxic_and_limits_words():
    comments = [
        {
            "author": "Alice",
            "original_text": "This video was an absolute masterclass in cinematography and sound design.",
            "likes": 120,
            "is_praise": True,
            "is_rude_or_toxic": False,
            "theme": "camera",
        },
        {
            "author": "Troll",
            "original_text": "You are the worst reviewer ever.",
            "likes": 500,
            "is_praise": False,
            "is_rude_or_toxic": True,
            "theme": "pacing",
        },
    ]
    quotes = pick_top_quotes(comments)
    assert len(quotes) == 1
    assert quotes[0]["author"] == "Alice"


def test_build_creator_report():
    video_info = {"title": "Test Video", "channel": "Test Channel", "views": 1000, "comment_count": 100}
    stats = {"sentiment_score": 75.0, "sentiment_label": "Mostly positive", "total": 50}
    report = build_creator_report(video_info, stats, "Summary paragraph.", [], [], [], [], [])
    assert "# CommentLens Report: Test Video" in report
    assert "**75.0/100** — Mostly positive" in report
    assert "## Comment Breakdown" in report


# ---------------------------------------------------------------------------
# 6. Export tests (FR-36, FR-37, FR-38)
# ---------------------------------------------------------------------------

def test_export_csv_utf8_bom():
    comments = [
        {
            "author": "Subhashis",
            "published_at": "2026-11-01",
            "original_text": "Khub bhalo video da!",
            "likes": 10,
            "primary_sentiment": "Positive",
            "sentiment_score": 0.9,
            "primary_category": "Praise",
            "theme": "audio",
            "is_question": False,
            "is_suggestion": False,
            "is_bug_report": False,
            "is_duplicate": False,
            "duplicate_count": 1,
            "priority": "Low",
            "language": "Bengali",
            "ai_summary": "Praise in Bengali",
            "suggested_reply": "Dhonnobad!",
        }
    ]
    csv_bytes = export_csv(comments)
    # Check UTF-8 BOM prefix (\xef\xbb\xbf)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    decoded = csv_bytes.decode("utf-8-sig")
    assert "Subhashis" in decoded
    assert "Khub bhalo video da!" in decoded


def test_export_filename():
    fname = export_filename("dQw4w9WgXcQ", "Negative", date_str="2026-11-16")
    assert fname == "commentlens_dQw4w9WgXcQ_negative_2026-11-16.csv"
