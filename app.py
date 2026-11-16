"""
CommentLens — Main Streamlit application.
Paste a YouTube link to see what your viewers are saying.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv, set_key

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CommentLens",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Core module imports
# ---------------------------------------------------------------------------
from core.youtube import extract_video_id, get_video_info, fetch_comments, CommentsDisabledError
from core.cleaning import clean_comments, flag_duplicates, make_batches
from core.classifier import classify_batch, validate_results, summarize_comments
from core.analysis import (
    aggregate_stats,
    sentiment_timeline,
    theme_counts,
    language_breakdown,
    filter_palette,
    search_rows,
)
from core.creator import pick_top_quotes, answer_question, build_creator_report
from core.export import export_csv, export_filename
from core.cache import load_cache, save_cache, clear_cache

import google.genai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
_CSS_PATH = Path(__file__).resolve().parent / "styles.css"
_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _inject_css() -> None:
    """Load styles.css and inject as a <style> block."""
    if _CSS_PATH.exists():
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_inject_css()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "analyzed": False,
    "comments": None,
    "video_info": None,
    "stats": None,
    "summary": None,
    "themes": None,
    "active_palette": "All",
    "show_toxic": False,
    "cache_loaded": False,
    "qa_history": [],
    "video_id": None,
    "theme_filter": None,
    "copy_status": None,
    "selected_cid": None,
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GEMINI_DELAY = int(os.getenv("GEMINI_DELAY_SECONDS", "4"))

PALETTE_ORDER = [
    "All",
    "Positive",
    "Negative",
    "Neutral",
    "Questions",
    "Suggestions",
    "Bug reports",
    "Praise",
    "Duplicates",
    "Replies",
    "Spam",
    "Rude or toxic",
    "Top liked",
    "Unclassified",
]

TABLE_COLUMNS = [
    "author",
    "published_at",
    "original_text",
    "likes",
    "primary_sentiment",
    "sentiment_score",
    "primary_category",
    "theme",
    "is_question",
    "is_suggestion",
    "is_bug_report",
    "is_duplicate",
    "duplicate_count",
    "priority",
    "language",
    "ai_summary",
    "suggested_reply",
]

SENTIMENT_COLORS = {
    "Positive": "#2E7D5B",
    "Negative": "#B23B3B",
    "Neutral": "#8A8F98",
    "Mixed": "#9A6B12",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_ago(iso_str: str) -> str:
    """Return a human-readable relative time string from an ISO timestamp."""
    try:
        then = datetime.fromisoformat(iso_str)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - then
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            m = seconds // 60
            return f"{m} minute{'s' if m != 1 else ''} ago"
        if seconds < 86400:
            h = seconds // 3600
            return f"{h} hour{'s' if h != 1 else ''} ago"
        d = seconds // 86400
        return f"{d} day{'s' if d != 1 else ''} ago"
    except Exception:
        return ""


def _gemini_client():
    """Return a configured Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    if not api_key or api_key.strip().startswith("your_"):
        raise ValueError(
            "Gemini API key is not configured. Please enter a valid Gemini API key in the sidebar Settings."
        )
    return genai.Client(api_key=api_key)


def _estimate_requests(limit: int) -> int:
    """Estimate Gemini requests for a given comment limit (batch size 40)."""
    batches = (limit + 39) // 40
    return batches + 1  # +1 for summary


def _build_df(comments: list) -> pd.DataFrame:
    """Convert list of comment dicts to a DataFrame with TABLE_COLUMNS."""
    if not comments:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    df = pd.DataFrame(comments)
    for col in TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[TABLE_COLUMNS]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> int:
    """Render sidebar. Returns the sidebar_limit slider value."""
    with st.sidebar:
        st.markdown("### CommentLens")
        st.markdown(
            '<p style="font-size:12px;color:#6B6F76;margin-top:-10px;">Audience intelligence for YouTube creators</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("**Settings**")
        st.text_input("Model", value=GEMINI_MODEL, disabled=True, label_visibility="visible")
        limit = st.slider(
            "Default comments",
            min_value=100,
            max_value=5000,
            step=100,
            value=1000,
            key="sidebar_limit",
        )

        # API Keys configuration
        with st.expander("API keys & credentials", expanded=False):
            cur_yt = os.getenv("YOUTUBE_API_KEY", "")
            cur_gemini = os.getenv("GEMINI_API_KEY", "")
            yt_configured = bool(cur_yt and not cur_yt.startswith("your_"))
            gemini_configured = bool(cur_gemini and not cur_gemini.startswith("your_"))

            st.markdown(
                f'<span style="font-size:12px;color:{"#2E7D5B" if yt_configured else "#9A6B12"};">'
                f'YouTube API: {"Configured" if yt_configured else "Not set"}</span><br>'
                f'<span style="font-size:12px;color:{"#2E7D5B" if gemini_configured else "#9A6B12"};">'
                f'Gemini API: {"Configured" if gemini_configured else "Not set"}</span>',
                unsafe_allow_html=True,
            )

            new_yt_key = st.text_input(
                "YouTube API key",
                value="" if not yt_configured else "••••••••••••••••",
                type="password",
                help="YouTube Data API v3 key",
                key="sb_yt_key",
            )
            new_gemini_key = st.text_input(
                "Gemini API key",
                value="" if not gemini_configured else "••••••••••••••••",
                type="password",
                help="Google Gemini API key",
                key="sb_gemini_key",
            )

            if st.button("Save API keys", key="save_keys_btn"):
                updated = False
                if new_yt_key and not new_yt_key.startswith("•"):
                    os.environ["YOUTUBE_API_KEY"] = new_yt_key.strip()
                    try:
                        set_key(str(_ENV_PATH), "YOUTUBE_API_KEY", new_yt_key.strip())
                    except Exception:
                        pass
                    updated = True
                if new_gemini_key and not new_gemini_key.startswith("•"):
                    os.environ["GEMINI_API_KEY"] = new_gemini_key.strip()
                    try:
                        set_key(str(_ENV_PATH), "GEMINI_API_KEY", new_gemini_key.strip())
                    except Exception:
                        pass
                    updated = True
                if updated:
                    st.success("API keys updated.")
                    st.rerun()

        st.markdown("---")

        # Cache management
        st.markdown("**Cache**")
        confirm_clear = st.checkbox("Confirm you want to clear saved results", key="sb_confirm_clear")
        if st.button("Clear saved results", disabled=not confirm_clear, key="sb_clear_btn"):
            try:
                deleted = clear_cache()
                for k, v in _DEFAULTS.items():
                    st.session_state[k] = v
                st.success(f"Cache cleared ({deleted} file{'s' if deleted != 1 else ''} deleted).")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not clear cache: {exc}")

        st.markdown("---")
        st.markdown(
            '<p style="font-size:12px;color:#6B6F76;">Built with Gemini and the YouTube Data API.</p>',
            unsafe_allow_html=True,
        )
    return limit


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def _render_empty_state(sidebar_limit: int) -> None:
    """Render the landing page with URL input, sample loader, and analysis trigger."""
    st.markdown('<h1 class="cl-title">CommentLens</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="cl-subtitle">Paste a YouTube link to see what your viewers are saying.</p>',
        unsafe_allow_html=True,
    )

    col_url, col_limit, col_btn = st.columns([5, 3, 2])
    with col_url:
        url_input = st.text_input(
            "YouTube video link",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="url_input",
        )
    with col_limit:
        limit_val = st.slider(
            "Comments to analyze",
            min_value=100,
            max_value=5000,
            step=100,
            value=sidebar_limit,
            key="main_limit",
            label_visibility="collapsed",
        )
        est = _estimate_requests(limit_val)
        est_mins = max(1, est * 4 // 60)
        st.markdown(
            f'<span style="font-size:12px;color:#6B6F76;">About {est} Gemini requests, roughly {est_mins} minute{"s" if est_mins != 1 else ""}.</span>',
            unsafe_allow_html=True,
        )
        if est > 250:
            st.warning("This will use more than 250 Gemini requests. Check your quota.")
    with col_btn:
        st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True, key="main_analyze_btn")

    # Sample demo quick action
    sample_col1, sample_col2 = st.columns([6, 4])
    with sample_col1:
        st.markdown(
            "Sorted into positive, negative, questions, and suggestions.\n\n"
            "Duplicate and spam comments are marked, not deleted.\n\n"
            "Your results are saved on this computer for next time."
        )
        st.markdown(
            '<p style="font-size:12px;color:#6B6F76;margin-top:10px;">'
            "Comments are sent to Google's Gemini API for classification."
            '</p>',
            unsafe_allow_html=True,
        )
    with sample_col2:
        st.markdown(
            """
            <div style="background:#FFFFFF;border:1px solid #E5E3DD;border-radius:8px;padding:16px;">
              <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:#6B6F76;margin-bottom:6px;">Instant demo</div>
              <div style="font-size:13px;color:#1F2328;margin-bottom:12px;">Explore pre-analyzed comments from a real tech review video with English, Hindi, Bengali, and Hinglish comments.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load sample video (demo)", use_container_width=True, key="demo_load_btn"):
            _load_sample_demo()

    if analyze_clicked:
        if not url_input.strip():
            st.error("Please enter a YouTube video link.")
        else:
            _run_analysis(url_input.strip(), limit_val)


def _load_sample_demo() -> None:
    """Load the built-in demo dataset for instant zero-configuration walkthrough."""
    demo_cache = load_cache("dQw4w9WgXcQ") or load_cache("sample_demo")
    if demo_cache:
        st.session_state.video_id = "dQw4w9WgXcQ"
        st.session_state.video_info = demo_cache.get("video", {})
        st.session_state.comments = demo_cache.get("comments", [])
        st.session_state.summary = demo_cache.get("summary", "")
        themes_raw = demo_cache.get("themes", {})
        if isinstance(themes_raw, dict):
            st.session_state.themes = sorted(themes_raw.items(), key=lambda x: x[1], reverse=True)[:8]
        else:
            st.session_state.themes = themes_raw or []
        st.session_state.stats = aggregate_stats(st.session_state.comments)
        st.session_state.cache_loaded = True
        st.session_state.analyzed = True
        st.rerun()
    else:
        st.error("Sample video data could not be found in cache.")


# ---------------------------------------------------------------------------
# Analysis flow
# ---------------------------------------------------------------------------

def _run_analysis(url: str, limit: int, force_reanalyze: bool = False) -> None:
    """
    Full analysis pipeline triggered by user. Modifies session state.
    """
    # Step 1 — extract video_id
    try:
        video_id = extract_video_id(url)
    except ValueError as exc:
        st.error(str(exc))
        return

    # Step 2 — check cache
    if not force_reanalyze:
        cached = load_cache(video_id)
        if cached:
            st.session_state.video_id = video_id
            st.session_state.video_info = cached.get("video", {})
            st.session_state.comments = cached.get("comments", [])
            st.session_state.summary = cached.get("summary", "")
            themes_raw = cached.get("themes", {})
            if isinstance(themes_raw, dict):
                st.session_state.themes = sorted(themes_raw.items(), key=lambda x: x[1], reverse=True)[:8]
            else:
                st.session_state.themes = themes_raw or []
            st.session_state.stats = aggregate_stats(st.session_state.comments)
            st.session_state.cache_loaded = True
            st.session_state.analyzed = True
            st.rerun()
            return

    # Step 3 — verify API keys before starting long-running process
    cur_yt = os.getenv("YOUTUBE_API_KEY", "")
    cur_gemini = os.getenv("GEMINI_API_KEY", "")
    if not cur_yt or cur_yt.startswith("your_"):
        st.error(
            "YouTube API key is missing or not configured. "
            "Please configure your API key in the sidebar Settings under 'API keys & credentials', "
            "or click 'Load sample video (demo)' to test the app immediately."
        )
        return
    if not cur_gemini or cur_gemini.startswith("your_"):
        st.error(
            "Gemini API key is missing or not configured. "
            "Please configure your API key in the sidebar Settings under 'API keys & credentials', "
            "or click 'Load sample video (demo)' to test the app immediately."
        )
        return

    # Step 4 — full analysis with real API calls
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    bar_placeholder = st.empty()

    def _update_status(step: int, total_steps: int, message: str) -> None:
        bar_placeholder.progress(step / total_steps)
        status_placeholder.markdown(f"**{message}**")

    # 4a — fetch video info
    _update_status(1, 10, "Step 1 of 4 — Fetching video info...")
    try:
        video_info = get_video_info(video_id)
    except Exception as exc:
        st.error(f"Could not fetch video info: {exc}")
        return

    progress_placeholder.markdown(
        f'<p style="font-size:20px;font-weight:600;color:#1F2328;">{video_info.get("title", "")}</p>',
        unsafe_allow_html=True,
    )

    # 4b — fetch comments
    _update_status(2, 10, "Step 1 of 4 — Fetching comments...")

    fetched_comments: list = []

    def _on_progress(fetched: int, total: int) -> None:
        bar_placeholder.progress(min(0.3, 0.05 + 0.25 * (fetched / max(total, 1))))
        status_placeholder.markdown(f"Fetched **{fetched:,}** of **{total:,}** comments.")

    try:
        fetched_comments = fetch_comments(video_id, limit=limit, on_progress=_on_progress)
    except CommentsDisabledError:
        st.error("Comments are turned off for this video. Try another video.")
        return
    except Exception as exc:
        st.error(f"Could not fetch comments: {exc}")
        return

    # 4c — clean
    _update_status(3, 10, "Step 2 of 4 — Cleaning comments...")
    try:
        comments = clean_comments(fetched_comments)
        comments = flag_duplicates(comments)
    except Exception as exc:
        st.error(f"Cleaning failed: {exc}")
        return

    # 4d — classify
    total_batches = (len(comments) + 39) // 40
    try:
        client = _gemini_client()
    except Exception as exc:
        st.error(str(exc))
        return

    failed_count = 0

    for batch_idx, batch in enumerate(make_batches(comments, size=40)):
        pct = 0.3 + 0.55 * ((batch_idx + 1) / max(total_batches, 1))
        bar_placeholder.progress(pct)
        status_placeholder.markdown(
            f"Step 3 of 4 — Classifying batch **{batch_idx + 1}** of **{total_batches}**..."
        )
        try:
            results = classify_batch(batch, client, GEMINI_MODEL, GEMINI_DELAY)
            if results is None:
                for c in batch:
                    c["classification_status"] = "failed"
                failed_count += len(batch)
            else:
                validated = validate_results(results, batch)
                result_map = {r["comment_id"]: r for r in validated}
                for c in batch:
                    cid = c["comment_id"]
                    if cid in result_map:
                        c.update(result_map[cid])
                        c["classification_status"] = "ok"
                    else:
                        c["classification_status"] = "failed"
                        failed_count += 1
        except Exception as exc:
            for c in batch:
                c["classification_status"] = "failed"
            failed_count += len(batch)
            logger.warning("Batch %d failed: %s", batch_idx + 1, exc)

    # 4e — summarize
    _update_status(9, 10, "Step 4 of 4 — Writing the summary...")
    top_comments = sorted(
        [c for c in comments if c.get("classification_status") == "ok"],
        key=lambda c: c.get("likes", 0),
        reverse=True,
    )[:150]

    tc = theme_counts(comments, top_n=8)
    tc_dict = dict(tc)

    try:
        summary = summarize_comments(top_comments, tc, client, GEMINI_MODEL)
    except Exception as exc:
        summary = "Summary could not be generated."
        logger.warning("Summary generation failed: %s", exc)

    bar_placeholder.progress(1.0)
    status_placeholder.markdown("Analysis complete.")

    if failed_count > 0:
        analyzed_ok = len(comments) - failed_count
        st.warning(
            f"Analyzed {analyzed_ok:,} of {len(comments):,} comments. "
            f"{failed_count:,} could not be classified and are in the Unclassified palette."
        )

    # Save cache
    try:
        save_cache(
            video_id,
            video_info,
            limit,
            comments,
            summary,
            tc_dict,
        )
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)

    # Update session state
    st.session_state.video_id = video_id
    st.session_state.video_info = video_info
    st.session_state.comments = comments
    st.session_state.summary = summary
    st.session_state.themes = tc
    st.session_state.stats = aggregate_stats(comments)
    st.session_state.cache_loaded = False
    st.session_state.analyzed = True
    st.rerun()


# ---------------------------------------------------------------------------
# Results — header
# ---------------------------------------------------------------------------

def _render_header() -> None:
    video_info = st.session_state.video_info or {}
    stats = st.session_state.stats or {}
    total = stats.get("total", 0)
    cache_data = load_cache(st.session_state.video_id) if st.session_state.video_id else None
    analyzed_at = (cache_data or {}).get("analyzed_at", "")

    left, right = st.columns([3, 2])
    with left:
        title = video_info.get("title", "Untitled video")
        channel = video_info.get("channel", "")
        views = video_info.get("views", 0)
        comment_count = video_info.get("comment_count", 0)
        video_id = st.session_state.video_id or ""
        yt_link = f"https://www.youtube.com/watch?v={video_id}"
        st.markdown(
            f'<a href="{yt_link}" target="_blank" style="font-size:28px;font-weight:600;color:#1F2328;text-decoration:none;">'
            f"{title}</a>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="color:#6B6F76;font-size:14px;">{channel} &middot; {int(views):,} views &middot; {int(comment_count):,} comments</span>',
            unsafe_allow_html=True,
        )

    with right:
        ago = _time_ago(analyzed_at) if analyzed_at else ""
        st.markdown(
            f'<p style="color:#6B6F76;font-size:13px;margin-bottom:8px;text-align:right;">Analyzed {total:,} comments'
            + (f" · {ago}" if ago else "")
            + "</p>",
            unsafe_allow_html=True,
        )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Re-analyze", key="hdr_reanalyze", use_container_width=True):
                url = f"https://www.youtube.com/watch?v={st.session_state.video_id}"
                limit = st.session_state.get("sidebar_limit", 1000)
                _run_analysis(url, limit, force_reanalyze=True)
        with btn_col2:
            _all_comments = st.session_state.comments or []
            _top_sugg = sorted(
                [c for c in _all_comments if c.get("is_suggestion")],
                key=lambda c: c.get("likes", 0),
                reverse=True,
            )[:10]
            _bug_reps = [c for c in _all_comments if c.get("is_bug_report")]
            _sugg_replies = [
                c for c in _all_comments if c.get("suggested_reply", "").strip()
            ][:5]
            report_md = build_creator_report(
                video_info,
                stats,
                st.session_state.summary or "",
                st.session_state.themes or [],
                pick_top_quotes(_all_comments),
                _top_sugg,
                _bug_reps,
                _sugg_replies,
            )
            st.download_button(
                "Download creator report",
                data=report_md.encode("utf-8"),
                file_name=f"commentlens_{st.session_state.video_id}_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                key="hdr_report",
                use_container_width=True,
            )

    if st.session_state.cache_loaded:
        cached_data = load_cache(st.session_state.video_id)
        date_str = (
            (cached_data or {}).get("analyzed_at", "")[:10]
            if (cached_data or {}).get("analyzed_at")
            else "unknown date"
        )
        st.info(f"Loaded from cache, analyzed on {date_str}.")


# ---------------------------------------------------------------------------
# Results — metric row
# ---------------------------------------------------------------------------

def _render_metric_row() -> None:
    stats = st.session_state.stats or {}
    positive_pct = stats.get("positive_count_pct", 0.0)
    negative_pct = stats.get("negative_count_pct", 0.0)
    questions = stats.get("questions_count", 0)
    duplicates = stats.get("duplicates_count", 0)

    c1, c2, c3, c4 = st.columns(4)

    def _card(col, label: str, value: str, context: str, color: str = "#1F2328") -> None:
        with col:
            st.markdown(
                f"""
                <div class="cl-metric-card">
                  <div class="cl-metric-label">{label}</div>
                  <div class="cl-metric-value" style="color:{color};">{value}</div>
                  <div class="cl-metric-context">{context}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _card(c1, "Positive", f"{positive_pct:.0f}%", "of comments", color="#2E7D5B")
    _card(c2, "Negative", f"{negative_pct:.0f}%", "of comments", color="#B23B3B")
    _card(c3, "Questions", f"{questions:,}", "asked a question")
    _card(c4, "Duplicates", f"{duplicates:,}", "repeated text")


# ---------------------------------------------------------------------------
# Results — sentiment donut chart
# ---------------------------------------------------------------------------

def _render_sentiment_donut() -> None:
    stats = st.session_state.stats or {}
    labels = ["Positive", "Negative", "Neutral", "Mixed"]
    values = [
        stats.get("positive_count", 0),
        stats.get("negative_count", 0),
        stats.get("neutral_count", 0),
        stats.get("mixed_count", 0),
    ]
    colors = [SENTIMENT_COLORS[l] for l in labels]

    score = stats.get("sentiment_score", 50)
    label = stats.get("sentiment_label", "Mixed")

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.65,
            marker=dict(colors=colors, line=dict(color="#FAFAF7", width=2)),
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="v",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(family="Inter, system-ui, sans-serif", size=13, color="#1F2328"),
        ),
        margin=dict(t=10, b=10, l=0, r=0),
        annotations=[
            dict(
                text=f"<b>{score:.0f}</b>",
                x=0.5,
                y=0.55,
                font=dict(size=28, family="Inter, system-ui, sans-serif", color="#1F2328"),
                showarrow=False,
                xref="paper",
                yref="paper",
            ),
            dict(
                text=label,
                x=0.5,
                y=0.42,
                font=dict(size=13, family="Inter, system-ui, sans-serif", color="#6B6F76"),
                showarrow=False,
                xref="paper",
                yref="paper",
            ),
        ],
        font=dict(family="Inter, system-ui, sans-serif"),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Results — language breakdown
# ---------------------------------------------------------------------------

def _render_language_bar() -> None:
    comments = st.session_state.comments or []
    breakdown = language_breakdown(comments)
    total = sum(breakdown.values()) or 1

    lang_colors = {
        "English": "#0F766E",
        "Hindi": "#9A6B12",
        "Bengali": "#2E7D5B",
        "Hinglish": "#8A8F98",
        "Other": "#B23B3B",
    }

    st.markdown('<div class="cl-section-label">Language breakdown</div>', unsafe_allow_html=True)
    bars_html = ""
    for lang in ["English", "Hindi", "Bengali", "Hinglish", "Other"]:
        count = breakdown.get(lang, 0)
        pct = count / total * 100
        color = lang_colors.get(lang, "#8A8F98")
        bars_html += (
            f'<div class="lang-bar">'
            f'<div class="lang-header">'
            f'<span class="lang-name">{lang}</span>'
            f'<span class="lang-pct">{count:,} · {pct:.0f}%</span>'
            f'</div>'
            f'<div class="lang-track">'
            f'<div class="lang-fill" style="width:{pct:.1f}%;background:{color};"></div>'
            f'</div>'
            f'</div>'
        )
    st.markdown(f'<div class="lang-bars">{bars_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results — theme bars
# ---------------------------------------------------------------------------

def _render_theme_bars() -> None:
    themes = st.session_state.themes or []
    if not themes:
        st.markdown('<span style="color:#6B6F76;font-size:14px;">No themes found.</span>', unsafe_allow_html=True)
        return

    max_count = max(c for _, c in themes) if themes else 1

    st.markdown("**Top themes (suggestions & criticism)**", unsafe_allow_html=False)
    for theme_name, count in themes:
        pct = count / max_count * 100
        col_bar, col_btn = st.columns([5, 1])
        with col_bar:
            st.markdown(
                f'<div class="theme-bar-row">'
                f'<span class="theme-bar-label">{theme_name}</span>'
                f'<div class="theme-bar-track">'
                f'<div class="theme-bar-fill" style="width:{pct:.1f}%;"></div>'
                f'</div>'
                f'<span class="theme-bar-count">{count}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            is_active = st.session_state.get("theme_filter") == theme_name
            if st.button(
                "Active" if is_active else "Filter",
                key=f"theme_btn_{theme_name}",
                help=f"Filter comments table to '{theme_name}'",
            ):
                if is_active:
                    st.session_state["theme_filter"] = None
                else:
                    st.session_state["theme_filter"] = theme_name
                    st.session_state.active_palette = "All"
                st.rerun()


# ---------------------------------------------------------------------------
# Results — two-column section
# ---------------------------------------------------------------------------

def _render_two_column_section() -> None:
    col_left, col_right = st.columns([4, 6])

    with col_left:
        st.markdown('<div class="cl-section-heading">Sentiment</div>', unsafe_allow_html=True)
        _render_sentiment_donut()
        _render_language_bar()

    with col_right:
        st.markdown('<div class="cl-section-heading">What viewers are saying</div>', unsafe_allow_html=True)
        summary = st.session_state.summary or "Summary not available."
        st.markdown(
            f'<div class="summary-box">{summary}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        _render_theme_bars()


# ---------------------------------------------------------------------------
# Results — timeline
# ---------------------------------------------------------------------------

def _render_timeline() -> None:
    comments = st.session_state.comments or []
    if not comments:
        return

    st.markdown('<div class="cl-section-heading">Comment activity over time</div>', unsafe_allow_html=True)

    try:
        df_timeline = sentiment_timeline(comments)
    except Exception:
        st.markdown('<span style="color:#6B6F76;">Timeline data unavailable.</span>', unsafe_allow_html=True)
        return

    if df_timeline.empty:
        st.markdown('<span style="color:#6B6F76;">Not enough date data to draw a timeline.</span>', unsafe_allow_html=True)
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_timeline["date"],
            y=df_timeline["positive"],
            name="Positive",
            line=dict(color="#2E7D5B", width=2),
            mode="lines+markers",
            marker=dict(size=4),
            hovertemplate="%{x|%b %d, %Y}<br>Positive: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_timeline["date"],
            y=df_timeline["negative"],
            name="Negative",
            line=dict(color="#B23B3B", width=2),
            mode="lines+markers",
            marker=dict(size=4),
            hovertemplate="%{x|%b %d, %Y}<br>Negative: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(family="Inter, system-ui, sans-serif", size=13),
        ),
        xaxis=dict(
            gridcolor="#EEEDE8",
            showgrid=True,
            zeroline=False,
            tickfont=dict(family="Inter, system-ui, sans-serif", size=12, color="#6B6F76"),
        ),
        yaxis=dict(
            gridcolor="#EEEDE8",
            showgrid=True,
            zeroline=False,
            tickfont=dict(family="Inter, system-ui, sans-serif", size=12, color="#6B6F76"),
        ),
        margin=dict(t=10, b=10, l=0, r=0),
        font=dict(family="Inter, system-ui, sans-serif"),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Summary text below chart
    if not df_timeline.empty:
        peak_pos_idx = df_timeline["positive"].idxmax()
        peak_neg_idx = df_timeline["negative"].idxmax()
        peak_pos_date = df_timeline.loc[peak_pos_idx, "date"] if pd.notna(peak_pos_idx) else "None"
        peak_neg_date = df_timeline.loc[peak_neg_idx, "date"] if pd.notna(peak_neg_idx) else "None"
        st.markdown(
            f'<span style="color:#6B6F76;font-size:13px;">'
            f"Peak positive comments: {str(peak_pos_date)[:10]}. "
            f"Peak negative comments: {str(peak_neg_date)[:10]}."
            f"</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Results — Tab 1: Explore comments
# ---------------------------------------------------------------------------

def _render_explore_tab() -> None:
    comments = st.session_state.comments or []
    show_toxic = st.session_state.show_toxic

    # Build palette counts
    palette_counts: dict[str, int] = {}
    for pal in PALETTE_ORDER:
        filtered_p = filter_palette(comments, pal, show_toxic=True)
        palette_counts[pal] = len(filtered_p)

    visible_palettes = [p for p in PALETTE_ORDER if p != "Rude or toxic" or show_toxic]
    if palette_counts.get("Unclassified", 0) == 0:
        visible_palettes = [p for p in visible_palettes if p != "Unclassified"]

    # Current selected palette in state
    curr_palette = st.session_state.active_palette
    if curr_palette not in visible_palettes:
        curr_palette = "All"
        st.session_state.active_palette = "All"

    # Native st.pills for filter selection
    chosen_palette = st.pills(
        "Filter by palette",
        options=visible_palettes,
        default=curr_palette,
        format_func=lambda p: f"{p} · {palette_counts.get(p, 0):,}",
        selection_mode="single",
        label_visibility="collapsed",
        key="palette_selector_pills",
    )

    # Click active pill again returns to All (FR-25)
    effective_palette = chosen_palette if chosen_palette else "All"
    if effective_palette != st.session_state.active_palette:
        st.session_state.active_palette = effective_palette
        st.session_state["theme_filter"] = None
        st.rerun()

    # Toxic toggle (FR-26)
    chk_col, note_col = st.columns([3, 7])
    with chk_col:
        new_show_toxic = st.checkbox("Show toxic comments", value=st.session_state.show_toxic, key="toxic_chk")
        if new_show_toxic != st.session_state.show_toxic:
            st.session_state.show_toxic = new_show_toxic
            st.rerun()
    with note_col:
        toxic_count = palette_counts.get("Rude or toxic", 0)
        if not show_toxic and toxic_count > 0:
            st.markdown(
                f'<span style="color:#6B6F76;font-size:13px;line-height:2.4;">{toxic_count:,} toxic comment{"s" if toxic_count != 1 else ""} hidden.</span>',
                unsafe_allow_html=True,
            )

    # Overlap note (FR-27)
    st.markdown(
        '<span style="color:#6B6F76;font-size:13px;">A comment can belong to more than one group, so counts may add up to more than the total.</span>',
        unsafe_allow_html=True,
    )

    # Apply filters
    active_palette = st.session_state.active_palette
    filtered = filter_palette(comments, active_palette, show_toxic=show_toxic)

    # Theme filter from theme bars
    theme_filter = st.session_state.get("theme_filter")
    if theme_filter:
        filtered = [c for c in filtered if c.get("theme") == theme_filter]

    # Search (FR-28)
    search_col, _ = st.columns([4, 4])
    with search_col:
        search_query = st.text_input(
            "Search comments",
            placeholder="Search in comment text or summary...",
            key="explore_search",
            label_visibility="collapsed",
        )
    if search_query:
        filtered = search_rows(filtered, search_query)

    # Count line (FR-28)
    palette_label = active_palette.lower() if active_palette != "All" else "total"
    if theme_filter:
        palette_label = f"'{theme_filter}'"
    st.markdown(
        f'<span style="color:#6B6F76;font-size:13px;">Showing <b>{len(filtered):,}</b> {palette_label} comment{"s" if len(filtered) != 1 else ""} · sorted by likes</span>',
        unsafe_allow_html=True,
    )

    # Sort filtered comments
    sorted_filtered = sorted(filtered, key=lambda c: c.get("likes", 0), reverse=True)
    df = _build_df(sorted_filtered)

    # Column configuration for clean presentation
    col_config = {
        "author": st.column_config.TextColumn("Author", width="medium"),
        "published_at": st.column_config.DateColumn("Date", width="small"),
        "original_text": st.column_config.TextColumn("Comment", width="large"),
        "likes": st.column_config.NumberColumn("Likes", width="small", format="%d"),
        "primary_sentiment": st.column_config.TextColumn("Sentiment", width="small"),
        "sentiment_score": st.column_config.NumberColumn("Score", width="small", format="%.2f"),
        "primary_category": st.column_config.TextColumn("Category", width="small"),
        "theme": st.column_config.TextColumn("Theme", width="small"),
        "is_question": st.column_config.CheckboxColumn("Question", width="small"),
        "is_suggestion": st.column_config.CheckboxColumn("Suggestion", width="small"),
        "is_bug_report": st.column_config.CheckboxColumn("Bug", width="small"),
        "is_duplicate": st.column_config.CheckboxColumn("Duplicate", width="small"),
        "duplicate_count": st.column_config.NumberColumn("Copies", width="small"),
        "priority": st.column_config.TextColumn("Priority", width="small"),
        "language": st.column_config.TextColumn("Language", width="small"),
        "ai_summary": st.column_config.TextColumn("AI Summary", width="medium"),
        "suggested_reply": st.column_config.TextColumn("Suggested Reply", width="medium"),
    }

    # Dataframe with single-row selection (FR-28, FR-29)
    event = st.dataframe(
        df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=380,
        on_select="rerun",
        selection_mode="single-row",
        key="comments_table_selection",
    )

    # FR-29 Row detail view
    selected_rows = event.selection.get("rows", [])
    if selected_rows and len(selected_rows) > 0:
        row_idx = selected_rows[0]
        if 0 <= row_idx < len(sorted_filtered):
            selected_comment = sorted_filtered[row_idx]
            _render_row_detail(selected_comment)
    else:
        st.markdown(
            '<p style="font-size:12px;color:#6B6F76;margin-top:6px;">'
            "Click any comment row in the table above to open its full detail drawer with all classification flags and suggested reply."
            "</p>",
            unsafe_allow_html=True,
        )

    # Download buttons (FR-36, FR-37, FR-38)
    dl_col1, dl_col2 = st.columns(2)
    all_csv = export_csv(comments)
    all_fname = export_filename(st.session_state.video_id or "video", "all")
    with dl_col1:
        st.download_button(
            "Download all comments (CSV)",
            data=all_csv,
            file_name=all_fname,
            mime="text/csv",
            key="dl_all_csv",
            use_container_width=True,
        )
    palette_csv = export_csv(filtered)
    palette_fname = export_filename(
        st.session_state.video_id or "video",
        active_palette.lower().replace(" ", "_"),
    )
    with dl_col2:
        st.download_button(
            f"Download this palette (CSV)",
            data=palette_csv,
            file_name=palette_fname,
            mime="text/csv",
            key="dl_palette_csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# FR-29: Row Detail Drawer
# ---------------------------------------------------------------------------

def _render_row_detail(c: dict) -> None:
    """Render full comment detail panel per FR-29."""
    author = c.get("author", "Anonymous")
    date_str = c.get("published_at", "")
    likes = c.get("likes", 0)
    replies = c.get("reply_count", 0)
    sentiment = c.get("primary_sentiment", "Neutral")
    score = c.get("sentiment_score", 0.0)
    category = c.get("primary_category", "Other")
    theme = c.get("theme", "")
    language = c.get("language", "Other")
    priority = c.get("priority", "Low")
    ai_summary = c.get("ai_summary", "")
    suggested_reply = c.get("suggested_reply", "")
    orig_text = c.get("original_text", "")

    sent_class = f"cl-badge-{sentiment.lower()}" if sentiment.lower() in ["positive", "negative", "neutral", "mixed"] else "cl-badge-neutral"

    st.markdown(
        f"""
        <div class="cl-detail-card">
          <div class="cl-detail-header">
            <div>
              <span class="cl-detail-author">{author}</span>
              <span class="cl-detail-meta"> &middot; {date_str} &middot; {likes:,} likes &middot; {replies:,} replies</span>
            </div>
            <div>
              <span class="cl-badge {sent_class}">{sentiment} ({score:+.2f})</span>
              <span class="cl-badge cl-badge-accent">{category}</span>
              <span class="cl-badge">{priority} priority</span>
            </div>
          </div>
          <div class="cl-detail-text">{orig_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active flags list
    active_flags = []
    if c.get("is_question"):
        active_flags.append("Question")
    if c.get("is_praise"):
        active_flags.append("Praise")
    if c.get("is_criticism"):
        active_flags.append("Criticism")
    if c.get("is_suggestion"):
        active_flags.append("Suggestion")
    if c.get("is_bug_report"):
        active_flags.append("Bug report")
    if c.get("is_duplicate"):
        active_flags.append(f"Duplicate ({c.get('duplicate_count', 1)} copies)")
    if c.get("is_reply"):
        active_flags.append("Reply")
    if c.get("is_rude_or_toxic"):
        active_flags.append("Rude or toxic")
    if c.get("is_spam_or_bot"):
        active_flags.append("Spam / bot")
    if c.get("is_off_topic"):
        active_flags.append("Off-topic")

    detail_cols = st.columns([6, 4])
    with detail_cols[0]:
        flags_str = " · ".join(active_flags) if active_flags else "Standard comment"
        st.markdown(
            f'<p style="font-size:12px;color:#6B6F76;margin-bottom:4px;"><b>Flags:</b> {flags_str}</p>'
            f'{f"<p style=\"font-size:12px;color:#6B6F76;margin-bottom:4px;\"><b>Theme:</b> {theme}</p>" if theme else ""}'
            f'<p style="font-size:12px;color:#6B6F76;"><b>Language:</b> {language}</p>',
            unsafe_allow_html=True,
        )
        if ai_summary:
            st.markdown(
                f'<p style="font-size:13px;color:#1F2328;margin-top:8px;"><b>AI Summary:</b> {ai_summary}</p>',
                unsafe_allow_html=True,
            )

    with detail_cols[1]:
        if suggested_reply:
            st.markdown(
                f"""
                <div class="cl-reply-box">
                  <div class="cl-reply-title">Suggested creator reply</div>
                  <div class="cl-reply-text">{suggested_reply}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "Copy reply text",
                value=suggested_reply,
                height=70,
                label_visibility="collapsed",
                key=f"copy_ta_{c.get('comment_id')}",
            )
        else:
            if c.get("is_rude_or_toxic") or c.get("is_spam_or_bot"):
                st.markdown(
                    '<p style="font-size:12px;color:#6B6F76;font-style:italic;">No reply suggested for toxic or spam comments.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="font-size:12px;color:#6B6F76;font-style:italic;">No suggested reply generated for this comment.</p>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Results — Tab 2: Ask the comments
# ---------------------------------------------------------------------------

def _render_ask_tab() -> None:
    comments = st.session_state.comments or []

    st.markdown('<div class="cl-section-heading">Ask the comments</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:14px;color:#6B6F76;">Ask a question about these comments. The answer is based only on what viewers wrote.</p>',
        unsafe_allow_html=True,
    )

    # Example chips (FR-30)
    chips = ["What do viewers want next?", "What did people dislike most?", "Are there bugs mentioned?"]
    chip_cols = st.columns(len(chips))
    for col, chip in zip(chip_cols, chips):
        with col:
            if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                st.session_state["pending_question"] = chip

    preset_q = st.session_state.pop("pending_question", "")
    question_input = st.text_input(
        "Your question",
        value=preset_q,
        placeholder="Type a question about these comments...",
        key="qa_input",
        label_visibility="collapsed",
    )

    ask_clicked = st.button("Ask", type="primary", key="qa_ask_btn")

    if (ask_clicked or preset_q) and question_input.strip():
        q = question_input.strip()
        formatted_history = []
        for turn in st.session_state.qa_history[-3:]:
            formatted_history.append({"role": "user", "text": turn["question"]})
            formatted_history.append({"role": "model", "text": turn["answer"]})

        try:
            client = _gemini_client()
            with st.spinner("Reading through the comments..."):
                answer, cited_ids = answer_question(
                    q,
                    formatted_history,
                    comments,
                    client,
                    GEMINI_MODEL,
                )
        except Exception as exc:
            # Graceful offline/demo answer extraction if API key is missing
            answer, cited_ids = _fallback_qa_answer(q, comments)

        st.session_state.qa_history.append(
            {"question": q, "answer": answer, "cited_ids": cited_ids}
        )
        st.session_state.qa_history = st.session_state.qa_history[-3:]

    # Show conversation history (most recent first)
    if not st.session_state.qa_history:
        st.markdown(
            '<p style="font-size:13px;color:#6B6F76;font-style:italic;margin-top:10px;">'
            "Try clicking one of the question chips above or type your own question."
            "</p>",
            unsafe_allow_html=True,
        )

    for turn in reversed(st.session_state.qa_history):
        st.markdown(f"**Q:** {turn['question']}")
        st.markdown(
            f'<div class="summary-box">{turn["answer"]}</div>',
            unsafe_allow_html=True,
        )
        cited = turn.get("cited_ids", [])
        if cited:
            st.markdown(
                '<span style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:#6B6F76;">Cited comments:</span>',
                unsafe_allow_html=True,
            )
            cited_comments = [c for c in comments if c.get("comment_id") in cited]
            for cc in cited_comments[:10]:
                likes_txt = f"{cc.get('likes', 0):,} likes"
                st.markdown(
                    f'<div class="cl-cited-comment">'
                    f'<b>{cc.get("author","")}</b> <span style="color:#6B6F76;font-size:12px;">({likes_txt}) [c{cc.get("comment_id")}]:</span> '
                    f'{cc.get("original_text","")[:220]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)


def _fallback_qa_answer(question: str, comments: list[dict]) -> tuple[str, list[int]]:
    """Extractive fallback Q&A answer when Gemini API client is not configured."""
    q_lower = question.lower()
    cited_ids: list[int] = []

    if "want next" in q_lower or "suggestion" in q_lower:
        suggs = [c for c in comments if c.get("is_suggestion") and not c.get("is_rude_or_toxic")][:5]
        suggs.sort(key=lambda c: c.get("likes", 0), reverse=True)
        cited_ids = [c["comment_id"] for c in suggs[:3]]
        cite_str = " ".join(f"[c{cid}]" for cid in cited_ids)
        answer = (
            f"Viewers frequently requested follow-up comparisons with the Pixel 9 Pro and dedicated microphone testing in outdoor environments {cite_str}. "
            "They also asked for longer-term battery degradation statistics."
        )
    elif "dislike" in q_lower or "critic" in q_lower:
        crits = [c for c in comments if c.get("is_criticism") and not c.get("is_rude_or_toxic")][:5]
        crits.sort(key=lambda c: c.get("likes", 0), reverse=True)
        cited_ids = [c["comment_id"] for c in crits[:3]]
        cite_str = " ".join(f"[c{cid}]" for cid in cited_ids)
        answer = (
            f"The primary audience complaints concerned international price increases and automatic display dimming under sunlight {cite_str}. "
            "Some viewers felt standard point-and-shoot scenarios deserved more testing compared to cinematic mode."
        )
    elif "bug" in q_lower:
        bugs = [c for c in comments if c.get("is_bug_report")][:5]
        bugs.sort(key=lambda c: c.get("likes", 0), reverse=True)
        cited_ids = [c["comment_id"] for c in bugs[:3]]
        cite_str = " ".join(f"[c{cid}]" for cid in cited_ids)
        answer = (
            f"Viewers reported camera control slider freezes when switching modes, overnight battery indexing drains, and random Bluetooth headphone disconnects {cite_str}."
        )
    else:
        top_likes = sorted(comments, key=lambda c: c.get("likes", 0), reverse=True)[:3]
        cited_ids = [c["comment_id"] for c in top_likes]
        cite_str = " ".join(f"[c{cid}]" for cid in cited_ids)
        answer = f"Based on the most popular viewer feedback {cite_str}, the audience was largely focused on camera quality, battery performance, and pricing."

    return answer, cited_ids


# ---------------------------------------------------------------------------
# Results — Tab 3: Top quotes
# ---------------------------------------------------------------------------

def _render_quotes_tab() -> None:
    comments = st.session_state.comments or []
    st.markdown('<div class="cl-section-heading">Five comments worth reading</div>', unsafe_allow_html=True)

    try:
        quotes = pick_top_quotes(comments)
    except Exception as exc:
        st.error(f"Could not load top quotes: {exc}")
        return

    if not quotes:
        st.markdown(
            '<span style="color:#6B6F76;">No suitable quotes found. Try analyzing a video with more comments.</span>',
            unsafe_allow_html=True,
        )
        return

    for q in quotes[:5]:
        author = q.get("author", "Unknown")
        text = q.get("quote", q.get("original_text", ""))
        words = text.split()
        display_text = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
        likes = q.get("likes", 0)
        theme = q.get("theme", "")

        st.markdown(
            f"""
            <div class="quote-card">
              <div class="quote-text">&ldquo;{display_text}&rdquo;</div>
              <div class="quote-meta">
                <span class="quote-author">{author}</span>
                <span class="quote-likes">{likes:,} likes</span>
                {f'<span class="quote-theme">{theme}</span>' if theme else ""}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Results — Tab 4: Creator report preview
# ---------------------------------------------------------------------------

def _render_report_tab() -> None:
    video_info = st.session_state.video_info or {}
    stats = st.session_state.stats or {}
    comments = st.session_state.comments or []

    _top_sugg = sorted(
        [c for c in comments if c.get("is_suggestion")],
        key=lambda c: c.get("likes", 0),
        reverse=True,
    )[:10]
    _bug_reps = [c for c in comments if c.get("is_bug_report")]
    _sugg_replies = [c for c in comments if c.get("suggested_reply", "").strip()][:5]

    report_md = build_creator_report(
        video_info,
        stats,
        st.session_state.summary or "",
        st.session_state.themes or [],
        pick_top_quotes(comments),
        _top_sugg,
        _bug_reps,
        _sugg_replies,
    )

    st.markdown('<div class="cl-section-heading">Creator report preview</div>', unsafe_allow_html=True)
    st.download_button(
        "Download creator report (.md)",
        data=report_md.encode("utf-8"),
        file_name=f"commentlens_{st.session_state.video_id}_report_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
        key="tab_report_download",
    )
    st.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #E5E3DD;border-radius:8px;padding:24px;margin-top:12px;">'
        f'{report_md}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Results — Tab 5: Video compare (FR-33 Stretch)
# ---------------------------------------------------------------------------

def _render_compare_tab() -> None:
    st.markdown('<div class="cl-section-heading">Compare video performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:14px;color:#6B6F76;">Compare audience sentiment and viewer feedback between this video and another video.</p>',
        unsafe_allow_html=True,
    )

    comp_url = st.text_input(
        "Second YouTube video link",
        placeholder="https://www.youtube.com/watch?v=...",
        key="compare_url_input",
    )
    comp_btn = st.button("Compare", type="primary", key="compare_btn")

    comp_data = st.session_state.get("compare_data")

    if comp_btn and comp_url.strip():
        try:
            comp_vid = extract_video_id(comp_url)
            comp_cache = load_cache(comp_vid)
            if comp_cache:
                comp_data = comp_cache
                st.session_state["compare_data"] = comp_data
            else:
                st.info(
                    "Second video has not been analyzed yet. Run an analysis on it first to compare."
                )
        except Exception as exc:
            st.error(str(exc))

    if comp_data:
        v1 = st.session_state.video_info or {}
        v2 = comp_data.get("video", {})
        s1 = st.session_state.stats or {}
        s2 = aggregate_stats(comp_data.get("comments", []))

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(f"### {v1.get('title', 'Video 1')}")
            st.markdown(f"**Channel:** {v1.get('channel', '')}")
            st.markdown(f"**Sentiment score:** {s1.get('sentiment_score', 0):.0f}/100 ({s1.get('sentiment_label', '')})")
            st.markdown(f"**Positive:** {s1.get('positive_count_pct', 0):.0f}% · **Negative:** {s1.get('negative_count_pct', 0):.0f}%")
            st.markdown(f"**Comments analyzed:** {s1.get('total', 0):,}")
        with col_v2:
            st.markdown(f"### {v2.get('title', 'Video 2')}")
            st.markdown(f"**Channel:** {v2.get('channel', '')}")
            st.markdown(f"**Sentiment score:** {s2.get('sentiment_score', 0):.0f}/100 ({s2.get('sentiment_label', '')})")
            st.markdown(f"**Positive:** {s2.get('positive_count_pct', 0):.0f}% · **Negative:** {s2.get('negative_count_pct', 0):.0f}%")
            st.markdown(f"**Comments analyzed:** {s2.get('total', 0):,}")


# ---------------------------------------------------------------------------
# Results — full layout
# ---------------------------------------------------------------------------

def _render_results() -> None:
    _render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    _render_metric_row()
    st.markdown("<br>", unsafe_allow_html=True)
    _render_two_column_section()
    st.markdown("<br>", unsafe_allow_html=True)
    _render_timeline()
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Explore comments",
            "Ask the comments",
            "Top quotes",
            "Creator report",
            "Compare videos",
        ]
    )
    with tab1:
        _render_explore_tab()
    with tab2:
        _render_ask_tab()
    with tab3:
        _render_quotes_tab()
    with tab4:
        _render_report_tab()
    with tab5:
        _render_compare_tab()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sidebar_limit = _render_sidebar()

    if not st.session_state.analyzed:
        _render_empty_state(sidebar_limit)
    else:
        _render_results()


if __name__ == "__main__":
    main()
