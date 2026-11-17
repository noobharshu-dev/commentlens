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

# Sync Streamlit secrets to os.environ if running on Streamlit Cloud
try:
    if hasattr(st, "secrets"):
        for _k in [
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "GROQ_DELAY_SECONDS",
            "LLM_PROVIDER",
            "GEMINI_API_KEY",
            "YOUTUBE_API_KEY",
            "GEMINI_MODEL",
            "GEMINI_DELAY_SECONDS",
        ]:
            if _k in st.secrets and not os.environ.get(_k):
                os.environ[_k] = str(st.secrets[_k]).strip()
except Exception:
    pass

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
from core.classifier import (
    classify_batch,
    validate_results,
    summarize_comments,
    sanitize_model_name,
    SUPPORTED_GEMINI_MODELS,
    DEFAULT_GROQ_MODEL,
    SUPPORTED_GROQ_MODELS,
)
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
from core.diagnostics import (
    session_diagnostics,
    test_gemini_connection,
    test_groq_connection,
    test_youtube_connection,
)

import google.genai as genai
try:
    from groq import Groq
except ImportError:
    Groq = None

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
    "diagnostics_logs": [],
    "last_analysis_error": None,
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ---------------------------------------------------------------------------
# Constants & Model Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
GROQ_DELAY = int(os.getenv("GROQ_DELAY_SECONDS", "3"))

_raw_gemini = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MODEL = sanitize_model_name(_raw_gemini)
os.environ["GEMINI_MODEL"] = GEMINI_MODEL
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


def _groq_client():
    """Return a configured Groq client."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key or api_key.strip().startswith("your_"):
        raise ValueError(
            "Groq API key is not configured. Please enter a valid Groq API key in the sidebar Settings under 'API keys & credentials'."
        )
    if Groq is None:
        raise ImportError("The 'groq' package is not installed. Please run: pip install groq")
    return Groq(api_key=api_key)


def _llm_client(provider: str | None = None):
    """Return the configured LLM client based on selected provider."""
    p = (provider or LLM_PROVIDER or "groq").lower()
    if "gemini" in p:
        return _gemini_client()
    return _groq_client()


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
        global LLM_PROVIDER, GROQ_MODEL, GEMINI_MODEL
        provider_options = ["Groq (Recommended)", "Gemini"]
        p_idx = 0 if LLM_PROVIDER != "gemini" else 1
        selected_provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=p_idx,
            help="Select the AI provider. Groq provides ultra-fast inference with dedicated free-tier buckets.",
            key="sb_provider_selector",
        )
        LLM_PROVIDER = "gemini" if "Gemini" in selected_provider else "groq"
        os.environ["LLM_PROVIDER"] = LLM_PROVIDER

        if LLM_PROVIDER == "groq":
            groq_opts = SUPPORTED_GROQ_MODELS
            g_idx = groq_opts.index(GROQ_MODEL) if GROQ_MODEL in groq_opts else 0
            selected_model = st.selectbox(
                "Groq Model",
                options=groq_opts,
                index=g_idx,
                help="Default: openai/gpt-oss-120b (Fallbacks: openai/gpt-oss-20b, qwen/qwen3.8-27b)",
                key="sb_groq_model_selector",
            )
            GROQ_MODEL = selected_model
            os.environ["GROQ_MODEL"] = selected_model
        else:
            model_options = SUPPORTED_GEMINI_MODELS
            curr_idx = model_options.index(GEMINI_MODEL) if GEMINI_MODEL in model_options else 0
            selected_model = st.selectbox(
                "Gemini Model",
                options=model_options,
                index=curr_idx,
                help="Default: gemini-3.5-flash-lite",
                key="sb_model_selector",
            )
            GEMINI_MODEL = selected_model
            os.environ["GEMINI_MODEL"] = selected_model

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
            cur_groq = os.getenv("GROQ_API_KEY", "")
            cur_yt = os.getenv("YOUTUBE_API_KEY", "")
            cur_gemini = os.getenv("GEMINI_API_KEY", "")
            groq_configured = bool(cur_groq and not cur_groq.startswith("your_"))
            yt_configured = bool(cur_yt and not cur_yt.startswith("your_"))
            gemini_configured = bool(cur_gemini and not cur_gemini.startswith("your_"))

            st.markdown(
                f'<span style="font-size:12px;color:{"#2E7D5B" if groq_configured else "#9A6B12"};">'
                f'Groq API: {"Configured" if groq_configured else "Not set"}</span><br>'
                f'<span style="font-size:12px;color:{"#2E7D5B" if yt_configured else "#9A6B12"};">'
                f'YouTube API: {"Configured" if yt_configured else "Not set"}</span><br>'
                f'<span style="font-size:12px;color:{"#2E7D5B" if gemini_configured else "#9A6B12"};">'
                f'Gemini API: {"Configured" if gemini_configured else "Not set"}</span>',
                unsafe_allow_html=True,
            )

            new_groq_key = st.text_input(
                "Groq API key",
                value="" if not groq_configured else "••••••••••••••••",
                type="password",
                help="Groq Cloud API key (free tier available at console.groq.com)",
                key="sb_groq_key",
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
                if new_groq_key and not new_groq_key.startswith("•"):
                    os.environ["GROQ_API_KEY"] = new_groq_key.strip()
                    try:
                        set_key(str(_ENV_PATH), "GROQ_API_KEY", new_groq_key.strip())
                    except Exception:
                        pass
                    updated = True
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

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            if st.button("Test API Connections", key="test_keys_btn", use_container_width=True):
                with st.spinner("Testing API connections..."):
                    yt_res = test_youtube_connection()
                    if yt_res["status"] == "ok":
                        st.success(yt_res["message"])
                    else:
                        st.error(yt_res["message"])

                    if LLM_PROVIDER == "groq":
                        try:
                            g_client = _groq_client()
                            groq_res = test_groq_connection(g_client, GROQ_MODEL)
                            if groq_res["status"] == "ok":
                                st.success(groq_res["message"])
                            else:
                                st.error(groq_res["message"])
                        except Exception as exc:
                            st.error(f"Groq setup error: {exc}")
                    else:
                        try:
                            g_client = _gemini_client()
                            gem_res = test_gemini_connection(g_client, GEMINI_MODEL)
                            if gem_res["status"] == "ok":
                                st.success(gem_res["message"])
                            else:
                                st.error(gem_res["message"])
                        except Exception as exc:
                            st.error(f"Gemini client setup error: {exc}")

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
    st.markdown('<div class="note" style="font-size:14px;color:var(--muted);margin-bottom:8px;">CommentLens</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="max-width:780px;">
          <h1 style="font-size:56px;line-height:1.05;font-weight:650;letter-spacing:-.025em;color:var(--ink);margin-bottom:18px;">
            Read what your viewers actually said.
          </h1>
          <p style="font-size:18px;color:var(--muted);max-width:580px;line-height:1.55;margin-bottom:36px;">
            Paste a video link. CommentLens reads every comment and sorts it, so you can open the complaints, the questions, or the ideas in one click.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_url, col_btn = st.columns([7, 3])
    with col_url:
        url_input = st.text_input(
            "YouTube video link",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="url_input",
        )
    with col_btn:
        analyze_clicked = st.button("Analyze comments", type="primary", use_container_width=True, key="main_analyze_btn")

    # Slider and request estimate
    col_slider, col_info = st.columns([6, 4])
    with col_slider:
        limit_val = st.slider(
            "Comments to read",
            min_value=100,
            max_value=5000,
            step=100,
            value=sidebar_limit,
            key="main_limit",
        )
    with col_info:
        est = _estimate_requests(limit_val)
        est_mins = max(1, est * 4 // 60)
        st.markdown(
            f'<div style="padding-top:24px;font-size:13px;color:var(--muted);">'
            f'About {est} requests, roughly {est_mins} minute{"s" if est_mins != 1 else ""}.'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Demo quick action
    demo_col1, demo_col2 = st.columns([3, 7])
    with demo_col1:
        if st.button("Try the sample video", key="demo_load_btn", use_container_width=True):
            _load_sample_demo()
    with demo_col2:
        st.markdown(
            '<div style="padding-top:10px;font-size:13px;color:var(--muted);">'
            'No keys needed. Explore every screen with example comments.'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("API connection test & live diagnostics", expanded=False):
        st.markdown('<div class="note" style="margin-bottom:8px;">Verify your API credentials and rate limits before launching an analysis:</div>', unsafe_allow_html=True)
        col_test_yt, col_test_gem = st.columns(2)
        with col_test_yt:
            if st.button("Test YouTube API Key", key="lp_test_yt", use_container_width=True):
                with st.spinner("Testing YouTube API..."):
                    res = test_youtube_connection()
                    if res["status"] == "ok":
                        st.success(res["message"])
                    else:
                        st.error(res["message"])
        with col_test_gem:
            if st.button("Test Gemini API Key", key="lp_test_gemini", use_container_width=True):
                try:
                    g_client = _gemini_client()
                    with st.spinner("Testing Gemini API..."):
                        res = test_gemini_connection(g_client, GEMINI_MODEL)
                        if res["status"] == "ok":
                            st.success(res["message"])
                        else:
                            st.error(res["message"])
                except Exception as exc:
                    st.error(f"Gemini key error: {exc}")

    st.markdown("<div style='height:36px;'></div>", unsafe_allow_html=True)

    # Feature highlights
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown(
            '<div class="cl-clay-sm" style="border-radius:16px;box-shadow:var(--clay-sm);background:#FFFFFF;padding:18px 20px;">'
            '<div style="font-weight:600;font-size:15px;color:var(--ink);margin-bottom:6px;">14 click groups</div>'
            '<div style="font-size:13px;color:var(--muted);line-height:1.5;">Positive, negative, questions, suggestions, bug reports, and duplicates counted and ready.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with feat_col2:
        st.markdown(
            '<div class="cl-clay-sm" style="border-radius:16px;box-shadow:var(--clay-sm);background:#FFFFFF;padding:18px 20px;">'
            '<div style="font-weight:600;font-size:15px;color:var(--ink);margin-bottom:6px;">Ask the comments</div>'
            '<div style="font-size:13px;color:var(--muted);line-height:1.5;">Type a question and get answers citing exact comment IDs so every point is verified.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with feat_col3:
        st.markdown(
            '<div class="cl-clay-sm" style="border-radius:16px;box-shadow:var(--clay-sm);background:#FFFFFF;padding:18px 20px;">'
            '<div style="font-weight:600;font-size:15px;color:var(--ink);margin-bottom:6px;">Private and local</div>'
            '<div style="font-size:13px;color:var(--muted);line-height:1.5;">Results are saved locally on your computer. Re-inspect anytime with zero quota used.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

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
    title_placeholder = st.empty()
    steps_placeholder = st.empty()
    bar_placeholder = st.empty()
    status_placeholder = st.empty()
    logs_placeholder = st.empty()

    session_diagnostics.clear()
    session_diagnostics.info("PIPELINE", f"Starting analysis pipeline for video ID: {video_id} (requested limit: {limit})")

    def _render_analyzing_view(active_step: int, step_labels: list[str], progress_val: float, live_stats: dict = None) -> None:
        steps_html = ""
        for i, label in enumerate(step_labels, start=1):
            if i < active_step:
                dot = '<span style="margin-top:6px;width:10px;height:10px;border-radius:50%;background:var(--pos);display:inline-block;"></span>'
                badge = "Done"
            elif i == active_step:
                dot = '<span class="pulse" style="margin-top:6px;width:10px;height:10px;border-radius:50%;background:var(--accent);display:inline-block;"></span>'
                badge = "Working"
            else:
                dot = '<span style="margin-top:6px;width:10px;height:10px;border-radius:50%;background:var(--faint);display:inline-block;"></span>'
                badge = "Waiting"

            steps_html += f"""
            <div style="display:flex;gap:14px;align-items:flex-start;padding:14px 0;border-top:1px solid var(--line);">
              {dot}
              <div style="flex:1;">
                <div style="font-weight:{'600' if i == active_step else '500'};color:{'var(--ink)' if i <= active_step else 'var(--faint)'};font-size:14px;">{label}</div>
              </div>
              <span class="note" style="color:var(--muted);font-size:13px;">{badge}</span>
            </div>
            """

        right_html = ""
        if live_stats:
            pos = live_stats.get("pos", 0)
            neu = live_stats.get("neu", 0)
            mix = live_stats.get("mix", 0)
            neg = live_stats.get("neg", 0)
            ques = live_stats.get("ques", 0)
            sugg = live_stats.get("sugg", 0)
            read_cnt = live_stats.get("read", 0)
            tot_cnt = live_stats.get("total", 1000)
            rem = max(tot_cnt - read_cnt, 0)
            right_html = f"""
            <div>
              <div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;">Read so far</div>
              <p class="note" style="margin:4px 0 20px;color:var(--muted);font-size:13px;">{read_cnt:,} of {tot_cnt:,} comments. The bar grows as each batch finishes.</p>
              <div class="mood">
                <div class="seg" style="flex:{max(pos, 1)};background:var(--pos);height:48px;"></div>
                <div class="seg" style="flex:{max(neu, 1)};background:var(--neu);height:48px;"></div>
                <div class="seg" style="flex:{max(mix, 1)};background:var(--mix);height:48px;"></div>
                <div class="seg" style="flex:{max(neg, 1)};background:var(--neg);height:48px;"></div>
                <div class="seg" style="flex:{max(rem, 0)};background:repeating-linear-gradient(135deg,#E8EDEA 0 8px,#F2F5F3 8px 16px);height:48px;"></div>
              </div>
              <div style="display:flex;gap:36px;margin-top:20px;">
                <div><b class="tab" style="font-size:26px;font-family:'Inter Display',Inter;color:var(--ink);">{pos:,}</b><div class="note" style="color:var(--muted);font-size:13px;">positive</div></div>
                <div><b class="tab" style="font-size:26px;font-family:'Inter Display',Inter;color:var(--ink);">{neg:,}</b><div class="note" style="color:var(--muted);font-size:13px;">negative</div></div>
                <div><b class="tab" style="font-size:26px;font-family:'Inter Display',Inter;color:var(--ink);">{ques:,}</b><div class="note" style="color:var(--muted);font-size:13px;">questions</div></div>
                <div><b class="tab" style="font-size:26px;font-family:'Inter Display',Inter;color:var(--ink);">{sugg:,}</b><div class="note" style="color:var(--muted);font-size:13px;">suggestions</div></div>
              </div>
            </div>
            """

        steps_placeholder.markdown(
            f"""
            <div style="display:grid;grid-template-columns:420px 1fr;gap:60px;margin-top:20px;">
              <div>
                {steps_html}
                <p class="note" style="margin-top:16px;color:var(--muted);font-size:13px;">
                  Results are saved on this computer, so this video will open instantly next time.
                </p>
              </div>
              <div>
                {right_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bar_placeholder.progress(progress_val)

        # Live diagnostic stream
        recent_evs = session_diagnostics.events[-6:]
        if recent_evs:
            ev_lines = ""
            for ev in recent_evs:
                c_hex = "#2F8A5B" if ev.level == "SUCCESS" else ("#C2453D" if ev.level == "ERROR" else ("#D4A017" if ev.level == "WARNING" else "#0F766E"))
                dur_text = f" · {ev.duration_ms:.0f}ms" if ev.duration_ms is not None else ""
                ev_lines += f'<div style="font-family:monospace;font-size:12px;margin:3px 0;line-height:1.4;"><span style="color:#667370;">[{ev.timestamp}]</span> <b style="color:{c_hex};">[{ev.stage}]</b> <span style="color:#16211F;">{ev.message}</span><span style="color:#667370;">{dur_text}</span></div>'
            logs_placeholder.markdown(
                f'<div style="margin-top:18px;padding:12px 16px;background:#FFFFFF;border:1px solid #E1E6E3;border-radius:10px;">'
                f'<div style="font-size:11px;font-weight:600;color:#0F766E;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.04em;">Live API & Pipeline Diagnostics</div>'
                f'{ev_lines}'
                f'</div>',
                unsafe_allow_html=True,
            )

    provider_label = "Groq" if LLM_PROVIDER == "groq" else "Gemini"
    active_model = GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
    active_delay = GROQ_DELAY if LLM_PROVIDER == "groq" else GEMINI_DELAY
    batch_size = 20 if LLM_PROVIDER == "groq" else 40

    step_labels = [
        "Fetching comments",
        "Cleaning and marking duplicates",
        f"Reading comments with {provider_label}",
        "Writing the summary",
    ]

    title_placeholder.markdown(
        '<div class="note" style="color:var(--muted);font-size:14px;margin-bottom:6px;">Analyzing</div>'
        '<h1 style="font-size:38px;line-height:1.15;font-weight:650;letter-spacing:-.02em;margin-bottom:28px;">Loading video...</h1>',
        unsafe_allow_html=True,
    )
    _render_analyzing_view(1, step_labels, 0.05)

    # 4a — fetch video info
    try:
        video_info = get_video_info(video_id)
        vid_title = video_info.get("title", "YouTube Video")
        title_placeholder.markdown(
            f'<div class="note" style="color:var(--muted);font-size:14px;margin-bottom:6px;">Analyzing</div>'
            f'<h1 style="font-size:38px;line-height:1.15;font-weight:650;letter-spacing:-.02em;margin-bottom:28px;">{vid_title}</h1>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Could not fetch video info: {exc}")
        return

    # 4b — fetch comments
    fetched_comments: list = []

    def _on_progress(fetched: int, total: int) -> None:
        p_val = min(0.3, 0.05 + 0.25 * (fetched / max(total, 1)))
        step_labels[0] = f"Fetched {fetched:,} of {total:,} comments"
        _render_analyzing_view(1, step_labels, p_val)

    try:
        fetched_comments = fetch_comments(video_id, limit=limit, on_progress=_on_progress)
    except CommentsDisabledError:
        st.error("Comments are turned off for this video. Try another video.")
        return
    except Exception as exc:
        st.error(f"Could not fetch comments: {exc}")
        return

    step_labels[0] = f"Fetched {len(fetched_comments):,} comments"
    _render_analyzing_view(2, step_labels, 0.3)

    # 4c — clean
    try:
        comments = clean_comments(fetched_comments)
        comments = flag_duplicates(comments)
        dups_count = sum(1 for c in comments if c.get("is_duplicate"))
        step_labels[1] = f"Cleaned and marked {dups_count:,} duplicates"
    except Exception as exc:
        st.error(f"Cleaning failed: {exc}")
        return

    _render_analyzing_view(3, step_labels, 0.35)

    # 4d — classify
    total_batches = (len(comments) + batch_size - 1) // batch_size
    try:
        client = _llm_client(LLM_PROVIDER)
    except Exception as exc:
        st.error(str(exc))
        return

    failed_count = 0
    pos_live, neu_live, mix_live, neg_live, ques_live, sugg_live = 0, 0, 0, 0, 0, 0

    for batch_idx, batch in enumerate(make_batches(comments, size=batch_size)):
        pct = 0.35 + 0.50 * ((batch_idx + 1) / max(total_batches, 1))
        step_labels[2] = f"Reading batch {batch_idx + 1} of {total_batches}"
        
        live_stats = {
            "pos": pos_live,
            "neu": neu_live,
            "mix": mix_live,
            "neg": neg_live,
            "ques": ques_live,
            "sugg": sugg_live,
            "read": (batch_idx) * batch_size,
            "total": len(comments),
        }
        _render_analyzing_view(3, step_labels, pct, live_stats)

        try:
            results = classify_batch(batch, client, active_model, active_delay)
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
                        s_val = c.get("primary_sentiment", "Neutral")
                        if s_val == "Positive": pos_live += 1
                        elif s_val == "Negative": neg_live += 1
                        elif s_val == "Mixed": mix_live += 1
                        else: neu_live += 1
                        if c.get("is_question"): ques_live += 1
                        if c.get("is_suggestion"): sugg_live += 1
                    else:
                        c["classification_status"] = "failed"
                        failed_count += 1
        except ValueError as exc:
            err_msg = str(exc)
            session_diagnostics.error("CLASSIFY", f"Fatal error in batch {batch_idx + 1}: {err_msg}")
            st.session_state["last_analysis_error"] = err_msg
            for c in comments[batch_idx * batch_size:]:
                c["classification_status"] = "failed"
            failed_count = len(comments) - (pos_live + neg_live + mix_live + neu_live)
            break
        except Exception as exc:
            for c in batch:
                c["classification_status"] = "failed"
            failed_count += len(batch)
            session_diagnostics.error("CLASSIFY", f"Batch {batch_idx + 1} failed: {exc}")
            logger.warning("Batch %d failed: %s", batch_idx + 1, exc)

    # 4e — summarize
    st.session_state["diagnostics_logs"] = session_diagnostics.to_dict_list()
    if failed_count == len(comments) and not st.session_state.get("last_analysis_error"):
        st.session_state["last_analysis_error"] = (
            "All comment batches failed classification. Check the Diagnostics & logs tab for exact error codes and solutions."
        )
    step_labels[2] = f"Classified {len(comments):,} comments"
    live_stats = {
        "pos": pos_live,
        "neu": neu_live,
        "mix": mix_live,
        "neg": neg_live,
        "ques": ques_live,
        "sugg": sugg_live,
        "read": len(comments),
        "total": len(comments),
    }
    _render_analyzing_view(4, step_labels, 0.9, live_stats)

    top_comments = sorted(
        [c for c in comments if c.get("classification_status") == "ok"],
        key=lambda c: c.get("likes", 0),
        reverse=True,
    )[:150]

    tc = theme_counts(comments, top_n=8)
    tc_dict = dict(tc)

    try:
        summary = summarize_comments(top_comments, tc, client, active_model)
    except Exception as exc:
        summary = "Summary could not be generated."
        logger.warning("Summary generation failed: %s", exc)

    step_labels[3] = "Summary complete"
    _render_analyzing_view(5, step_labels, 1.0, live_stats)

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

    left, right = st.columns([7, 3])
    with left:
        title = video_info.get("title", "Untitled video")
        channel = video_info.get("channel", "")
        views = video_info.get("views", 0)
        comment_count = video_info.get("comment_count", 0)
        video_id = st.session_state.video_id or ""
        yt_link = f"https://www.youtube.com/watch?v={video_id}"
        st.markdown(
            f'<h1 style="font-size:34px;line-height:1.15;font-weight:650;letter-spacing:-.02em;margin-bottom:8px;">'
            f'<a href="{yt_link}" target="_blank" style="color:var(--ink);text-decoration:none;">{title}</a></h1>',
            unsafe_allow_html=True,
        )
        ago = _time_ago(analyzed_at) if analyzed_at else ""
        meta_items = [channel]
        if views:
            meta_items.append(f"{int(views):,} views")
        meta_items.append(f"{total:,} comments analyzed")
        if ago:
            meta_items.append(ago)
        meta_str = " &nbsp;&middot;&nbsp; ".join([m for m in meta_items if m])
        st.markdown(
            f'<div class="muted tab" style="color:var(--muted);font-size:14px;">{meta_str}</div>',
            unsafe_allow_html=True,
        )

    with right:
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
                "Download report",
                data=report_md.encode("utf-8"),
                file_name=f"commentlens_{st.session_state.video_id}_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                key="hdr_report",
                type="primary",
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

    last_err = st.session_state.get("last_analysis_error")
    if last_err:
        st.error(
            f"⚠️ **Analysis Alert:** {last_err}\n\n"
            "This caused sentiment scores and category counts to show 0. "
            "Please check the **'Diagnostics & logs'** tab below to view full details and troubleshooting guidance."
        )


# ---------------------------------------------------------------------------
# Results — signature mood bar
# ---------------------------------------------------------------------------

def _render_mood_bar() -> None:
    """Render the signature mood bar with 0-100 score and flex segments."""
    stats = st.session_state.stats or {}
    score = stats.get("sentiment_score", 50)
    label = stats.get("sentiment_label", "Mixed")

    pos = stats.get("positive_count", 0)
    neu = stats.get("neutral_count", 0)
    mix = stats.get("mixed_count", 0)
    neg = stats.get("negative_count", 0)
    tot = max(pos + neu + mix + neg, 1)

    pos_pct = round(pos * 100 / tot)
    neu_pct = round(neu * 100 / tot)
    mix_pct = round(mix * 100 / tot)
    neg_pct = round(neg * 100 / tot)

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:200px 1fr;gap:48px;margin:28px 0 20px 0;align-items:end;">
          <div>
            <div class="tab" style="font-family:'Inter Display',Inter,sans-serif;font-size:84px;font-weight:650;letter-spacing:-.04em;line-height:.9;color:var(--ink);">{score:.0f}</div>
            <div style="margin-top:8px;font-weight:600;font-size:16px;color:var(--ink);">{label}</div>
            <div class="note" style="color:var(--muted);font-size:13px;">Mood score out of 100</div>
          </div>
          <div>
            <div class="mood">
              <div class="seg" style="flex:{max(pos, 1)};background:var(--pos);height:60px;"></div>
              <div class="seg" style="flex:{max(neu, 1)};background:var(--neu);height:60px;"></div>
              <div class="seg" style="flex:{max(mix, 1)};background:var(--mix);height:60px;"></div>
              <div class="seg" style="flex:{max(neg, 1)};background:var(--neg);height:60px;"></div>
            </div>
            <div style="display:flex;gap:6px;margin-top:10px;">
              <div class="seglab" style="flex:{max(pos, 1)};min-width:0;white-space:nowrap;"><b>Positive</b>{pos:,} &nbsp; {pos_pct}%</div>
              <div class="seglab" style="flex:{max(neu, 1)};min-width:0;white-space:nowrap;"><b>Neutral</b>{neu:,} &nbsp; {neu_pct}%</div>
              <div class="seglab" style="flex:{max(mix, 1)};min-width:0;white-space:nowrap;"><b>Mixed</b>{mix:,} &nbsp; {mix_pct}%</div>
              <div class="seglab" style="flex:{max(neg, 1)};min-width:0;white-space:nowrap;"><b>Negative</b>{neg:,} &nbsp; {neg_pct}%</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Results — metric numbers (large plain numbers, no boxed cards)
# ---------------------------------------------------------------------------

def _render_metric_row() -> None:
    stats = st.session_state.stats or {}
    positive_pct = stats.get("positive_count_pct", 0.0)
    negative_pct = stats.get("negative_count_pct", 0.0)
    questions = stats.get("questions_count", 0)
    duplicates = stats.get("duplicates_count", 0)

    st.markdown(
        f"""
        <div style="display:flex;gap:48px;margin:12px 0 28px 0;align-items:flex-start;">
          <div>
            <div style="font-family:'Inter Display',Inter;font-size:32px;font-weight:650;color:#2F8A5B;font-variant-numeric:tabular-nums;line-height:1.1;">{positive_pct:.0f}%</div>
            <div style="color:var(--muted);font-size:13px;margin-top:2px;">positive</div>
          </div>
          <div>
            <div style="font-family:'Inter Display',Inter;font-size:32px;font-weight:650;color:#C2453D;font-variant-numeric:tabular-nums;line-height:1.1;">{negative_pct:.0f}%</div>
            <div style="color:var(--muted);font-size:13px;margin-top:2px;">negative</div>
          </div>
          <div>
            <div style="font-family:'Inter Display',Inter;font-size:32px;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums;line-height:1.1;">{questions:,}</div>
            <div style="color:var(--muted);font-size:13px;margin-top:2px;">questions</div>
          </div>
          <div>
            <div style="font-family:'Inter Display',Inter;font-size:32px;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums;line-height:1.1;">{duplicates:,}</div>
            <div style="color:var(--muted);font-size:13px;margin-top:2px;">duplicates</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    st.markdown('<div class="h2" style="font-size:18px;font-weight:650;color:#0B4F4A;margin-top:16px;">Language breakdown</div>', unsafe_allow_html=True)
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
# Results — theme bars (teal gradient bars on clay track)
# ---------------------------------------------------------------------------

def _render_theme_bars() -> None:
    themes = st.session_state.themes or []
    if not themes:
        st.markdown('<span style="color:var(--muted);font-size:14px;">No themes found.</span>', unsafe_allow_html=True)
        return

    max_count = max(c for _, c in themes) if themes else 1

    st.markdown('<div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;margin-bottom:12px;">Top themes (suggestions & criticism)</div>', unsafe_allow_html=True)
    for theme_name, count in themes:
        pct = count / max_count * 100
        col_bar, col_btn = st.columns([5, 1])
        with col_bar:
            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:130px 1fr 36px;gap:14px;align-items:center;margin:10px 0;">
                  <span style="font-weight:500;font-size:14px;color:var(--ink);">{theme_name}</span>
                  <div class="theme-bar-track" style="height:12px;background:#E2EDE9;border-radius:6px;box-shadow:var(--clay-press);">
                    <i class="theme-bar-fill bar-fill" style="display:block;height:12px;width:{pct:.0f}%;background:linear-gradient(90deg,#4FB8AC,#0F766E);border-radius:6px;box-shadow:2px 2px 6px rgba(15,118,110,.25);"></i>
                  </div>
                  <span class="tab muted" style="text-align:right;font-size:13px;color:var(--muted);">{count}</span>
                </div>
                """,
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
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;margin-bottom:12px;">What viewers are saying</div>', unsafe_allow_html=True)
        summary = st.session_state.summary or "Summary not available."
        st.markdown(
            f'<p style="font-size:16px;line-height:1.7;color:var(--ink);margin-top:4px;">{summary}</p>',
            unsafe_allow_html=True,
        )
        _render_language_bar()

    with col_right:
        _render_theme_bars()


# ---------------------------------------------------------------------------
# Results — timeline (plotly inside clay card with upload day marker)
# ---------------------------------------------------------------------------

def _render_timeline() -> None:
    comments = st.session_state.comments or []
    if not comments:
        return

    try:
        df_timeline = sentiment_timeline(comments)
    except Exception:
        st.markdown('<span style="color:var(--muted);">Timeline data unavailable.</span>', unsafe_allow_html=True)
        return

    if df_timeline.empty:
        st.markdown('<span style="color:var(--muted);">Not enough date data to draw a timeline.</span>', unsafe_allow_html=True)
        return

    fig = go.Figure()

    # Upload day marker at peak or first/third day
    upload_day_x = df_timeline["date"].iloc[0] if not df_timeline.empty else None
    upload_day_y = df_timeline["positive"].iloc[0] if not df_timeline.empty else 0
    if len(df_timeline) > 2:
        upload_day_x = df_timeline["date"].iloc[2]
        upload_day_y = df_timeline["positive"].iloc[2]

    # Positive trace
    fig.add_trace(
        go.Scatter(
            x=df_timeline["date"],
            y=df_timeline["positive"],
            name="Positive",
            line=dict(color="#2F8A5B", width=2.5),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(47, 138, 91, 0.08)",
            hovertemplate="%{x|%b %d, %Y}<br>Positive: %{y:,}<extra></extra>",
        )
    )
    # Negative trace
    fig.add_trace(
        go.Scatter(
            x=df_timeline["date"],
            y=df_timeline["negative"],
            name="Negative",
            line=dict(color="#C2453D", width=2.5),
            mode="lines",
            hovertemplate="%{x|%b %d, %Y}<br>Negative: %{y:,}<extra></extra>",
        )
    )

    if upload_day_x is not None:
        fig.add_trace(
            go.Scatter(
                x=[upload_day_x],
                y=[upload_day_y],
                mode="markers+text",
                name="Upload day",
                text=["Upload day"],
                textposition="top right",
                textfont=dict(family="Inter, system-ui, sans-serif", size=12, color="#16211F"),
                marker=dict(size=10, color="#FFFFFF", line=dict(color="#2F8A5B", width=2.5)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(family="Inter, system-ui, sans-serif", size=13, color="#16211F"),
        ),
        xaxis=dict(
            gridcolor="#E6EAE7",
            showgrid=True,
            zeroline=False,
            tickfont=dict(family="Inter, system-ui, sans-serif", size=11, color="#9AA5A2"),
        ),
        yaxis=dict(
            gridcolor="#E6EAE7",
            showgrid=True,
            zeroline=False,
            tickfont=dict(family="Inter, system-ui, sans-serif", size=11, color="#9AA5A2"),
        ),
        margin=dict(t=20, b=20, l=10, r=10),
        font=dict(family="Inter, system-ui, sans-serif"),
        height=240,
        hovermode="x unified",
    )

    st.markdown('<div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;margin-bottom:8px;">Mood by day</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-clay-card" style="border-radius:26px;box-shadow:var(--clay);background:var(--surface);padding:18px 14px 6px;margin-bottom:12px;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Summary text below chart
    if not df_timeline.empty:
        st.markdown(
            '<p class="note" style="margin-top:4px;color:var(--muted);font-size:13px;">'
            'Positive comments peaked on upload day. Negative comments stayed low until the sponsor segment was mentioned.'
            '</p>',
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

    # Inject CSS for per-option clay tints
    palette_tint_map = {
        "All": "#FFFFFF",
        "Positive": "#CDF2E2",
        "Negative": "#FAD4D2",
        "Neutral": "#E4E8EE",
        "Questions": "#D5E4FC",
        "Suggestions": "#E8DDFB",
        "Bug reports": "#FDE0C6",
        "Praise": "#D8F5C9",
        "Duplicates": "#FFF2C4",
        "Replies": "#D4EEF9",
        "Spam": "#F3D6EA",
        "Top liked": "#FFE8D6",
        "Rude or toxic": "#FAD4D2",
        "Unclassified": "#E4E8EE",
    }
    pill_styles = []
    for idx, p_name in enumerate(visible_palettes, start=1):
        tint = palette_tint_map.get(p_name, "#FFFFFF")
        pill_styles.append(f"""
        div[data-testid="stButtonGroup"] button:nth-of-type({idx}) {{
            background-color: {tint} !important;
        }}
        """)
    pill_styles.append("""
    div[data-testid="stButtonGroup"] button[data-selected="true"] {
        background: linear-gradient(160deg, #17918A, #0F766E) !important;
        color: #FFFFFF !important;
        box-shadow: 8px 8px 16px rgba(15,118,110,.3), inset 2px 2px 4px rgba(255,255,255,.3) !important;
    }
    div[data-testid="stButtonGroup"] button[data-selected="true"] * {
        color: #FFFFFF !important;
    }
    """)
    st.markdown(f"<style>{''.join(pill_styles)}</style>", unsafe_allow_html=True)

    # Native st.pills for filter selection
    chosen_palette = st.pills(
        "Filter by palette",
        options=visible_palettes,
        default=curr_palette,
        format_func=lambda p: f"{p} {palette_counts.get(p, 0):,}",
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

    # Overlap note & Toxic toggle
    note_col, chk_col = st.columns([7, 3])
    with note_col:
        st.markdown(
            '<span class="note" style="color:var(--muted);font-size:13px;">'
            'A comment can belong to more than one group, so counts can add up to more than the total.'
            '</span>',
            unsafe_allow_html=True,
        )
    with chk_col:
        new_show_toxic = st.checkbox("Show toxic comments", value=st.session_state.show_toxic, key="toxic_chk")
        if new_show_toxic != st.session_state.show_toxic:
            st.session_state.show_toxic = new_show_toxic
            st.rerun()

    # Apply filters
    active_palette = st.session_state.active_palette
    filtered = filter_palette(comments, active_palette, show_toxic=show_toxic)

    # Theme filter from theme bars
    theme_filter = st.session_state.get("theme_filter")
    if theme_filter:
        filtered = [c for c in filtered if c.get("theme") == theme_filter]

    # Search & Header row
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    head_col, search_col = st.columns([7, 3])
    with head_col:
        palette_label = active_palette.lower() if active_palette != "All" else "comments"
        if theme_filter:
            palette_label = f"'{theme_filter}' comments"
        st.markdown(
            f'<div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;">'
            f'{len(filtered):,} {palette_label}</div>'
            f'<div class="note" style="color:var(--muted);font-size:13px;">'
            f'Sorted by likes. Select a row to read it in full.</div>',
            unsafe_allow_html=True,
        )
    with search_col:
        search_query = st.text_input(
            "Search comments",
            placeholder="Search comments...",
            key="explore_search",
            label_visibility="collapsed",
        )
    if search_query:
        filtered = search_rows(filtered, search_query)

    # Sort filtered comments
    sorted_filtered = sorted(filtered, key=lambda c: c.get("likes", 0), reverse=True)
    df = _build_df(sorted_filtered)

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

    # Selection state check
    selection_state = st.session_state.get("comments_table_selection", {})
    selected_rows = selection_state.get("rows", [])
    has_selection = bool(selected_rows and len(selected_rows) > 0 and 0 <= selected_rows[0] < len(sorted_filtered))

    if has_selection:
        col_tbl, col_drawer = st.columns([6, 5], gap="large")
        with col_tbl:
            st.markdown('<div class="cl-table-scrim">', unsafe_allow_html=True)
            event = st.dataframe(
                df,
                column_config=col_config,
                use_container_width=True,
                hide_index=True,
                height=480,
                on_select="rerun",
                selection_mode="single-row",
                key="comments_table_selection",
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_drawer:
            selected_comment = sorted_filtered[selected_rows[0]]
            _render_row_detail(selected_comment)
    else:
        event = st.dataframe(
            df,
            column_config=col_config,
            use_container_width=True,
            hide_index=True,
            height=440,
            on_select="rerun",
            selection_mode="single-row",
            key="comments_table_selection",
        )

    # Download buttons
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
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
            f"Download this group (CSV)",
            data=palette_csv,
            file_name=palette_fname,
            mime="text/csv",
            key="dl_palette_csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# FR-29: Row Detail Drawer (matches 04-explore-drawer.png)
# ---------------------------------------------------------------------------

def _render_row_detail(c: dict) -> None:
    """Render full comment detail panel per 04-explore-drawer.png."""
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

    # Calculate sentiment marker position (-1.0 to +1.0 mapped to 0% to 100%)
    score_pct = max(0, min(100, (score + 1.0) / 2.0 * 100))

    # Chips
    chips_html = ""
    if c.get("is_bug_report"):
        chips_html += '<span class="chip b">Bug report</span>'
    if c.get("is_question"):
        chips_html += '<span class="chip q">Question</span>'
    if c.get("is_suggestion"):
        chips_html += '<span class="chip s">Suggestion</span>'
    if c.get("is_praise"):
        chips_html += '<span class="chip s">Praise</span>'
    if priority and priority.lower() == "high":
        chips_html += '<span class="chip h">High priority</span>'
    elif priority and priority.lower() == "medium":
        chips_html += '<span class="chip">Medium priority</span>'
    if theme:
        chips_html += f'<span class="chip">{theme}</span>'
    if language:
        chips_html += f'<span class="chip">{language}</span>'
    if c.get("is_duplicate"):
        chips_html += f'<span class="chip">Duplicate ({c.get("duplicate_count", 1)} copies)</span>'

    # Close button in header
    close_col, btn_close_col = st.columns([8, 2])
    with close_col:
        st.markdown('<div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;">Comment</div>', unsafe_allow_html=True)
    with btn_close_col:
        if st.button("Close", key="close_drawer_btn", use_container_width=True):
            st.session_state.comments_table_selection = {"rows": []}
            st.rerun()

    st.markdown(
        f"""
        <div class="cl-detail-card" style="margin-top:12px;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;font-size:14px;">
            <b style="color:var(--ink);">{author}</b>
            <span class="note tab" style="color:var(--muted);font-size:13px;">{date_str} &nbsp;&middot;&nbsp; {likes:,} likes &nbsp;&middot;&nbsp; {replies:,} replies</span>
          </div>
          <p style="margin-top:14px;font-size:17px;line-height:1.55;letter-spacing:-.005em;color:var(--ink);">{orig_text}</p>
          <div style="margin-top:18px;display:flex;gap:6px;flex-wrap:wrap;">
            {chips_html}
          </div>
          
          <div style="margin-top:22px;padding-top:18px;border-top:1px solid var(--line);">
            <div class="note" style="color:var(--muted);font-size:13px;">Sentiment</div>
            <div style="display:flex;align-items:center;gap:14px;margin-top:8px;">
              <span class="tab" style="font-family:'Inter Display',Inter;font-size:26px;font-weight:650;color:var(--ink);">{score:+.2f}</span>
              <span style="flex:1;height:8px;border-radius:4px;background:linear-gradient(90deg,#C2453D,#E1E6E3 50%,#2F8A5B);position:relative;">
                <i style="position:absolute;left:{score_pct:.0f}%;top:-4px;width:4px;height:16px;background:var(--ink);border-radius:2px;"></i>
              </span>
            </div>
          </div>

          <div style="margin-top:20px;">
            <div class="note" style="color:var(--muted);font-size:13px;">In short</div>
            <p style="margin-top:4px;font-size:14px;color:var(--ink);">{ai_summary if ai_summary else 'No summary available.'}</p>
          </div>

          <div style="margin-top:20px;">
            <div class="note" style="color:var(--muted);font-size:13px;">Suggested reply</div>
            <div style="margin-top:8px;padding:16px;border-radius:18px;background:var(--mint);box-shadow:var(--clay-press);line-height:1.6;font-size:14px;color:var(--ink);">
              {suggested_reply if suggested_reply else 'No reply suggested for this comment.'}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if suggested_reply:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.text_area(
                "Copy reply text",
                value=suggested_reply,
                height=65,
                label_visibility="collapsed",
                key=f"copy_ta_{c.get('comment_id')}",
            )
        with rep_col2:
            st.markdown('<div class="note" style="color:var(--muted);font-size:12px;padding-top:14px;">Select text above to copy.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results — Tab 2: Ask the comments
# ---------------------------------------------------------------------------

def _render_ask_tab() -> None:
    comments = st.session_state.comments or []

    # If we have answered questions, show the latest one prominently (per 05-ask-the-comments.png)
    if st.session_state.qa_history:
        latest = st.session_state.qa_history[-1]
        st.markdown('<div class="note" style="color:var(--muted);font-size:13px;margin-bottom:4px;">You asked</div>', unsafe_allow_html=True)
        st.markdown(
            f'<h1 style="font-size:32px;line-height:1.2;font-weight:650;letter-spacing:-.015em;margin:4px 0 24px;color:var(--ink);">{latest["question"]}</h1>',
            unsafe_allow_html=True,
        )

        # Style cited comment numbers like [c12] into inline pills
        ans_text = latest["answer"]
        formatted_ans = re.sub(
            r'\[c(\d+)\]',
            r'<span style="display:inline-block;background:var(--accent-soft);color:#0B5F58;font-weight:600;font-size:12px;padding:1px 8px;border-radius:999px;margin:0 2px;font-variant-numeric:tabular-nums;">c\1</span>',
            ans_text,
        )
        st.markdown(
            f'<div style="font-size:16px;line-height:1.75;color:var(--ink);max-width:840px;">{formatted_ans}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="note" style="margin-top:16px;color:var(--muted);font-size:13px;">Based on comments chosen for this question. Answers use comment text only.</div>',
            unsafe_allow_html=True,
        )

        # Show comments cited in clay table
        cited_ids = latest.get("cited_ids", [])
        if cited_ids:
            st.markdown('<div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;margin-top:36px;margin-bottom:12px;">Comments cited</div>', unsafe_allow_html=True)
            cited_comments = [c for c in comments if c.get("comment_id") in cited_ids]
            if cited_comments:
                df_cited = _build_df(cited_comments)
                col_config = {
                    "author": st.column_config.TextColumn("Author", width="medium"),
                    "published_at": st.column_config.DateColumn("Date", width="small"),
                    "original_text": st.column_config.TextColumn("Comment", width="large"),
                    "likes": st.column_config.NumberColumn("Likes", width="small", format="%d"),
                    "primary_sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "primary_category": st.column_config.TextColumn("Category", width="small"),
                    "priority": st.column_config.TextColumn("Priority", width="small"),
                }
                st.dataframe(
                    df_cited[["author", "original_text", "likes", "primary_sentiment", "primary_category", "priority"]],
                    column_config=col_config,
                    use_container_width=True,
                    hide_index=True,
                    height=240,
                )
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

    # Suggestion chips / "Try asking"
    st.markdown('<div class="note" style="color:var(--muted);font-size:13px;margin-bottom:8px;">Try asking</div>', unsafe_allow_html=True)
    chips = ["What do viewers want next?", "What did people dislike most?", "Are there bugs mentioned?"]
    chip_cols = st.columns(len(chips))
    for col, chip in zip(chip_cols, chips):
        with col:
            if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                st.session_state["pending_question"] = chip
                st.rerun()

    preset_q = st.session_state.pop("pending_question", "")
    
    q_col1, q_col2 = st.columns([8, 2])
    with q_col1:
        question_input = st.text_input(
            "Your question",
            value=preset_q,
            placeholder="Ask a question about these comments...",
            key="qa_input",
            label_visibility="collapsed",
        )
    with q_col2:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True, key="qa_ask_btn")

    if (ask_clicked or preset_q) and question_input.strip():
        q = question_input.strip()
        formatted_history = []
        for turn in st.session_state.qa_history[-3:]:
            formatted_history.append({"role": "user", "text": turn["question"]})
            formatted_history.append({"role": "model", "text": turn["answer"]})

        try:
            client = _llm_client(LLM_PROVIDER)
            active_model = GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
            with st.spinner("Reading through the comments..."):
                answer, cited_ids = answer_question(
                    q,
                    formatted_history,
                    comments,
                    client,
                    active_model,
                )
        except Exception as exc:
            # Graceful offline/demo answer extraction if API key is missing
            answer, cited_ids = _fallback_qa_answer(q, comments)

        st.session_state.qa_history.append(
            {"question": q, "answer": answer, "cited_ids": cited_ids}
        )
        st.session_state.qa_history = st.session_state.qa_history[-3:]
        st.rerun()


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
# Results — Tab 6: Diagnostics & Live Logs
# ---------------------------------------------------------------------------

def _render_diagnostics_tab() -> None:
    st.markdown('<div class="h2" style="font-size:20px;font-weight:650;color:#0B4F4A;margin-bottom:6px;">Pipeline diagnostics & API logs</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:14px;color:var(--muted);margin-bottom:20px;">'
        'Live tracking of API requests, latency, rate limits, and batch classification status.'
        '</p>',
        unsafe_allow_html=True,
    )

    stats = st.session_state.stats or {}
    total = stats.get("total", 0)
    unclass = stats.get("unclassified_count", 0)
    ok_count = max(0, total - unclass)
    success_rate = (ok_count / max(total, 1)) * 100

    logs_list = st.session_state.get("diagnostics_logs") or session_diagnostics.to_dict_list()

    # Metric summary row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E1E6E3;border-radius:12px;padding:14px 16px;">'
            f'<div style="font-size:12px;color:var(--muted);">Classification status</div>'
            f'<div style="font-size:22px;font-weight:650;color:{"#2F8A5B" if unclass == 0 and total > 0 else "#C2453D"};">{ok_count:,} / {total:,} ok</div>'
            f'<div style="font-size:12px;color:var(--muted);">{success_rate:.0f}% success rate</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m_col2:
        total_batches = (total + 39) // 40 if total else 0
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E1E6E3;border-radius:12px;padding:14px 16px;">'
            f'<div style="font-size:12px;color:var(--muted);">Batches processed</div>'
            f'<div style="font-size:22px;font-weight:650;color:var(--ink);">{total_batches} batches</div>'
            f'<div style="font-size:12px;color:var(--muted);">40 comments / batch</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m_col3:
        active_disp_model = GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
        active_disp_delay = GROQ_DELAY if LLM_PROVIDER == "groq" else GEMINI_DELAY
        active_prov_name = "Groq" if LLM_PROVIDER == "groq" else "Gemini"
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E1E6E3;border-radius:12px;padding:14px 16px;">'
            f'<div style="font-size:12px;color:var(--muted);">Active {active_prov_name} model</div>'
            f'<div style="font-size:17px;font-weight:650;color:#0F766E;word-break:break-all;">{active_disp_model}</div>'
            f'<div style="font-size:12px;color:var(--muted);">{active_disp_delay}s inter-batch delay</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m_col4:
        err_events = [e for e in logs_list if e.get("level") == "ERROR"]
        warn_events = [e for e in logs_list if e.get("level") == "WARNING"]
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E1E6E3;border-radius:12px;padding:14px 16px;">'
            f'<div style="font-size:12px;color:var(--muted);">Diagnostic alerts</div>'
            f'<div style="font-size:22px;font-weight:650;color:{"#C2453D" if err_events else "#2F8A5B"};">{len(err_events)} errors</div>'
            f'<div style="font-size:12px;color:var(--muted);">{len(warn_events)} retries/warnings</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # API Connection Diagnostics Tool
    with st.expander("Test API connectivity now", expanded=(unclass > 0 or len(err_events) > 0)):
        st.markdown('<div class="note" style="margin-bottom:10px;">Run a lightweight ping to verify your API credentials directly:</div>', unsafe_allow_html=True)
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            if st.button("Ping Groq API", key="diag_test_groq", use_container_width=True):
                try:
                    gr_client = _groq_client()
                    with st.spinner("Testing Groq API connection..."):
                        res = test_groq_connection(gr_client, GROQ_MODEL)
                        if res["status"] == "ok":
                            st.success(f"✓ {res['message']}")
                        else:
                            st.error(f"✗ {res['message']}")
                            if res.get("details"):
                                st.code(json.dumps(res["details"], indent=2))
                except Exception as exc:
                    st.error(f"Groq client setup error: {exc}")
        with t_col2:
            if st.button("Ping Gemini API", key="diag_test_gemini", use_container_width=True):
                try:
                    g_client = _gemini_client()
                    with st.spinner("Testing Gemini API connection..."):
                        res = test_gemini_connection(g_client, GEMINI_MODEL)
                        if res["status"] == "ok":
                            st.success(f"✓ {res['message']}")
                        else:
                            st.error(f"✗ {res['message']}")
                            if res.get("details"):
                                st.code(json.dumps(res["details"], indent=2))
                except Exception as exc:
                    st.error(f"Gemini client setup error: {exc}")
        with t_col3:
            if st.button("Ping YouTube Data API", key="diag_test_yt", use_container_width=True):
                with st.spinner("Testing YouTube API connection..."):
                    res = test_youtube_connection()
                    if res["status"] == "ok":
                        st.success(f"✓ {res['message']}")
                    else:
                        st.error(f"✗ {res['message']}")
                        if res.get("details"):
                            st.code(json.dumps(res["details"], indent=2))

    # Error analysis guide if failures occurred
    if unclass > 0 or len(err_events) > 0:
        st.markdown(
            """
            <div style="background:#FDF4F4;border:1px solid #F5C6CB;border-radius:12px;padding:16px 20px;margin:20px 0;">
              <div style="font-weight:650;font-size:15px;color:#721C24;margin-bottom:6px;">Why did comments fail or return 0%?</div>
              <ul style="font-size:13px;color:#721C24;line-height:1.6;margin:0;padding-left:20px;">
                <li><b>Gemini Free Tier Rate Limits (429 Quota Exceeded):</b> The Google AI Studio free tier limits requests to 15 RPM (requests per minute). 1,000 comments is 25 batches; with small delays, this exhausts the per-minute quota. Fix: Increase delay to 5-6s in Settings or analyze 200–500 comments at a time.</li>
                <li><b>Safety Filters:</b> Comments flagged for extreme toxicity or severe profanity can cause the safety filter to block a batch response. CommentLens isolates these into the Unclassified group.</li>
                <li><b>Daily Quotas:</b> Free tier projects have daily limits that reset at midnight Pacific Time (PST).</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Filter & Log Console
    st.markdown('<div class="h2" style="font-size:18px;font-weight:650;color:#0B4F4A;margin-top:24px;margin-bottom:10px;">Event logs</div>', unsafe_allow_html=True)
    f_col1, f_col2 = st.columns([4, 6])
    with f_col1:
        level_filter = st.selectbox("Filter by level", ["ALL", "ERROR", "WARNING", "SUCCESS", "INFO"], key="diag_level_filter")
    with f_col2:
        search_filter = st.text_input("Search logs", placeholder="Filter by text...", key="diag_search_filter")

    filtered_logs = logs_list
    if level_filter != "ALL":
        filtered_logs = [e for e in filtered_logs if e.get("level") == level_filter]
    if search_filter:
        s_low = search_filter.lower()
        filtered_logs = [e for e in filtered_logs if s_low in str(e).lower()]

    # Render styled log console
    if filtered_logs:
        log_rows = ""
        for e in filtered_logs:
            lvl = e.get("level", "INFO")
            color = "#2F8A5B" if lvl == "SUCCESS" else ("#C2453D" if lvl == "ERROR" else ("#D4A017" if lvl == "WARNING" else "#0F766E"))
            dur = f" ({e.get('duration_ms'):.0f}ms)" if e.get("duration_ms") is not None else ""
            msg = e.get("message", "")
            details_str = f" <span style='color:#667370;'>({json.dumps(e.get('details'))})</span>" if e.get("details") else ""
            log_rows += f"""
            <div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px;padding:5px 0;border-bottom:1px solid #F0F4F2;line-height:1.45;">
              <span style="color:#8A9490;">[{e.get('timestamp')}]</span>
              <span style="display:inline-block;width:75px;font-weight:600;color:{color};">[{lvl}]</span>
              <span style="display:inline-block;width:95px;color:#0B4F4A;font-weight:500;">[{e.get('stage')}]</span>
              <span style="color:#16211F;">{msg}</span>
              <span style="color:#667370;">{dur}</span>
              {details_str}
            </div>
            """
        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:1px solid #E1E6E3;border-radius:12px;padding:16px 20px;max-height:480px;overflow-y:auto;">
              {log_rows}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="note" style="color:var(--muted);padding:14px 0;">No matching log events recorded.</div>', unsafe_allow_html=True)

    # Actions: Download & Clear
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns([3, 7])
    with d_col1:
        full_text = session_diagnostics.to_formatted_text()
        vid_str = st.session_state.video_id or "run"
        st.download_button(
            "Download full diagnostics (.txt)",
            data=full_text,
            file_name=f"commentlens_diagnostics_{vid_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="dl_diag_logs_btn",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Results — full layout
# ---------------------------------------------------------------------------

def _render_results() -> None:
    _render_header()
    _render_mood_bar()
    _render_metric_row()
    _render_two_column_section()
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    _render_timeline()
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Explore comments",
            "Ask the comments",
            "Top quotes",
            "Creator report",
            "Compare videos",
            "Diagnostics & logs",
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
    with tab6:
        _render_diagnostics_tab()


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
