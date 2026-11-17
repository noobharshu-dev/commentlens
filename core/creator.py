"""
core/creator.py
---------------
Creator-focused tools: top quotes, Q&A chat, and Markdown report generation.

Functions
---------
pick_top_quotes(comments)                                           -> list[dict]
answer_question(question, history, comments, client, model_name)   -> tuple[str, list[int]]
build_creator_report(video_info, stats, summary, themes, quotes,
                     top_suggestions, bug_reports, suggested_replies) -> str
"""

from __future__ import annotations

import re
import textwrap
from datetime import date
from typing import Optional

from google import genai
from google.genai import types

from core.classifier import (
    sanitize_model_name,
    SUPPORTED_GEMINI_MODELS,
    DEFAULT_GROQ_MODEL,
    SUPPORTED_GROQ_MODELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_citations(text: str) -> list[int]:
    """Extract comment IDs from citations like [c12], [c12, c88], or [12]."""
    cited: list[int] = []
    for block in re.findall(r"\[(.*?)\]", text):
        for num in re.findall(r"\bc?(\d+)\b", block, re.IGNORECASE):
            try:
                cited.append(int(num))
            except ValueError:
                pass
    return list(dict.fromkeys(cited))


def _trim_words(text: str, max_words: int = 25) -> str:
    """Return *text* truncated to *max_words* words, appending '…' if clipped."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _keyword_match(comment: dict, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the comment's text or summary."""
    haystack = (
        comment.get("original_text", "").lower()
        + " "
        + comment.get("ai_summary", "").lower()
    )
    return any(kw in haystack for kw in keywords)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pick_top_quotes(comments: list[dict]) -> list[dict]:
    """
    Select up to 5 highlight quotes from non-toxic praise or suggestion comments.

    Strategy
    --------
    - Exclude toxic, spam, and emoji-only comments.
    - Consider only praise (is_praise=True) or suggestion (is_suggestion=True).
    - Sort by likes descending.
    - Pick one per unique theme where possible (up to 5 total).

    Returns
    -------
    List of dicts with keys: author, quote, likes, theme.
    """
    candidates = [
        c for c in comments
        if not c.get("is_rude_or_toxic")
        and not c.get("is_spam_or_bot")
        and not c.get("is_emoji_only")
        and (c.get("is_praise") or c.get("is_suggestion"))
        and c.get("original_text", "").strip()
    ]

    # Sort by likes descending.
    candidates.sort(key=lambda c: c.get("likes", 0), reverse=True)

    seen_themes: set[str] = set()
    quotes: list[dict] = []

    # First pass: one per theme.
    for c in candidates:
        if len(quotes) >= 5:
            break
        theme = c.get("theme", "").strip() or "general"
        if theme not in seen_themes:
            seen_themes.add(theme)
            quotes.append(
                {
                    "author": c.get("author", "Anonymous"),
                    "quote": _trim_words(c.get("original_text", ""), 25),
                    "likes": c.get("likes", 0),
                    "theme": theme,
                }
            )

    # Second pass: fill up to 5 from remaining candidates (any theme).
    if len(quotes) < 5:
        used_ids = {q["quote"] for q in quotes}
        for c in candidates:
            if len(quotes) >= 5:
                break
            trimmed = _trim_words(c.get("original_text", ""), 25)
            if trimmed not in used_ids:
                used_ids.add(trimmed)
                theme = c.get("theme", "").strip() or "general"
                quotes.append(
                    {
                        "author": c.get("author", "Anonymous"),
                        "quote": trimmed,
                        "likes": c.get("likes", 0),
                        "theme": theme,
                    }
                )

    return quotes[:5]


def answer_question_groq(
    question: str,
    history: list[dict],
    comments: list[dict],
    client,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> tuple[str, list[int]]:
    """
    Answer a creator's question about comments using Groq.
    """
    system_instruction = (
        "You are an assistant helping a YouTube creator understand their audience feedback.\n"
        "Answer the question concisely and accurately based ONLY on the provided comment data.\n"
        "Cite specific comments using [c<id>] format (e.g. [c12], [c45]).\n"
        "Do not invent feedback that is not in the data.\n"
        "Keep your answer under 150 words. No emoji."
    )

    sample = sorted(
        comments,
        key=lambda c: (str(c.get("priority", "")).lower() == "high", c.get("likes", 0)),
        reverse=True,
    )[:60]

    context_lines = "\n".join(
        f"[c{c.get('comment_id', i)}] ({c.get('primary_sentiment', 'Neutral')}) {c.get('original_text', '')[:200]}"
        for i, c in enumerate(sample)
    )

    messages = [{"role": "system", "content": system_instruction}]
    recent_history = history[-3:] if len(history) > 3 else history
    for turn in recent_history:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("text", "")})

    messages.append({
        "role": "user",
        "content": f"Comment data:\n{context_lines}\n\nQuestion: {question}"
    })

    models_to_try = [model_name] if model_name else []
    for m in SUPPORTED_GROQ_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    answer_str = ""
    for model_candidate in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_candidate,
                messages=messages,
                temperature=0.3,
            )
            if response and response.choices and response.choices[0].message.content:
                answer_str = response.choices[0].message.content.strip()
                break
        except Exception:
            continue

    if not answer_str:
        raise ValueError("Could not generate an answer with the available Groq models.")

    cited_ids = _extract_citations(answer_str)
    return answer_str, cited_ids


def answer_question(
    question: str,
    history: list[dict],
    comments: list[dict],
    client,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> tuple[str, list[int]]:
    """
    Answer a creator's question grounded in comment data.
    Routes to Groq or Gemini based on client or model name.

    Strategy
    --------
    - Extract keywords from *question* (words > 3 chars).
    - Retrieve up to 150 comments matching any keyword.
    - Add top 100 by likes (de-duplicated).
    - Cap total context at 200 comments.
    - Build a Gemini chat with the last 3 turns of *history*.
    - Instruct the model to cite comment IDs like [c12] or [c88, c34].

    Parameters
    ----------
    question   : User's natural-language question.
    history    : List of {"role": "user"|"model", "text": str} turns.
    comments   : Full list of classified comment dicts.
    client     : Initialised google.genai.Client or Groq client.
    model_name : Model identifier.

    Returns
    -------
    (answer_str, cited_ids_list)
        answer_str    : Plain answer text with inline [cN] citations.
        cited_ids_list: List of unique integer comment IDs cited.
    """
    if hasattr(client, "chat") or model_name in SUPPORTED_GROQ_MODELS or any(p in str(model_name) for p in ["openai/", "qwen/", "llama", "groq"]):
        return answer_question_groq(question, history, comments, client, model_name)
    # Build keyword list.
    keywords = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]

    # Keyword-matching comments.
    matched = [c for c in comments if _keyword_match(c, keywords)][:150]

    # Top-liked comments (top 100).
    by_likes = sorted(comments, key=lambda c: c.get("likes", 0), reverse=True)[:100]

    # Merge, de-duplicate by comment_id.
    seen_ids: set[int] = set()
    context_comments: list[dict] = []
    for c in matched + by_likes:
        cid = c.get("comment_id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            context_comments.append(c)
        if len(context_comments) >= 200:
            break

    # Build context block.
    context_lines = "\n".join(
        f"[c{c['comment_id']}] ({c.get('primary_sentiment','?')}, "
        f"{c.get('likes',0)} likes) {c.get('original_text','')[:300]}"
        for c in context_comments
    )

    system_instruction = (
        "You are an assistant helping a YouTube content creator understand their audience.\n"
        "Answer questions based ONLY on the comment data provided.\n"
        "When you reference a specific comment, cite it inline as [cN] where N is the comment_id.\n"
        "Be concise. Use plain prose. No emoji. If the data does not support a claim, say so.\n"
    )

    # Build conversation history (last 3 turns only).
    recent_history = history[-3:] if len(history) > 3 else history

    contents: list[types.Content] = []
    for turn in recent_history:
        role = turn.get("role", "user")
        text = turn.get("text", "")
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    # Add current question with context.
    user_message = (
        f"Comment data:\n{context_lines}\n\n"
        f"Question: {question}"
    )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    active_model = sanitize_model_name(model_name)
    models_to_try = [active_model]
    for fallback in SUPPORTED_GEMINI_MODELS:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    answer_str = ""
    for model_candidate in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_candidate,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                ),
            )
            answer_str = response.text.strip()
            break
        except Exception:
            continue

    if not answer_str:
        raise ValueError("Could not generate an answer with the available Gemini models.")

    # Extract cited comment IDs.
    cited_ids = _extract_citations(answer_str)

    return answer_str, cited_ids


def build_creator_report(
    video_info: dict,
    stats: dict,
    summary: str,
    themes: list[tuple[str, int]],
    quotes: list[dict],
    top_suggestions: list[dict],
    bug_reports: list[dict],
    suggested_replies: list[dict],
) -> str:
    """
    Build a plain Markdown creator report.

    Parameters
    ----------
    video_info        : dict with keys title, channel, views, comment_count.
    stats             : dict from analysis.aggregate_stats().
    summary           : Plain text summary string from classifier.summarize_comments().
    themes            : List of (theme, count) from analysis.theme_counts().
    quotes            : List of quote dicts from pick_top_quotes().
    top_suggestions   : Up to 10 comment dicts where is_suggestion=True.
    bug_reports       : All comment dicts where is_bug_report=True.
    suggested_replies : Up to 5 comment dicts with non-empty suggested_reply.

    Returns
    -------
    Markdown string.
    """
    today = date.today().isoformat()
    title = video_info.get("title", "Unknown")
    channel = video_info.get("channel", "Unknown")
    views = f"{video_info.get('views', 0):,}"
    comment_count = f"{video_info.get('comment_count', 0):,}"

    score = stats.get("sentiment_score", 0.0)
    label = stats.get("sentiment_label", "Mixed")
    total = stats.get("total", 0)

    lines: list[str] = []

    # Header
    lines.append(f"# CommentLens Report: {title}")
    lines.append("")
    lines.append(f"**Channel:** {channel}")
    lines.append(f"**Views:** {views}")
    lines.append(f"**YouTube comment count:** {comment_count}")
    lines.append(f"**Date analyzed:** {today}")
    lines.append(f"**Comments analyzed:** {total:,}")
    lines.append("")

    # Sentiment score
    lines.append("## Sentiment Score")
    lines.append("")
    lines.append(f"**{score}/100** — {label}")
    lines.append("")

    # Palette counts table
    lines.append("## Comment Breakdown")
    lines.append("")
    lines.append("| Category | Count | % |")
    lines.append("| --- | --- | --- |")
    palette_rows = [
        ("Positive", "positive_count", "positive_count_pct"),
        ("Negative", "negative_count", "negative_count_pct"),
        ("Neutral", "neutral_count", "neutral_count_pct"),
        ("Mixed", "mixed_count", "mixed_count_pct"),
        ("Questions", "questions_count", "questions_count_pct"),
        ("Suggestions", "suggestions_count", "suggestions_count_pct"),
        ("Bug reports", "bug_reports_count", "bug_reports_count_pct"),
        ("Praise", "praise_count", "praise_count_pct"),
        ("Duplicates", "duplicates_count", "duplicates_count_pct"),
        ("Replies", "replies_count", "replies_count_pct"),
        ("Spam / bot", "spam_count", "spam_count_pct"),
        ("Rude or toxic", "toxic_count", "toxic_count_pct"),
        ("Unclassified", "unclassified_count", "unclassified_count_pct"),
    ]
    for label_str, count_key, pct_key in palette_rows:
        count = stats.get(count_key, 0)
        pct = stats.get(pct_key, 0.0)
        lines.append(f"| {label_str} | {count:,} | {pct}% |")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(summary or "_No summary available._")
    lines.append("")

    # Top themes
    lines.append("## Top Themes (Suggestions and Criticism)")
    lines.append("")
    if themes:
        for i, (theme, count) in enumerate(themes[:8], start=1):
            lines.append(f"{i}. {theme} ({count})")
    else:
        lines.append("_No themes detected._")
    lines.append("")

    # Top quotes
    lines.append("## Top Viewer Quotes")
    lines.append("")
    if quotes:
        for q in quotes[:5]:
            lines.append(f"> \"{q['quote']}\"")
            lines.append(f"> — {q['author']} ({q['likes']} likes, theme: {q['theme']})")
            lines.append("")
    else:
        lines.append("_No quotes available._")
        lines.append("")

    # Top suggestions
    lines.append("## Top Suggestions")
    lines.append("")
    if top_suggestions:
        for i, c in enumerate(top_suggestions[:10], start=1):
            text = c.get("original_text", "")[:300]
            author = c.get("author", "Anonymous")
            likes = c.get("likes", 0)
            lines.append(f"{i}. **{author}** ({likes} likes): {text}")
    else:
        lines.append("_No suggestions found._")
    lines.append("")

    # Bug reports
    lines.append("## Bug Reports")
    lines.append("")
    if bug_reports:
        for i, c in enumerate(bug_reports, start=1):
            text = c.get("original_text", "")[:400]
            author = c.get("author", "Anonymous")
            likes = c.get("likes", 0)
            lines.append(f"{i}. **{author}** ({likes} likes): {text}")
    else:
        lines.append("_No bug reports found._")
    lines.append("")

    # Suggested replies
    lines.append("## Suggested Replies")
    lines.append("")
    if suggested_replies:
        for i, c in enumerate(suggested_replies[:5], start=1):
            original = c.get("original_text", "")[:200]
            reply = c.get("suggested_reply", "")
            author = c.get("author", "Anonymous")
            lines.append(f"**{i}. Comment by {author}:**")
            lines.append(f"> {original}")
            lines.append("")
            lines.append(f"**Suggested reply:** {reply}")
            lines.append("")
    else:
        lines.append("_No suggested replies._")
        lines.append("")

    lines.append("---")
    lines.append(f"_Generated by CommentLens on {today}._")

    return "\n".join(lines)
