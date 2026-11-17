"""
core/classifier.py
------------------
Gemini-powered comment classification utilities for CommentLens.

Functions
---------
build_prompt(batch)                                          -> tuple[str, str]
classify_batch(batch, client, model_name, delay)             -> list[dict] | None
validate_results(results, batch)                             -> list[dict]
summarize_comments(top_comments, theme_counts, client, model_name) -> str
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_DELAY_SECONDS: float = float(os.environ.get("GEMINI_DELAY_SECONDS", 4))

# ---------------------------------------------------------------------------
# Enums (string values must match what Gemini returns)
# ---------------------------------------------------------------------------

_SENTIMENT_ENUM = ["Positive", "Negative", "Neutral", "Mixed"]
_CATEGORY_ENUM = [
    "Praise",
    "Criticism",
    "Question",
    "Suggestion",
    "Bug report",
    "Rude or toxic",
    "Spam",
    "Off-topic",
    "Other",
]
_PRIORITY_ENUM = ["High", "Medium", "Low"]
_LANGUAGE_ENUM = ["English", "Hindi", "Bengali", "Hinglish", "Other"]

# ---------------------------------------------------------------------------
# Response schema (google.genai.types)
# ---------------------------------------------------------------------------

_COMMENT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "comment_id": types.Schema(type=types.Type.INTEGER),
        "primary_sentiment": types.Schema(
            type=types.Type.STRING,
            enum=_SENTIMENT_ENUM,
        ),
        "sentiment_score": types.Schema(type=types.Type.NUMBER),
        "primary_category": types.Schema(
            type=types.Type.STRING,
            enum=_CATEGORY_ENUM,
        ),
        "is_question": types.Schema(type=types.Type.BOOLEAN),
        "is_praise": types.Schema(type=types.Type.BOOLEAN),
        "is_criticism": types.Schema(type=types.Type.BOOLEAN),
        "is_suggestion": types.Schema(type=types.Type.BOOLEAN),
        "is_bug_report": types.Schema(type=types.Type.BOOLEAN),
        "is_rude_or_toxic": types.Schema(type=types.Type.BOOLEAN),
        "is_spam_or_bot": types.Schema(type=types.Type.BOOLEAN),
        "is_off_topic": types.Schema(type=types.Type.BOOLEAN),
        "theme": types.Schema(type=types.Type.STRING),
        "language": types.Schema(
            type=types.Type.STRING,
            enum=_LANGUAGE_ENUM,
        ),
        "priority": types.Schema(
            type=types.Type.STRING,
            enum=_PRIORITY_ENUM,
        ),
        "ai_summary": types.Schema(type=types.Type.STRING),
        "suggested_reply": types.Schema(type=types.Type.STRING),
    },
    required=[
        "comment_id",
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
    ],
)

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=_COMMENT_SCHEMA,
)

# ---------------------------------------------------------------------------
# System instruction (embedded JSON schema for clarity)
# ---------------------------------------------------------------------------

_SCHEMA_DESCRIPTION = """
JSON output schema (array of objects):
[
  {
    "comment_id":        <int>    — matches the [id] in the user prompt,
    "primary_sentiment": <"Positive"|"Negative"|"Neutral"|"Mixed">,
    "sentiment_score":   <float -1.0 to 1.0, where -1.0=most negative, 1.0=most positive>,
    "primary_category":  <"Praise"|"Criticism"|"Question"|"Suggestion"|"Bug report"|"Rude or toxic"|"Spam"|"Off-topic"|"Other">,
    "is_question":       <bool>,
    "is_praise":         <bool>,
    "is_criticism":      <bool>,
    "is_suggestion":     <bool>,
    "is_bug_report":     <bool>,
    "is_rude_or_toxic":  <bool>,
    "is_spam_or_bot":    <bool>,
    "is_off_topic":      <bool>,
    "theme":             <str — 1 to 3 words, e.g. "audio quality", "pacing", "part 2 request">,
    "language":          <"English"|"Hindi"|"Bengali"|"Hinglish"|"Other">,
    "priority":          <"High"|"Medium"|"Low">,
    "ai_summary":        <str — max 15 words plain sentence summarising the comment>,
    "suggested_reply":   <str — max 25 words friendly creator reply; completely empty string if spam, bot, or toxic>
  }
]
"""

_SYSTEM_INSTRUCTION = (
    "You are a careful reader of YouTube comments for a content creator. "
    "Classify only. Do not add commentary.\n\n"
    "Rules:\n"
    "- Classify by meaning. Sarcasm is Negative.\n"
    "- A comment can be a Question and Negative at the same time. Set every boolean flag that applies.\n"
    "- Priority High = bug reports, serious criticism, or questions from comments with 10 or more likes. Medium = constructive feedback. Low = generic chatter or praise.\n"
    "- Never invent facts about the video. Base classification on comment text only.\n"
    "- Suggested reply: friendly, max 25 words. Leave completely empty for spam, bot, or toxic comments.\n"
    "- Classify every comment in the list. Return exactly one JSON object per comment.\n"
    "- Use the comment_id value from the prompt; never invent IDs.\n"
    "- sentiment_score must be a float between -1.0 (most negative) and 1.0 (most positive).\n"
    "- theme should be a 1-3 word phrase, e.g. 'audio quality', 'pacing', 'part 2 request'.\n"
    "- Language: detect English, Hindi, Bengali, Hinglish, or Other.\n"
    "- Respond with a valid JSON array and nothing else.\n\n"
    + _SCHEMA_DESCRIPTION
)

# ---------------------------------------------------------------------------
# Safe defaults for missing classifications
# ---------------------------------------------------------------------------

_DEFAULT_RESULT: dict = {
    "primary_sentiment": "Neutral",
    "sentiment_score": 0.0,
    "primary_category": "Other",
    "is_question": False,
    "is_praise": False,
    "is_criticism": False,
    "is_suggestion": False,
    "is_bug_report": False,
    "is_rude_or_toxic": False,
    "is_spam_or_bot": False,
    "is_off_topic": False,
    "theme": "",
    "language": "Other",
    "priority": "Low",
    "ai_summary": "",
    "suggested_reply": "",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompt(batch: list[dict]) -> tuple[str, str]:
    """
    Build the system instruction and user prompt for a batch of comments.

    Returns
    -------
    (system_instruction, user_prompt) — both str.
    """
    lines = [
        f"[{c['comment_id']}] {c.get('original_text', '').strip()[:500]}"
        for c in batch
    ]
    user_prompt = "Classify the following YouTube comments:\n\n" + "\n".join(lines)
    return _SYSTEM_INSTRUCTION, user_prompt


def classify_batch(
    batch: list[dict],
    client: genai.Client,
    model_name: str = GEMINI_MODEL,
    delay: float = GEMINI_DELAY_SECONDS,
) -> Optional[list[dict]]:
    """
    Classify a batch of comments via the Gemini API.

    Retries up to 3 times with exponential backoff (2 s, 4 s, 8 s).
    Automatically falls back across supported Gemini models if a model is unavailable.

    Returns
    -------
    List of classification result dicts, or None on total failure.
    """
    system_instruction, user_prompt = build_prompt(batch)

    generation_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
        temperature=0.1,
    )

    models_to_try = [model_name]
    for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.5-flash-lite"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    backoff_seconds = [2, 4, 8]

    for model_candidate in models_to_try:
        for attempt, wait in enumerate(backoff_seconds, start=1):
            try:
                response = client.models.generate_content(
                    model=model_candidate,
                    contents=user_prompt,
                    config=generation_config,
                )
                raw_text = response.text.strip()
                results: list[dict] = json.loads(raw_text)
                if isinstance(results, list):
                    if delay > 0:
                        time.sleep(delay)
                    return results
                logger.warning("Gemini returned non-list JSON on attempt %d.", attempt)
            except json.JSONDecodeError as exc:
                logger.warning("JSON decode error on attempt %d: %s", attempt, exc)
            except Exception as exc:  # noqa: BLE001
                err_str = str(exc)
                if "404" in err_str or "NOT_FOUND" in err_str or "503" in err_str:
                    logger.warning("Model %s unavailable, trying next model: %s", model_candidate, exc)
                    break
                logger.warning("Gemini API error on attempt %d with %s: %s", attempt, model_candidate, exc)

            if attempt < len(backoff_seconds):
                time.sleep(wait)

    logger.error("classify_batch failed across all models.")
    return None


def validate_results(results: list[dict], batch: list[dict]) -> list[dict]:
    """
    Ensure every comment in *batch* has a matching result in *results*.

    - Builds a lookup {comment_id -> result}.
    - For any comment_id missing from results, inserts a default.
    - Normalises casing, bounds, priorities, and enforces safety rules.
    - Logs each fix applied. Never crashes.

    Returns
    -------
    List of result dicts aligned to *batch* order.
    """
    result_map: dict[int, dict] = {}
    for r in results:
        cid = r.get("comment_id")
        if cid is not None:
            try:
                result_map[int(cid)] = r
            except (ValueError, TypeError):
                continue

    validated: list[dict] = []
    for comment in batch:
        cid = comment["comment_id"]
        raw_res = result_map.get(cid)
        if raw_res:
            res = dict(_DEFAULT_RESULT)
            res["comment_id"] = cid

            # primary_sentiment (FR-12: Positive, Negative, Neutral, Mixed)
            raw_sent = str(raw_res.get("primary_sentiment", "Neutral")).strip().capitalize()
            res["primary_sentiment"] = raw_sent if raw_sent in _SENTIMENT_ENUM else "Neutral"

            # sentiment_score (FR-12: -1.0 to 1.0)
            try:
                raw_score = float(raw_res.get("sentiment_score", 0.0))
                res["sentiment_score"] = round(max(-1.0, min(1.0, raw_score)), 2)
            except (ValueError, TypeError):
                res["sentiment_score"] = 0.0

            # primary_category (FR-12: Praise, Criticism, Question, Suggestion, Bug report, Rude or toxic, Spam, Off-topic, Other)
            raw_cat = str(raw_res.get("primary_category", "Other")).strip()
            cat_match = next((c for c in _CATEGORY_ENUM if c.lower() == raw_cat.lower()), "Other")
            res["primary_category"] = cat_match

            # Boolean flags
            for b_flag in [
                "is_question", "is_praise", "is_criticism", "is_suggestion",
                "is_bug_report", "is_rude_or_toxic", "is_spam_or_bot", "is_off_topic"
            ]:
                res[b_flag] = bool(raw_res.get(b_flag, False))

            # theme: 1 to 3 words
            res["theme"] = str(raw_res.get("theme", "")).strip()[:50]

            # language: English, Hindi, Bengali, Hinglish, Other
            raw_lang = str(raw_res.get("language", "Other")).strip()
            lang_match = next((l for l in _LANGUAGE_ENUM if l.lower() == raw_lang.lower()), "Other")
            res["language"] = lang_match

            # priority: High, Medium, Low
            raw_pri = str(raw_res.get("priority", "Low")).strip().capitalize()
            res["priority"] = raw_pri if raw_pri in _PRIORITY_ENUM else "Low"

            # FR-13: Priority High = bug reports, serious criticism, or questions from comments with 10 or more likes
            likes = int(comment.get("likes", 0))
            if res["is_bug_report"] or (res["is_criticism"] and likes >= 10) or (res["is_question"] and likes >= 10):
                res["priority"] = "High"

            # ai_summary: max 15 words
            res["ai_summary"] = str(raw_res.get("ai_summary", "")).strip()

            # suggested_reply: max 25 words, friendly. Empty for spam and toxic (FR-12, FR-13)
            if res["is_rude_or_toxic"] or res["is_spam_or_bot"]:
                res["suggested_reply"] = ""
            else:
                res["suggested_reply"] = str(raw_res.get("suggested_reply", "")).strip()

            validated.append(res)
        else:
            logger.warning(
                "Missing classification for comment_id=%d; applying safe defaults.", cid
            )
            default = dict(_DEFAULT_RESULT)
            default["comment_id"] = cid
            validated.append(default)

    return validated


def summarize_comments(
    top_comments: list[dict],
    theme_counts: list[tuple[str, int]],
    client: genai.Client,
    model_name: str = GEMINI_MODEL,
) -> str:
    """
    Generate a 3-4 sentence plain-text summary of the comment section.

    Covers: what viewers liked, what they criticised, what they asked for.
    No hype words, no emoji.

    Parameters
    ----------
    top_comments  : A representative sample of classified comment dicts.
    theme_counts  : List of (theme, count) from analysis.theme_counts().
    client        : Initialised google.genai.Client.
    model_name    : Gemini model identifier.

    Returns
    -------
    Plain summary string (3-4 sentences).
    """
    # Build theme list string.
    theme_lines = "\n".join(
        f"  - {theme} ({count})" for theme, count in theme_counts[:8]
    )

    # Sample comments: up to 80 entries, prefer high-priority and high-likes.
    sample = sorted(
        top_comments,
        key=lambda c: (str(c.get("priority", "")).lower() == "high", c.get("likes", 0)),
        reverse=True,
    )[:80]

    comment_lines = "\n".join(
        f"[{c['comment_id']}] ({c.get('primary_sentiment','?')}) "
        f"{c.get('original_text','')[:200]}"
        for c in sample
    )

    prompt = (
        "You are summarizing YouTube comment section data for a content creator.\n"
        "Write 3 to 4 plain sentences. No emoji. No hype. No filler words like 'amazing' or 'fantastic'.\n"
        "Cover:\n"
        "1. What viewers generally liked or found valuable.\n"
        "2. What they criticised or found lacking.\n"
        "3. What they asked for (questions / suggestions).\n\n"
        f"Top themes (from suggestions and criticism):\n{theme_lines or '  (none)'}\n\n"
        f"Sample comments:\n{comment_lines}\n\n"
        "Summary:"
    )

    models_to_try = [model_name]
    for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.5-flash-lite"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    for model_candidate in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_candidate,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            return response.text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("summarize_comments failed with %s: %s", model_candidate, exc)

    return "Summary could not be generated."
