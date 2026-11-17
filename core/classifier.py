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

import asyncio
import json
import logging
import os
import re
import time
from typing import Callable, Optional, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.diagnostics import session_diagnostics, is_fatal_gemini_error, is_fatal_groq_error
from core.cleaning import make_batches
from core.key_pool import GroqKeyPool

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment defaults & Models
# ---------------------------------------------------------------------------

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
SUPPORTED_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
_ACTIVE_GROQ_MODEL: Optional[str] = None


def reset_active_groq_model() -> None:
    """Reset sticky Groq model memory so fresh runs start with the user's chosen default."""
    global _ACTIVE_GROQ_MODEL
    _ACTIVE_GROQ_MODEL = None
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
GROQ_DELAY_SECONDS: float = float(os.environ.get("GROQ_DELAY_SECONDS", 3))

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
SUPPORTED_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]


def sanitize_model_name(name: str | None) -> str:
    """
    Ensure deprecated or retired models (such as gemini-2.5-flash, gemini-2.0-flash,
    gemini-1.5-flash) are automatically replaced with active gemini-3.5-flash-lite.
    """
    if not name or not isinstance(name, str):
        return DEFAULT_GEMINI_MODEL
    clean = name.strip()
    if (
        clean in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro")
        or "2.5" in clean
        or "2.0" in clean
        or "1.5" in clean
    ):
        return DEFAULT_GEMINI_MODEL
    return clean


GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = sanitize_model_name(os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
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


def extract_json_array(raw_text: str) -> list[dict]:
    """
    Safely extract a JSON array of dicts from raw LLM text.
    Handles bare arrays, dictionary envelopes (e.g. {"comments": [...]}, {"data": [...]}),
    markdown-fenced code blocks (```json ... ```), trailing commas, and boundary noise.
    """
    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        clean_text = clean_text.strip()

    def _unpack(parsed_obj):
        if isinstance(parsed_obj, list):
            return [item for item in parsed_obj if isinstance(item, dict)]
        if isinstance(parsed_obj, dict):
            for key in ["comments", "data", "results", "classifications", "items"]:
                if key in parsed_obj and isinstance(parsed_obj[key], list):
                    return [item for item in parsed_obj[key] if isinstance(item, dict)]
            for val in parsed_obj.values():
                if isinstance(val, list):
                    return [item for item in val if isinstance(item, dict)]
        return None

    # Attempt 1: Direct JSON parse
    try:
        data = json.loads(clean_text)
        unpacked = _unpack(data)
        if unpacked is not None:
            return unpacked
    except Exception:
        pass

    # Attempt 2: Repair trailing commas and find outermost [ ... ] or { ... }
    repaired = re.sub(r",\s*([\]}])", r"\1", clean_text)
    match = re.search(r"(\[.*\]|\{.*\})", repaired, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            unpacked = _unpack(data)
            if unpacked is not None:
                return unpacked
        except Exception:
            pass

    raise ValueError(f"Could not extract JSON list from model response: {raw_text[:200]}")


def classify_batch_groq(
    batch: list[dict],
    client,
    model_name: str = DEFAULT_GROQ_MODEL,
    delay: float = GROQ_DELAY_SECONDS,
) -> list[dict] | None:
    """
    Classify a batch of comments using the Groq API.
    Uses OpenAI-compatible chat completions with response_format={"type": "json_object"}.
    Includes cross-model fallback across SUPPORTED_GROQ_MODELS when hitting rate limits (8K TPM)
    or temporary model unavailabilities.
    """
    if not batch:
        return []

    system_instruction, user_prompt = build_prompt(batch)
    full_system = (
        system_instruction
        + "\n\nCRITICAL INSTRUCTION: Return a JSON object with a single key 'comments' containing the array of objects matching the schema. Example: {\"comments\": [...]}. Do not include markdown or extra commentary."
    )

    global _ACTIVE_GROQ_MODEL
    models_to_try: list[str] = []
    if _ACTIVE_GROQ_MODEL and _ACTIVE_GROQ_MODEL in SUPPORTED_GROQ_MODELS:
        models_to_try.append(_ACTIVE_GROQ_MODEL)
    if model_name and model_name not in models_to_try:
        models_to_try.append(model_name)
    for m in SUPPORTED_GROQ_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    backoff_seconds = [3, 6, 12]
    rate_limit_hits = 0
    total_models_tried = 0

    for model_candidate in models_to_try:
        total_models_tried += 1
        for attempt, wait in enumerate(backoff_seconds, start=1):
            t0 = time.time()
            try:
                session_diagnostics.info(
                    "CLASSIFY",
                    f"Classifying batch of {len(batch)} comments using Groq {model_candidate} (attempt {attempt}/{len(backoff_seconds)})",
                )
                response = client.chat.completions.create(
                    model=model_candidate,
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                raw_text = (
                    response.choices[0].message.content.strip()
                    if response and response.choices and response.choices[0].message.content
                    else ""
                )
                results = extract_json_array(raw_text)
                if isinstance(results, list):
                    latency = (time.time() - t0) * 1000
                    session_diagnostics.success(
                        "CLASSIFY",
                        f"Batch of {len(results)} comments classified using Groq {model_candidate} in {latency:.0f}ms",
                        duration_ms=latency,
                    )
                    _ACTIVE_GROQ_MODEL = model_candidate
                    if delay > 0:
                        time.sleep(delay)
                    return results

                session_diagnostics.warning("CLASSIFY", f"Groq {model_candidate} returned invalid JSON array on attempt {attempt}")
            except Exception as exc:  # noqa: BLE001
                err_str = str(exc)
                err_lower = err_str.lower()
                is_fatal, reason = is_fatal_groq_error(exc)
                if is_fatal and "rate limit" not in reason.lower() and "429" not in err_str:
                    session_diagnostics.error(
                        "CLASSIFY",
                        f"Fatal Groq API error: {reason}",
                        details={"model": model_candidate, "error": err_str},
                    )
                    raise ValueError(reason) from exc

                # If 429 Rate Limit (e.g. 8K TPM hit) or model 404/not found:
                # Switch to next fallback model immediately to use its separate quota bucket
                if any(x in err_lower for x in ["429", "rate limit", "rate_limit", "tpm", "rpm", "404", "not_found", "not found"]):
                    if any(x in err_lower for x in ["429", "rate limit", "rate_limit", "tpm", "rpm"]):
                        rate_limit_hits += 1
                    session_diagnostics.warning(
                        "CLASSIFY",
                        f"Groq model {model_candidate} rate limit or availability issue ({exc}); switching to fallback model bucket...",
                    )
                    break

                logger.warning("Groq API error on attempt %d with %s: %s", attempt, model_candidate, exc)
                session_diagnostics.warning("CLASSIFY", f"Attempt {attempt} failed on Groq {model_candidate}: {exc}")

            if attempt < len(backoff_seconds):
                time.sleep(wait)

    if total_models_tried > 0 and rate_limit_hits >= total_models_tried:
        session_diagnostics.error(
            "CLASSIFY",
            f"Groq Rate Limit Exceeded (HTTP 429) across all {rate_limit_hits} fallback models. Stopping analysis.",
        )
        raise ValueError(
            "Groq Rate Limit Exceeded (HTTP 429). The free-tier TPM (8K) or RPM (30) limit was reached across all fallback models. "
            "Please wait 1 minute before trying again or reduce batch size."
        )

    session_diagnostics.error("CLASSIFY", f"Batch of {len(batch)} comments failed across all attempted Groq models.")
    return None


def classify_batch(
    batch: list[dict],
    client,
    model_name: str = DEFAULT_GROQ_MODEL,
    delay: float = GROQ_DELAY_SECONDS,
) -> Optional[list[dict]]:
    """
    Classify a batch of comments. Routes to Groq or Gemini based on client or model name.
    """
    if hasattr(client, "chat") or model_name in SUPPORTED_GROQ_MODELS or any(p in str(model_name) for p in ["openai/", "qwen/", "llama", "groq"]):
        return classify_batch_groq(batch, client, model_name, delay)
    system_instruction, user_prompt = build_prompt(batch)

    generation_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
        temperature=0.1,
    )

    if model_name in SUPPORTED_GROQ_MODELS or any(p in str(model_name) for p in ["openai/", "qwen/", "llama", "groq"]):
        active_model = DEFAULT_GEMINI_MODEL
    else:
        active_model = sanitize_model_name(model_name)
    models_to_try = [active_model]
    for fallback in SUPPORTED_GEMINI_MODELS:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    backoff_seconds = [2, 4, 8]

    for model_candidate in models_to_try:
        for attempt, wait in enumerate(backoff_seconds, start=1):
            t0 = time.time()
            try:
                session_diagnostics.info(
                    "CLASSIFY",
                    f"Classifying batch of {len(batch)} comments using {model_candidate} (attempt {attempt}/{len(backoff_seconds)})",
                )
                response = client.models.generate_content(
                    model=model_candidate,
                    contents=user_prompt,
                    config=generation_config,
                )
                raw_text = response.text.strip() if response and response.text else ""
                results: list[dict] = json.loads(raw_text)
                if isinstance(results, list):
                    latency = (time.time() - t0) * 1000
                    session_diagnostics.success(
                        "CLASSIFY",
                        f"Batch of {len(results)} comments classified using {model_candidate} in {latency:.0f}ms",
                        duration_ms=latency,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    return results
                logger.warning("Gemini returned non-list JSON on attempt %d.", attempt)
                session_diagnostics.warning("CLASSIFY", f"Gemini returned non-list JSON on attempt {attempt}")
            except json.JSONDecodeError as exc:
                logger.warning("JSON decode error on attempt %d: %s", attempt, exc)
                session_diagnostics.warning("CLASSIFY", f"JSON decode error on attempt {attempt}: {exc}")
            except Exception as exc:  # noqa: BLE001
                err_str = str(exc)
                err_lower = err_str.lower()
                is_fatal, reason = is_fatal_gemini_error(exc)
                if is_fatal:
                    session_diagnostics.error(
                        "CLASSIFY",
                        f"Fatal Gemini API error: {reason}",
                        details={"model": model_candidate, "error": err_str},
                    )
                    raise ValueError(reason) from exc

                if any(x in err_lower for x in ["404", "not_found", "not found", "no longer available", "503", "unsupported"]):
                    logger.warning("Model %s unavailable, trying next model: %s", model_candidate, exc)
                    session_diagnostics.warning(
                        "CLASSIFY",
                        f"Model {model_candidate} unavailable ({exc}); switching to fallback model...",
                    )
                    break

                logger.warning("Gemini API error on attempt %d with %s: %s", attempt, model_candidate, exc)
                session_diagnostics.warning(
                    "CLASSIFY",
                    f"Attempt {attempt} failed on {model_candidate}: {exc}",
                )

            if attempt < len(backoff_seconds):
                time.sleep(wait)

    session_diagnostics.error("CLASSIFY", f"Batch of {len(batch)} comments failed across all attempted models.")
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
            likes = int(comment.get("likes") or 0)
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


async def classify_batches_async(
    comments: list[dict],
    key_pool: GroqKeyPool,
    model_name: str = DEFAULT_GROQ_MODEL,
    batch_size: int = 20,
    on_batch_complete: Optional[Callable[[list[dict], dict], None]] = None,
) -> list[dict]:
    """
    Classify comments asynchronously using a pool of Groq API keys.
    Distributes batches across healthy keys, automatically recovers from HTTP 429
    by cooling rate-limited keys and requeueing batches, and emits detailed diagnostics.
    """
    if not comments:
        return []

    batches = list(make_batches(comments, size=batch_size))
    total_batches = len(batches)
    queue: asyncio.Queue[tuple[int, list[dict], int]] = asyncio.Queue()

    for idx, b in enumerate(batches):
        queue.put_nowait((idx, b, 0))  # (batch_idx, batch, attempt_count)

    system_instruction, _ = build_prompt([])
    full_system = (
        system_instruction
        + "\n\nCRITICAL INSTRUCTION: Return a JSON object with a single key 'comments' containing the array of objects matching the schema. Example: {\"comments\": [...]}. Do not include markdown or extra commentary."
    )

    stats_lock = asyncio.Lock()
    live_stats = {
        "pos": 0,
        "neu": 0,
        "mix": 0,
        "neg": 0,
        "ques": 0,
        "sugg": 0,
        "read": 0,
        "total": len(comments),
    }

    # Worker concurrency bounded by available keys in the pool (min 1, max 5)
    num_workers = min(max(1, key_pool.size), 5)

    async def worker(worker_id: int) -> None:
        while True:
            try:
                batch_idx, batch, attempts = await queue.get()
            except asyncio.CancelledError:
                break

            slot = None
            try:
                slot = key_pool.acquire_key(wait_if_cooling=False)
                if slot.is_cooling:
                    wait_time = min(slot.cooldown_remaining, 5.0)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

                client = slot.get_async_client()
                _, user_prompt = build_prompt(batch)

                session_diagnostics.info(
                    "CLASSIFY",
                    f"[Key {slot.index} {slot.masked_key}] Classifying batch {batch_idx + 1}/{total_batches} ({len(batch)} comments) via {model_name}",
                    details={"key_index": slot.index, "batch": batch_idx + 1, "model": model_name},
                )
                t0 = time.time()

                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )

                latency = (time.time() - t0) * 1000
                raw_text = (
                    response.choices[0].message.content.strip()
                    if response and response.choices and response.choices[0].message.content
                    else ""
                )

                results = extract_json_array(raw_text)
                if not isinstance(results, list):
                    raise ValueError(f"Model returned invalid non-list structure: {raw_text[:120]}")

                validated = validate_results(results, batch)
                result_map = {r["comment_id"]: r for r in validated}

                async with stats_lock:
                    for c in batch:
                        cid = c["comment_id"]
                        if cid in result_map:
                            c.update(result_map[cid])
                            c["classification_status"] = "ok"
                            s_val = c.get("primary_sentiment", "Neutral")
                            if s_val == "Positive":
                                live_stats["pos"] += 1
                            elif s_val == "Negative":
                                live_stats["neg"] += 1
                            elif s_val == "Mixed":
                                live_stats["mix"] += 1
                            else:
                                live_stats["neu"] += 1
                            if c.get("is_question"):
                                live_stats["ques"] += 1
                            if c.get("is_suggestion"):
                                live_stats["sugg"] += 1
                        else:
                            c["classification_status"] = "failed"
                    live_stats["read"] += len(batch)

                slot.record_success(latency)
                session_diagnostics.success(
                    "CLASSIFY",
                    f"[Key {slot.index}] Batch {batch_idx + 1}/{total_batches} classified in {latency:.0f}ms",
                    duration_ms=latency,
                    details={"key_index": slot.index, "batch": batch_idx + 1, "count": len(batch)},
                )

                if on_batch_complete:
                    try:
                        on_batch_complete(batch, dict(live_stats))
                    except Exception:
                        pass

                # Modest pacing between requests on the same key
                inter_delay = 2.0 / max(1, key_pool.size)
                if inter_delay > 0:
                    await asyncio.sleep(inter_delay)

                queue.task_done()

            except Exception as exc:
                err_str = str(exc)
                err_lower = err_str.lower()
                is_rate_limit = any(x in err_lower for x in ["429", "rate limit", "rate_limit", "tpm", "rpm"])

                if is_rate_limit and slot is not None:
                    key_pool.mark_rate_limited(slot.index, cooldown_seconds=60.0)
                    session_diagnostics.warning(
                        "KEY-POOL",
                        f"[Key {slot.index} {slot.masked_key}] Hit rate limit (429). Cooling for 60s. Auto-reassigning batch {batch_idx + 1}...",
                    )
                    await queue.put((batch_idx, batch, attempts + 1))
                    queue.task_done()
                    await asyncio.sleep(0.5)
                elif attempts < 3:
                    slot_label = f"[Key {slot.index}]" if slot else "[Worker]"
                    session_diagnostics.warning(
                        "CLASSIFY",
                        f"{slot_label} Attempt {attempts + 1} on batch {batch_idx + 1} failed: {exc}. Retrying...",
                    )
                    if slot:
                        slot.record_error()
                    await queue.put((batch_idx, batch, attempts + 1))
                    queue.task_done()
                    await asyncio.sleep(1.0)
                else:
                    session_diagnostics.error(
                        "CLASSIFY",
                        f"Batch {batch_idx + 1} failed after {attempts + 1} attempts across keys: {exc}",
                    )
                    if slot:
                        slot.record_error()
                    for c in batch:
                        c["classification_status"] = "failed"
                    async with stats_lock:
                        live_stats["read"] += len(batch)
                    if on_batch_complete:
                        try:
                            on_batch_complete(batch, dict(live_stats))
                        except Exception:
                            pass
                    queue.task_done()

    tasks = [asyncio.create_task(worker(i)) for i in range(num_workers)]
    await queue.join()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    return comments


def summarize_comments_groq(
    top_comments: list[dict],
    theme_counts: list[tuple[str, int]],
    client,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    Generate a 3-4 sentence plain-text summary of the comment section using Groq.
    """
    theme_lines = "\n".join(
        f"  - {theme} ({count})" for theme, count in theme_counts[:8]
    )

    sample = sorted(
        top_comments,
        key=lambda c: (str(c.get("priority", "")).lower() == "high", int(c.get("likes") or 0)),
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

    models_to_try = [model_name] if model_name else []
    for m in SUPPORTED_GROQ_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    session_diagnostics.info("SUMMARIZE", f"Starting summary generation via Groq with {len(sample)} sample comments...")
    for model_candidate in models_to_try:
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model_candidate,
                messages=[
                    {"role": "system", "content": "You are a concise analytical assistant for creators. Plain text only. No emoji."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            dur = (time.time() - t0) * 1000
            txt = response.choices[0].message.content.strip() if response and response.choices else "Summary empty"
            session_diagnostics.success("SUMMARIZE", f"Generated summary using Groq {model_candidate} in {dur:.0f}ms", duration_ms=dur)
            return txt
        except Exception as exc:  # noqa: BLE001
            dur = (time.time() - t0) * 1000
            session_diagnostics.warning("SUMMARIZE", f"Summary attempt failed on Groq {model_candidate} ({exc})", duration_ms=dur)

    session_diagnostics.error("SUMMARIZE", "All Groq model candidates failed to generate summary.")
    return "Summary could not be generated."


def summarize_comments(
    top_comments: list[dict],
    theme_counts: list[tuple[str, int]],
    client,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    Generate a 3-4 sentence plain-text summary of the comment section.
    Routes to Groq or Gemini based on client or model name.
    """
    if hasattr(client, "chat") or model_name in SUPPORTED_GROQ_MODELS or any(p in str(model_name) for p in ["openai/", "qwen/", "llama", "groq"]):
        return summarize_comments_groq(top_comments, theme_counts, client, model_name)
    # Build theme list string.
    theme_lines = "\n".join(
        f"  - {theme} ({count})" for theme, count in theme_counts[:8]
    )

    # Sample comments: up to 80 entries, prefer high-priority and high-likes.
    sample = sorted(
        top_comments,
        key=lambda c: (str(c.get("priority", "")).lower() == "high", int(c.get("likes") or 0)),
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

    if model_name in SUPPORTED_GROQ_MODELS or any(p in str(model_name) for p in ["openai/", "qwen/", "llama", "groq"]):
        active_model = DEFAULT_GEMINI_MODEL
    else:
        active_model = sanitize_model_name(model_name)
    models_to_try = [active_model]
    for fallback in SUPPORTED_GEMINI_MODELS:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    session_diagnostics.info("SUMMARIZE", f"Starting summary generation with {len(sample)} sample comments...")
    for model_candidate in models_to_try:
        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=model_candidate,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            dur = (time.time() - t0) * 1000
            txt = response.text.strip() if response and response.text else "Summary empty"
            session_diagnostics.success("SUMMARIZE", f"Generated summary using {model_candidate} in {dur:.0f}ms", duration_ms=dur)
            return txt
        except Exception as exc:  # noqa: BLE001
            dur = (time.time() - t0) * 1000
            session_diagnostics.warning("SUMMARIZE", f"Summary attempt failed on {model_candidate} ({exc})", duration_ms=dur)
            logger.warning("summarize_comments failed with %s: %s", model_candidate, exc)

    session_diagnostics.error("SUMMARIZE", "All model candidates failed to generate summary.")
    return "Summary could not be generated."
