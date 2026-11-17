"""
core/diagnostics.py
-------------------
In-memory and persistent diagnostic logger for CommentLens.
Captures live events during YouTube fetch, batch classification, and summarization,
allowing users to see what happens during, between, and after API calls.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LogEvent:
    timestamp: str
    level: str  # "INFO", "SUCCESS", "WARNING", "ERROR"
    stage: str  # "FETCH", "CLEAN", "CLASSIFY", "SUMMARIZE", "CACHE", "HEALTH"
    message: str
    details: Optional[dict] = None
    duration_ms: Optional[float] = None


class DiagnosticsLogger:
    """Thread-safe, in-memory diagnostic session logger."""

    def __init__(self) -> None:
        self.events: list[LogEvent] = []
        self._listeners: list[Callable[[LogEvent], None]] = []

    def add_listener(self, listener: Callable[[LogEvent], None]) -> None:
        """Register a callback that receives every logged event in real-time."""
        self._listeners.append(listener)

    def log(
        self,
        level: str,
        stage: str,
        message: str,
        details: Optional[dict] = None,
        duration_ms: Optional[float] = None,
    ) -> LogEvent:
        """Record a diagnostic event and notify listeners."""
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        event = LogEvent(
            timestamp=now_str,
            level=level.upper(),
            stage=stage.upper(),
            message=message,
            details=details,
            duration_ms=duration_ms,
        )
        self.events.append(event)

        # Notify any live UI callbacks
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

        # Also emit to python logger
        log_msg = f"[{event.stage}] {event.message}"
        if event.level == "ERROR":
            logger.error(log_msg)
        elif event.level == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return event

    def info(self, stage: str, message: str, details: Optional[dict] = None, duration_ms: Optional[float] = None) -> LogEvent:
        return self.log("INFO", stage, message, details, duration_ms)

    def success(self, stage: str, message: str, details: Optional[dict] = None, duration_ms: Optional[float] = None) -> LogEvent:
        return self.log("SUCCESS", stage, message, details, duration_ms)

    def warning(self, stage: str, message: str, details: Optional[dict] = None, duration_ms: Optional[float] = None) -> LogEvent:
        return self.log("WARNING", stage, message, details, duration_ms)

    def error(self, stage: str, message: str, details: Optional[dict] = None, duration_ms: Optional[float] = None) -> LogEvent:
        return self.log("ERROR", stage, message, details, duration_ms)

    def clear(self) -> None:
        self.events.clear()

    def to_dict_list(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def to_formatted_text(self) -> str:
        """Format all events into an easily readable console log string."""
        lines = []
        for e in self.events:
            dur = f" ({e.duration_ms:.0f}ms)" if e.duration_ms is not None else ""
            lines.append(f"[{e.timestamp}] [{e.level:<7}] [{e.stage:<9}] {e.message}{dur}")
            if e.details:
                for k, v in e.details.items():
                    lines.append(f"    - {k}: {v}")
        return "\n".join(lines)


# Global singleton instance for current runtime session
session_diagnostics = DiagnosticsLogger()


def is_fatal_gemini_error(exc: Exception) -> tuple[bool, str]:
    """
    Check if an exception is an unrecoverable failure (e.g. quota exhausted or invalid key)
    that should fail-fast instead of wasting 10 minutes retrying doomed batches.

    Returns:
        (is_fatal, human_readable_reason)
    """
    err_str = str(exc)
    err_lower = err_str.lower()

    if "api key not valid" in err_lower or "api_key_invalid" in err_lower or "invalid api key" in err_lower or "keyinvalid" in err_lower:
        return True, "Invalid Gemini API Key. Please check the key in Settings."

    if "429" in err_str or "resource_exhausted" in err_lower or "quota exceeded" in err_lower:
        return True, "Gemini API Quota Exceeded (HTTP 429). Your API key has hit its requests/minute or daily free-tier limit."

    if "permission_denied" in err_lower or "403" in err_str:
        return True, "Permission Denied (HTTP 403). The Gemini API is not enabled for this project or key is restricted."

    return False, ""


def test_gemini_connection(client, model_name: str = "gemini-3.5-flash-lite") -> dict:
    """
    Test Gemini API connectivity with a minimal ping prompt.

    Returns dict with keys:
        status: "ok" | "error"
        latency_ms: float
        model: str
        message: str
        details: dict
    """
    clean_model = model_name.strip() if model_name else "gemini-3.5-flash-lite"
    if (
        clean_model in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro")
        or "2.5" in clean_model
        or "2.0" in clean_model
        or "1.5" in clean_model
    ):
        clean_model = "gemini-3.5-flash-lite"

    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=clean_model,
            contents="Ping. Reply with one word: Pong.",
        )
        latency = (time.time() - t0) * 1000
        reply = response.text.strip() if response and response.text else "No text"
        session_diagnostics.success(
            "HEALTH",
            f"Gemini connection OK ({clean_model}) in {latency:.0f}ms",
            {"reply": reply, "latency_ms": round(latency, 1)},
            duration_ms=latency,
        )
        return {
            "status": "ok",
            "latency_ms": round(latency, 1),
            "model": clean_model,
            "message": f"Successfully connected to Gemini API ({clean_model}) in {latency:.0f}ms.",
            "details": {"reply": reply},
        }
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        is_fatal, reason = is_fatal_gemini_error(exc)
        err_msg = reason if is_fatal else str(exc)
        session_diagnostics.error(
            "HEALTH",
            f"Gemini connection test failed ({clean_model}): {err_msg}",
            {"error": str(exc), "fatal": is_fatal},
            duration_ms=latency,
        )
        return {
            "status": "error",
            "latency_ms": round(latency, 1),
            "model": clean_model,
            "message": f"Gemini API test failed: {err_msg}",
            "details": {"error": str(exc), "is_fatal": is_fatal},
        }


def test_youtube_connection(api_key: Optional[str] = None) -> dict:
    """
    Test YouTube Data API v3 connectivity using a public video probe.

    Returns dict with keys:
        status: "ok" | "error"
        latency_ms: float
        message: str
        details: dict
    """
    from core.youtube import get_video_info
    t0 = time.time()
    try:
        # Probe known public video (Rick Astley / dQw4w9WgXcQ)
        info = get_video_info("dQw4w9WgXcQ", api_key=api_key)
        latency = (time.time() - t0) * 1000
        session_diagnostics.success(
            "HEALTH",
            f"YouTube API connection OK in {latency:.0f}ms",
            {"title": info.get("title")},
            duration_ms=latency,
        )
        return {
            "status": "ok",
            "latency_ms": round(latency, 1),
            "message": f"Successfully verified YouTube Data API v3 key in {latency:.0f}ms.",
            "details": info,
        }
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        session_diagnostics.error(
            "HEALTH",
            f"YouTube API connection test failed: {exc}",
            {"error": str(exc)},
            duration_ms=latency,
        )
        return {
            "status": "error",
            "latency_ms": round(latency, 1),
            "message": f"YouTube API test failed: {exc}",
            "details": {"error": str(exc)},
        }

