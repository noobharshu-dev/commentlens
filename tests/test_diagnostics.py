"""
Unit tests for core.diagnostics and error detection.
"""

import pytest
from core.diagnostics import DiagnosticsLogger, is_fatal_gemini_error


def test_diagnostics_logger_records_events():
    diag = DiagnosticsLogger()
    diag.info("FETCH", "Starting comment fetch", {"video_id": "test_123"})
    diag.success("CLEAN", "Cleaned 50 comments", duration_ms=12.5)
    diag.warning("CLASSIFY", "Retrying batch 1")
    diag.error("CLASSIFY", "Batch failed with 429", {"error": "quota exceeded"})

    events = diag.to_dict_list()
    assert len(events) == 4
    assert events[0]["level"] == "INFO"
    assert events[0]["stage"] == "FETCH"
    assert events[0]["details"]["video_id"] == "test_123"

    assert events[1]["level"] == "SUCCESS"
    assert events[1]["duration_ms"] == 12.5

    assert events[3]["level"] == "ERROR"

    text = diag.to_formatted_text()
    assert "[INFO   ] [FETCH    ] Starting comment fetch" in text
    assert "video_id: test_123" in text


def test_fatal_error_detection():
    # 429 Quota
    is_fatal, reason = is_fatal_gemini_error(Exception("429 ResourceExhausted: Quota exceeded"))
    assert is_fatal is True
    assert "Quota Exceeded" in reason

    # Invalid Key
    is_fatal, reason = is_fatal_gemini_error(Exception("API_KEY_INVALID: API key not valid"))
    assert is_fatal is True
    assert "Invalid Gemini API Key" in reason

    # Recoverable / network glitch
    is_fatal, reason = is_fatal_gemini_error(Exception("Temporary connection reset by peer"))
    assert is_fatal is False


def test_sanitize_model_name():
    from core.classifier import sanitize_model_name, SUPPORTED_GEMINI_MODELS, DEFAULT_GEMINI_MODEL

    assert DEFAULT_GEMINI_MODEL == "gemini-3.5-flash-lite"
    assert sanitize_model_name("gemini-2.5-flash") == "gemini-3.5-flash-lite"
    assert sanitize_model_name("gemini-2.0-flash") == "gemini-3.5-flash-lite"
    assert sanitize_model_name("gemini-1.5-flash") == "gemini-3.5-flash-lite"
    assert sanitize_model_name("") == "gemini-3.5-flash-lite"
    assert sanitize_model_name(None) == "gemini-3.5-flash-lite"
    assert sanitize_model_name("gemini-3.8-flash") == "gemini-3.8-flash"
    assert sanitize_model_name("gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"

    # Ensure no deprecated models in supported list
    for m in SUPPORTED_GEMINI_MODELS:
        assert "2.5" not in m
        assert "2.0" not in m
        assert "1.5" not in m
    assert SUPPORTED_GEMINI_MODELS[0] == "gemini-3.5-flash-lite"
