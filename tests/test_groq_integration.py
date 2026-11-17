"""
Unit tests for Groq integration: JSON extraction, error handling, batch classification, and creator Q&A.
"""

from unittest.mock import MagicMock
import pytest

from core.classifier import (
    DEFAULT_GROQ_MODEL,
    SUPPORTED_GROQ_MODELS,
    extract_json_array,
    classify_batch_groq,
    summarize_comments_groq,
)
from core.creator import answer_question_groq
from core.diagnostics import is_fatal_groq_error, test_groq_connection as check_groq_connection


def test_extract_json_array_formats():
    # 1. Bare list
    bare = '[{"comment_id": 0, "sentiment": "Positive"}]'
    assert extract_json_array(bare) == [{"comment_id": 0, "sentiment": "Positive"}]

    # 2. Dictionary envelope with "comments"
    wrapped = '{"comments": [{"comment_id": 1, "sentiment": "Negative"}]}'
    assert extract_json_array(wrapped) == [{"comment_id": 1, "sentiment": "Negative"}]

    # 3. Markdown fenced code block
    fenced = '```json\n[{"comment_id": 2, "sentiment": "Neutral"}]\n```'
    assert extract_json_array(fenced) == [{"comment_id": 2, "sentiment": "Neutral"}]

    # 4. Fenced dictionary
    fenced_dict = '```json\n{"data": [{"comment_id": 3, "sentiment": "Mixed"}]}\n```'
    assert extract_json_array(fenced_dict) == [{"comment_id": 3, "sentiment": "Mixed"}]

    # 5. Trailing commas (common in 20B/27B models)
    trailing_comma_list = '[{"comment_id": 4, "sentiment": "Positive"},]'
    assert extract_json_array(trailing_comma_list) == [{"comment_id": 4, "sentiment": "Positive"}]

    # 6. Conversational text wrapping JSON
    wrapped_text = 'Here is the classification:\n[{"comment_id": 5, "sentiment": "Neutral"}]\nDone!'
    assert extract_json_array(wrapped_text) == [{"comment_id": 5, "sentiment": "Neutral"}]


def test_is_fatal_groq_error():
    # Invalid key
    is_fatal, reason = is_fatal_groq_error(Exception("Error 401: Invalid API Key provided"))
    assert is_fatal is True
    assert "Invalid Groq API Key" in reason

    # Rate limit (TPM/RPM/429)
    is_fatal, reason = is_fatal_groq_error(Exception("Error 429: Rate limit reached for TPM"))
    assert is_fatal is True
    assert "Rate Limit" in reason

    # Transient error
    is_fatal, reason = is_fatal_groq_error(Exception("Connection reset by peer"))
    assert is_fatal is False


def test_test_groq_connection_mock():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Pong"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    res = check_groq_connection(mock_client, "openai/gpt-oss-120b")
    assert res["status"] == "ok"
    assert "openai/gpt-oss-120b" in res["model"]
    assert res["details"]["reply"] == "Pong"


def test_classify_batch_groq_mock():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"comments": [{"comment_id": 0, "primary_sentiment": "Positive"}]}'
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    batch = [{"comment_id": 0, "original_text": "Great video!"}]
    results = classify_batch_groq(batch, mock_client, model_name="openai/gpt-oss-120b", delay=0)
    assert results is not None
    assert len(results) == 1
    assert results[0]["comment_id"] == 0
    assert results[0]["primary_sentiment"] == "Positive"


def test_summarize_comments_groq_mock():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Viewers appreciated the detailed tutorial. Some noted audio was slightly low. Many requested a follow-up."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    top_comments = [{"comment_id": 0, "likes": 10, "original_text": "Nice video"}]
    summary = summarize_comments_groq(top_comments, [("audio", 2)], mock_client, model_name="openai/gpt-oss-120b")
    assert "Viewers appreciated" in summary


def test_answer_question_groq_mock():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Viewers requested more tutorials on audio editing [c12]."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    comments = [{"comment_id": 12, "original_text": "Need audio editing tutorial"}]
    answer, citations = answer_question_groq(
        "What did viewers ask for?",
        [],
        comments,
        mock_client,
        model_name="openai/gpt-oss-120b",
    )
    assert "audio editing" in answer
    assert 12 in citations


def test_answer_question_groq_role_normalization_and_keywords():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Answer citing [c99]."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    history = [
        {"role": "user", "text": "Hello"},
        {"role": "model", "text": "Hi there! How can I help?"},
    ]
    comments = [
        {"comment_id": 99, "original_text": "Battery life drains so fast", "likes": 0},
        {"comment_id": 100, "original_text": "Completely unrelated text", "likes": 500},
    ]

    answer, citations = answer_question_groq(
        "What about the battery drain?",
        history,
        comments,
        mock_client,
        model_name="openai/gpt-oss-120b",
    )

    assert 99 in citations
    # Check messages passed to client
    call_args = mock_client.chat.completions.create.call_args[1]
    msgs = call_args["messages"]

    # History turns must have role normalized from 'model' to 'assistant'
    roles = [m["role"] for m in msgs]
    assert "model" not in roles
    assert "assistant" in roles
    assert any(m["role"] == "assistant" and m["content"] == "Hi there! How can I help?" for m in msgs)

    # Keyword-matching comment c99 should be present in user prompt context
    user_msg = next(m for m in msgs if "Comment data:" in m["content"])
    assert "[c99]" in user_msg["content"]


def test_validate_results_none_likes():
    from core.classifier import validate_results
    from core.analysis import aggregate_stats, filter_palette

    raw_results = [
        {"comment_id": 0, "sentiment": "Positive", "category": "Praise", "priority": "Low"}
    ]
    # likes is None, should not raise TypeError
    batch = [{"comment_id": 0, "original_text": "Good", "likes": None}]
    validated = validate_results(raw_results, batch)
    assert len(validated) == 1
    assert validated[0]["comment_id"] == 0
    assert validated[0]["priority"] == "Low"

    # Also test aggregate_stats and filter_palette with likes=None
    stats = aggregate_stats(batch)
    assert stats["top_liked_count"] == 0
    top_liked = filter_palette(batch, "Top liked")
    assert len(top_liked) == 1


def test_classify_batch_groq_rate_limit_fail_fast():
    mock_client = MagicMock()
    # Mock chat.completions.create raising 429 Rate Limit for all models
    mock_client.chat.completions.create.side_effect = Exception("HTTP 429: Rate limit reached for TPM on model")

    batch = [{"comment_id": 0, "original_text": "Test"}]
    with pytest.raises(ValueError) as exc_info:
        classify_batch_groq(batch, mock_client, model_name="openai/gpt-oss-120b", delay=0)

    assert "Groq Rate Limit Exceeded" in str(exc_info.value)
    assert "429" in str(exc_info.value)


def test_classify_batch_groq_active_model_persistence():
    import core.classifier as clf

    mock_client = MagicMock()
    # Model 1 fails with 429, Model 2 succeeds
    mock_choice = MagicMock()
    mock_choice.message.content = '[{"comment_id": 0, "sentiment": "Positive"}]'

    def side_effect(*args, **kwargs):
        if kwargs.get("model") == "openai/gpt-oss-120b":
            raise Exception("HTTP 429: TPM exceeded")
        return MagicMock(choices=[mock_choice])

    mock_client.chat.completions.create.side_effect = side_effect

    clf._ACTIVE_GROQ_MODEL = None
    batch = [{"comment_id": 0, "original_text": "Test"}]
    res = clf.classify_batch_groq(batch, mock_client, model_name="openai/gpt-oss-120b", delay=0)

    assert res is not None
    # _ACTIVE_GROQ_MODEL should now be the fallback model that succeeded (openai/gpt-oss-20b)
    assert clf._ACTIVE_GROQ_MODEL == "openai/gpt-oss-20b"

