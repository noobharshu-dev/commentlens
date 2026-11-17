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
