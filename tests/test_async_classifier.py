"""
Unit tests for async multi-key batch classification in core/classifier.py.
Verifies concurrent worker dispatch, rate-limit 429 recovery, and progress callbacks.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.key_pool import GroqKeyPool
from core.classifier import classify_batches_async


@pytest.mark.asyncio
async def test_classify_batches_async_distributes_across_pool():
    fake_keys = [
        "gsk_key11111111111111111111111111111111",
        "gsk_key22222222222222222222222222222222",
    ]
    pool = GroqKeyPool(fake_keys)

    # Mock responses
    mock_resp1 = MagicMock()
    mock_resp1.choices = [MagicMock()]
    mock_resp1.choices[0].message.content = (
        '{"comments": [{"comment_id": 1, "primary_sentiment": "Positive", "sentiment_score": 0.8, "primary_category": "Praise"}]}'
    )

    mock_resp2 = MagicMock()
    mock_resp2.choices = [MagicMock()]
    mock_resp2.choices[0].message.content = (
        '{"comments": [{"comment_id": 2, "primary_sentiment": "Negative", "sentiment_score": -0.6, "primary_category": "Criticism"}]}'
    )

    for slot in pool.slots:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[mock_resp1, mock_resp2])
        slot._client_async = client

    comments = [
        {"comment_id": 1, "original_text": "Great video!"},
        {"comment_id": 2, "original_text": "Too long."},
    ]

    completed_batches = []

    def on_batch(batch, stats):
        completed_batches.append(batch)

    results = await classify_batches_async(
        comments=comments,
        key_pool=pool,
        model_name="openai/gpt-oss-120b",
        batch_size=1,
        on_batch_complete=on_batch,
    )

    assert len(results) == 2
    assert results[0]["classification_status"] == "ok"
    assert results[1]["classification_status"] == "ok"
    assert len(completed_batches) == 2


@pytest.mark.asyncio
async def test_classify_batches_async_recovers_from_429():
    fake_keys = [
        "gsk_key11111111111111111111111111111111",
        "gsk_key22222222222222222222222222222222",
    ]
    pool = GroqKeyPool(fake_keys)

    # Key 1 throws 429 RateLimitError
    client1 = MagicMock()
    client1.chat.completions.create = AsyncMock(
        side_effect=Exception("Error code 429: Rate limit reached. 8K TPM exceeded.")
    )
    pool.slots[0]._client_async = client1

    # Key 2 succeeds
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = (
        '{"comments": [{"comment_id": 10, "primary_sentiment": "Positive", "sentiment_score": 0.9, "primary_category": "Praise"}]}'
    )
    client2 = MagicMock()
    client2.chat.completions.create = AsyncMock(return_value=mock_resp)
    pool.slots[1]._client_async = client2

    comments = [{"comment_id": 10, "original_text": "Saved me so much time"}]

    results = await classify_batches_async(
        comments=comments,
        key_pool=pool,
        model_name="openai/gpt-oss-120b",
        batch_size=1,
    )

    assert len(results) == 1
    assert results[0]["classification_status"] == "ok"
    assert results[0]["primary_sentiment"] == "Positive"
    # Key 1 should be cooling
    assert pool.slots[0].is_cooling is True
    # Key 2 handled the request
    assert pool.slots[1].requests_count == 1
