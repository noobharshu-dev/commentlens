"""
Unit tests for core/key_pool.py
Verifies parsing, round-robin rotation, cooldown tracking, and key masking.
"""

import time
import pytest
from core.key_pool import parse_api_keys, GroqKeyPool, KeySlot, mask_api_key


def test_parse_api_keys_formats():
    raw = (
        ' "gsk_key11111111111111111111111111111111", gsk_key22222222222222222222222222222222\n'
        "gsk_key33333333333333333333333333333333; 'gsk_key44444444444444444444444444444444' \n"
        "   invalid_key_too_short\n"
        "your_api_key_placeholder"
    )
    keys = parse_api_keys(raw)
    assert len(keys) == 4
    assert keys[0] == "gsk_key11111111111111111111111111111111"
    assert keys[1] == "gsk_key22222222222222222222222222222222"
    assert keys[2] == "gsk_key33333333333333333333333333333333"
    assert keys[3] == "gsk_key44444444444444444444444444444444"


def test_mask_api_key():
    assert mask_api_key("gsk_1234567890abcdef1234567890abcdef") == "gsk_...cdef"
    assert mask_api_key("") == "empty"
    assert mask_api_key("short") == "short"


def test_key_pool_round_robin():
    keys = [
        "gsk_testkey11111111111111111111111111111111",
        "gsk_testkey22222222222222222222222222222222",
    ]
    pool = GroqKeyPool(keys)
    assert pool.size == 2
    k1 = pool.acquire_key()
    assert k1.index == 1
    k2 = pool.acquire_key()
    assert k2.index == 2
    k3 = pool.acquire_key()
    assert k3.index == 1


def test_key_cooldown_on_rate_limit():
    keys = [
        "gsk_testkey11111111111111111111111111111111",
        "gsk_testkey22222222222222222222222222222222",
    ]
    pool = GroqKeyPool(keys)
    pool.mark_rate_limited(1, cooldown_seconds=10)
    # Next acquire must pick key 2 since key 1 is on cooldown
    picked = pool.acquire_key()
    assert picked.index == 2
    assert pool.slots[0].is_cooling is True
    assert pool.slots[1].is_cooling is False


def test_key_pool_status_summary():
    keys = ["gsk_testkey11111111111111111111111111111111"]
    pool = GroqKeyPool(keys)
    slot = pool.acquire_key()
    slot.record_success(latency_ms=250.0)
    summary = pool.get_status_summary()
    assert len(summary) == 1
    assert summary[0]["index"] == 1
    assert summary[0]["requests_count"] == 1
    assert summary[0]["status"] == "Active"
