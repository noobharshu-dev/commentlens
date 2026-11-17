"""
Integration test for multi-key configuration and estimate helpers.
"""

import os
from core.key_pool import parse_api_keys, GroqKeyPool


def test_multi_key_env_parsing(monkeypatch):
    monkeypatch.setenv(
        "GROQ_API_KEYS",
        "gsk_key11111111111111111111111111111111, gsk_key22222222222222222222222222222222",
    )
    keys = parse_api_keys(os.getenv("GROQ_API_KEYS"))
    assert len(keys) == 2
    pool = GroqKeyPool(keys)
    assert pool.size == 2
    assert pool.slots[0].masked_key.startswith("gsk_")
    assert pool.slots[1].masked_key.startswith("gsk_")
