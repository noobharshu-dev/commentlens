# Multi-Key Groq Async Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an asynchronous multi-key pooling architecture for Groq in CommentLens that safely distributes batches across multiple API keys, multiplies RPM/TPM capacity, automatically recovers from rate limits, and surfaces real-time key usage in the website's live diagnostic logs.

**Architecture:** A `GroqKeyPool` manager handles multiple Groq API keys (from `GROQ_API_KEYS` / `GROQ_API_KEY` or the UI), tracking per-key cooldowns and request timestamps. An async classification engine runs parallel workers (bounded by available active keys) using `AsyncGroq`, sending smaller 20-comment batches to stay under Groq's 8K TPM limit per key. Comprehensive events are emitted to `session_diagnostics` so users can see which key processed each batch, latency, token usage, and automatic failovers in real time on the website.

**Tech Stack:** Python 3.10+, `groq>=0.11.0` (`AsyncGroq`), `asyncio`, `streamlit>=1.40.0`, `pytest`.

**Spec:** User requirement: "what if i use multiple api key like 4-5 from different accounts?? make a plan for it. make sure i can see everything in the logs on the website." + Groq Console quotas (30 RPM, 8K TPM per key).

## Global Constraints

- Must support both single key (`GROQ_API_KEY`) and multi-key (`GROQ_API_KEYS="gsk_1,gsk_2,..."`) transparently.
- Default batch size for Groq must be 20 comments to fit comfortably within the 8,000 TPM limit.
- API keys must always be masked in logs and UI (e.g., `Key 1 (gsk_...3a8f)`), never leaking full secrets.
- On HTTP 429, the offending key must automatically enter a 60-second cooldown without losing the batch, instantly rerouting the batch to the next available healthy key.
- Live event logs in the Streamlit UI and Diagnostics tab must explicitly display key rotation, batch dispatch, inference latency, and failovers.
- Backward compatibility: Gemini provider and single-key setups must remain 100% operational.

## Review Focus

1. **All keys in cooldown:** If all configured keys hit rate limits simultaneously, system must gracefully back off and log clear countdown status rather than crashing.
2. **Key parsing robustness:** Comma, newline, and space-separated keys, with or without surrounding quotes, must parse cleanly with invalid/truncated keys filtered out.
3. **Async event loop in Streamlit:** Running async tasks inside Streamlit must not block the script runner or raise `ScriptRunContext` warnings during progress callbacks.
4. **Zero dropped batches on 429:** When a key fails mid-analysis, its batch must be requeued and processed by another key.
5. **Masked diagnostics:** Full API keys must never appear in diagnostic logs, `.txt` exports, or UI traces.

---

### Task 1: Groq Key Pool & Client Manager

**Files:**
- Create: `core/key_pool.py`
- Test: `tests/test_key_pool.py`

**Interfaces:**
- Consumes: `os.environ`, `groq.AsyncGroq`, `core.diagnostics.session_diagnostics`
- Produces: `GroqKeyPool`, `KeySlot`, `parse_api_keys(raw_input: str) -> list[str]`

- [ ] **Step 1: Write failing tests for key parsing, rotation, and cooldowns**

```python
# tests/test_key_pool.py
import pytest
import time
from core.key_pool import parse_api_keys, GroqKeyPool, KeySlot

def test_parse_api_keys_formats():
    raw = "gsk_key11111111111111111111111111111111, gsk_key22222222222222222222222222222222\ngsk_key33333333333333333333333333333333"
    keys = parse_api_keys(raw)
    assert len(keys) == 3
    assert keys[0].startswith("gsk_key1")
    assert keys[1].startswith("gsk_key2")
    assert keys[2].startswith("gsk_key3")

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
    keys = ["gsk_testkey11111111111111111111111111111111", "gsk_testkey22222222222222222222222222222222"]
    pool = GroqKeyPool(keys)
    pool.mark_rate_limited(1, cooldown_seconds=10)
    # Next acquire must pick key 2 since key 1 is on cooldown
    picked = pool.acquire_key()
    assert picked.index == 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_key_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.key_pool'`

- [ ] **Step 3: Implement `core/key_pool.py`**

Implement `parse_api_keys(raw: str) -> list[str]` to split on commas, newlines, or semicolons, stripping whitespace and quotes, and filtering out invalid keys.
Implement `KeySlot` dataclass containing `index: int`, `key: str`, `masked_key: str`, `client: AsyncGroq`, `last_used_time: float`, `cooldown_until: float`, `requests_count: int`, `error_count: int`.
Implement `GroqKeyPool` with:
- `acquire_key() -> KeySlot`: picks the next healthy, non-cooling key in round-robin order.
- `mark_rate_limited(index: int, cooldown_seconds: float = 60)`: flags key as cooling, logs diagnostic warning via `session_diagnostics.warning("KEY-POOL", ...)`.
- `get_status_summary() -> list[dict]`: returns status dict per key for the UI.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_key_pool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/key_pool.py tests/test_key_pool.py
git commit -m "feat: add GroqKeyPool with round-robin rotation and cooldown tracking"
```

---

### Task 2: Asynchronous Multi-Key Classification Engine

**Files:**
- Modify: `core/classifier.py`
- Test: `tests/test_async_classifier.py`

**Interfaces:**
- Consumes: `core.key_pool.GroqKeyPool`, `core.classifier.build_prompt`, `extract_json_array`, `validate_results`
- Produces: `classify_batches_async(comments, key_pool, model_name, batch_size=20, on_batch_complete=None) -> list[dict]`

- [ ] **Step 1: Write test for async batch classification and failover**

```python
# tests/test_async_classifier.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.key_pool import GroqKeyPool
from core.classifier import classify_batches_async

@pytest.mark.asyncio
async def test_classify_batches_async_distributes_across_pool():
    fake_keys = ["gsk_key11111111111111111111111111111111", "gsk_key22222222222222222222222222222222"]
    pool = GroqKeyPool(fake_keys)
    
    # Mock responses
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = '{"comments": [{"comment_id": 1, "primary_sentiment": "Positive", "sentiment_score": 0.8, "primary_category": "Praise"}]}'

    for slot in pool.slots:
        slot.client = MagicMock()
        slot.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    comments = [{"comment_id": 1, "original_text": "Awesome video!"}]
    results = await classify_batches_async(comments, pool, model_name="openai/gpt-oss-120b", batch_size=20)
    
    assert len(results) == 1
    assert results[0]["primary_sentiment"] == "Positive"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_async_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'classify_batches_async'`

- [ ] **Step 3: Implement `classify_batches_async` in `core/classifier.py`**

- Use `AsyncGroq` client instances stored in `KeySlot`.
- Dispatch batches into an `asyncio.Queue`.
- Run worker coroutines (bounded by `min(len(pool.active_slots), 5)`).
- For each batch, acquire a ready key, call `slot.client.chat.completions.create(...)`, record latency and emit diagnostics:
  - `session_diagnostics.info("CLASSIFY", f"[Key {slot.index} {slot.masked_key}] Classifying batch {batch_num}/{total_batches} ({len(batch)} comments)...")`
  - `session_diagnostics.success("CLASSIFY", f"[Key {slot.index}] Batch {batch_num} classified in {latency:.0f}ms", duration_ms=latency)`
- If a 429 RateLimitError occurs:
  - Call `pool.mark_rate_limited(slot.index, 60)`
  - Log `session_diagnostics.warning("KEY-POOL", f"[Key {slot.index}] Rate limited (429). Cooling for 60s. Auto-reassigning batch {batch_num} to next key...")`
  - Put the batch back onto the queue for immediate reprocessing by a healthy key.
- Invoke optional `on_batch_complete(completed_batch, live_stats)` callback for real-time UI updates.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_async_classifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/classifier.py tests/test_async_classifier.py
git commit -m "feat: implement classify_batches_async with multi-key pooling and instant 429 failover"
```

---

### Task 3: Streamlit UI Integration & Real-Time Key Pool Diagnostics

**Files:**
- Modify: `app.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `core.key_pool.GroqKeyPool`, `core.classifier.classify_batches_async`
- Produces: Live multi-key status card in Sidebar, live batch-level key stream in Analyzing view, Key Pool Status in Diagnostics tab.

- [ ] **Step 1: Write test verifying diagnostic log event formatting includes key tags**

```python
# In tests/test_diagnostics.py
def test_diagnostics_records_key_pool_events():
    from core.diagnostics import session_diagnostics
    session_diagnostics.clear()
    session_diagnostics.info("KEY-POOL", "Key 1 (gsk_...3a8f) dispatched batch 1/10")
    events = session_diagnostics.to_dict_list()
    assert any("Key 1" in e["message"] for e in events)
```

- [ ] **Step 2: Run test to verify it passes or fails**

Run: `pytest tests/test_diagnostics.py -v`

- [ ] **Step 3: Update `app.py`**

1. **Sidebar API Settings:**
   - Update Groq key input to support single or multiple keys (e.g., textarea or text input for comma-separated keys).
   - Display active key count indicator:
     `✓ 4 Groq Keys Active (Pooled: 120 RPM / 32K TPM capacity)`.
2. **Analysis Pipeline Execution (`_run_analysis`):**
   - Initialize `pool = GroqKeyPool(keys)`.
   - Set batch size to 20 for Groq.
   - Run `classify_batches_async` via `asyncio.run(...)`.
   - Update `logs_placeholder` and mood bar in real time via the `on_batch_complete` callback so every key's progress appears instantly on screen.
3. **Diagnostics & Logs Tab (`_render_diagnostics_tab`):**
   - Add a dedicated **Key Pool Health Card** showing:
     - Key index and masked ID (e.g. `Key 1: gsk_...91ac`)
     - State (`Active` in green or `Cooling (45s left)` in amber)
     - Total requests processed and error count
   - Add a **Ping All Keys** button in the connection tester to test all configured keys simultaneously with one click.

- [ ] **Step 4: Run full test suite to verify zero regressions**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_diagnostics.py
git commit -m "feat: add multi-key UI controls, live stream diagnostics, and pool health monitor"
```
