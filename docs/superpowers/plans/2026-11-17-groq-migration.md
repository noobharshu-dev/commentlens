# Groq Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate CommentLens's LLM engine to Groq (powered by ultra-fast Llama 3.3 70B & Llama 3.1 8B) as the primary, default provider to eliminate Gemini 404/rate-limit issues while keeping full backward compatibility.

**Architecture:** Introduce a unified, provider-agnostic LLM interface in `core/classifier.py` and `core/creator.py` that defaults to Groq using the official `groq` SDK (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`). Add robust JSON array extraction, Groq health diagnostics, and UI controls in `app.py` for API key input and model selection.

**Tech Stack:** Python 3.10+, `groq>=0.11.0`, `streamlit>=1.40.0`, `pytest`.

**Spec:** User request: "facing a lot of error bcz of gemini. we'll use groq. plan and update" + `DESIGN-BRIEF.md`.

## Global Constraints

- Default LLM provider must be Groq, with default model `llama-3.3-70b-versatile`.
- Supported Groq fallback chain: `["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]`.
- The classification schema (sentiment score, categories, flags, theme, language, priority, summary, suggested reply) must remain 100% identical and pass all `validate_results` constraints.
- UI styling must follow `DESIGN-BRIEF.md` (light theme only, clay aesthetic, no emoji in AI responses, tabular numbers).
- Streamlit Cloud secrets must be supported seamlessly via `GROQ_API_KEY` and `GROQ_MODEL`.

## Review Focus

1. **Non-standard JSON formatting from LLM:** Groq in JSON mode may return an object `{"comments": [...]}` or markdown-wrapped JSON ` ```json ... ``` ` instead of a bare list. The parser must robustly extract list items under any key or structure.
2. **Missing or invalid Groq API key:** UI and backend must detect missing/invalid Groq keys with clear, actionable error messages instead of crashing.
3. **Groq rate limits (HTTP 429):** Diagnostic engine must classify 429 as fatal/fast-fail so users are alerted immediately instead of waiting for retries.
4. **Creator Q&A citations:** Creator responses must maintain citation format `[c12]` extracted cleanly from Groq completion responses.
5. **Full test suite compatibility:** All 19 existing pytest tests and new Groq tests must pass with zero regression.

---

### Task 1: Add Groq Dependencies & Environment Defaults

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Test: `tests/test_groq_env.py`

**Interfaces:**
- Consumes: `os.environ`, `st.secrets`
- Produces: `GROQ_API_KEY: str`, `GROQ_MODEL: str` defaults

- [ ] **Step 1: Write failing test for Groq dependency and env defaults**

```python
# tests/test_groq_env.py
import os
import groq

def test_groq_sdk_available():
    assert groq.__version__ is not None

def test_groq_defaults():
    from core.classifier import DEFAULT_GROQ_MODEL, SUPPORTED_GROQ_MODELS
    assert DEFAULT_GROQ_MODEL == "llama-3.3-70b-versatile"
    assert "llama-3.1-8b-instant" in SUPPORTED_GROQ_MODELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_groq_env.py -v`
Expected: FAIL (cannot import `DEFAULT_GROQ_MODEL` from `core.classifier`)

- [ ] **Step 3: Update `requirements.txt` and `.env.example`**

Add `groq>=0.11.0` to `requirements.txt`.
Add `GROQ_API_KEY` and `GROQ_MODEL=llama-3.3-70b-versatile` to `.env.example`.

- [ ] **Step 4: Add Groq constants to `core/classifier.py`**

Define `DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"`, `SUPPORTED_GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_groq_env.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example core/classifier.py tests/test_groq_env.py
git commit -m "feat(groq): add groq dependency, constants and environment defaults"
```

---

### Task 2: Implement Robust JSON Array Parser & Groq Batch Classifier

**Files:**
- Modify: `core/classifier.py`
- Test: `tests/test_groq_classifier.py`

**Interfaces:**
- Consumes: `groq.Client`, comment batches
- Produces: `classify_batch_groq(batch, client, model_name, delay) -> list[dict]`, `extract_json_array(text: str) -> list[dict]`

- [ ] **Step 1: Write failing test for `extract_json_array`**

```python
# tests/test_groq_classifier.py
from core.classifier import extract_json_array

def test_extract_json_array_bare_list():
    res = extract_json_array('[{"comment_id": 1}]')
    assert len(res) == 1 and res[0]["comment_id"] == 1

def test_extract_json_array_wrapped_object():
    res = extract_json_array('{"comments": [{"comment_id": 2}]}')
    assert len(res) == 1 and res[0]["comment_id"] == 2

def test_extract_json_array_markdown_block():
    res = extract_json_array('```json\n[{"comment_id": 3}]\n```')
    assert len(res) == 1 and res[0]["comment_id"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_groq_classifier.py -v`
Expected: FAIL (`extract_json_array` not found)

- [ ] **Step 3: Implement `extract_json_array` and `classify_batch_groq` in `core/classifier.py`**

Implement `extract_json_array` handling regex stripping of markdown codeblocks and extracting lists from dict keys (`comments`, `data`, `results`).
Implement `classify_batch_groq` using `groq.Client.chat.completions.create(model=..., response_format={"type": "json_object"}, ...)` with fallback iteration across `SUPPORTED_GROQ_MODELS`.
Update `classify_batch` to route to `classify_batch_groq` when given a Groq client or Groq model name.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_groq_classifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/classifier.py tests/test_groq_classifier.py
git commit -m "feat(groq): implement resilient JSON array extraction and Groq batch classification"
```

---

### Task 3: Implement Groq Comment Summarization & Creator Q&A

**Files:**
- Modify: `core/classifier.py` (`summarize_comments_groq`)
- Modify: `core/creator.py` (`answer_question_groq`)
- Test: `tests/test_groq_creator.py`

**Interfaces:**
- Consumes: top comments, themes, questions, `groq.Client`
- Produces: `summarize_comments_groq(...) -> str`, `answer_question_groq(...) -> tuple[str, list[int]]`

- [ ] **Step 1: Write test for Groq prompt formatting and mock dispatch**

```python
# tests/test_groq_creator.py
from unittest.mock import MagicMock
from core.classifier import summarize_comments
from core.creator import answer_question

def test_answer_question_groq_format():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Viewers loved the pacing [c12]."
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    
    answer, citations = answer_question("What did viewers like?", [], [{"comment_id": 12, "original_text": "Good pace"}], mock_client, model_name="llama-3.3-70b-versatile", provider="groq")
    assert "Viewers loved" in answer
    assert 12 in citations
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_groq_creator.py -v`
Expected: FAIL (`provider` parameter / Groq dispatch not yet supported)

- [ ] **Step 3: Implement `summarize_comments_groq` and `answer_question_groq`**

Implement Groq chat completion call for summarization and Q&A with system instructions ensuring no emoji and proper citation `[c<id>]` generation.
Update `summarize_comments` and `answer_question` to seamlessly support both Groq and Gemini clients.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_groq_creator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/classifier.py core/creator.py tests/test_groq_creator.py
git commit -m "feat(groq): implement Groq summarization and creator Q&A answering"
```

---

### Task 4: Add Groq Diagnostics & Connection Health Checker

**Files:**
- Modify: `core/diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `groq.Client`
- Produces: `is_fatal_groq_error(exc) -> tuple[bool, str]`, `test_groq_connection(client, model_name) -> dict`

- [ ] **Step 1: Write test for `is_fatal_groq_error` and `test_groq_connection`**

```python
# in tests/test_diagnostics.py
from core.diagnostics import is_fatal_groq_error, test_groq_connection
from unittest.mock import MagicMock

def test_groq_fatal_errors():
    is_fatal, msg = is_fatal_groq_error(Exception("Invalid API Key provided"))
    assert is_fatal is True
    assert "Invalid Groq API Key" in msg

    is_fatal, msg = is_fatal_groq_error(Exception("Rate limit reached 429"))
    assert is_fatal is True
    assert "Rate Limit" in msg

def test_test_groq_connection_mock():
    client = MagicMock()
    msg = MagicMock()
    msg.content = "Pong"
    client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
    res = test_groq_connection(client, "llama-3.3-70b-versatile")
    assert res["status"] == "ok"
    assert "Pong" in res["details"]["reply"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostics.py -k "groq" -v`
Expected: FAIL (`is_fatal_groq_error` / `test_groq_connection` not defined)

- [ ] **Step 3: Implement Groq diagnostics in `core/diagnostics.py`**

Add `is_fatal_groq_error(exc: Exception) -> tuple[bool, str]`.
Add `test_groq_connection(client, model_name: str = "llama-3.3-70b-versatile") -> dict` measuring latency and logging to `session_diagnostics`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostics.py -k "groq" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/diagnostics.py tests/test_diagnostics.py
git commit -m "feat(groq): add Groq error classification and connection health diagnostics"
```

---

### Task 5: Update Streamlit UI in `app.py` with Groq as Default

**Files:**
- Modify: `app.py`
- Test: `tests/test_all.py`

**Interfaces:**
- Consumes: `groq.Groq`, `core/classifier.py`, `core/diagnostics.py`
- Produces: Updated Streamlit UI with Groq provider default, API credentials manager, and health check buttons

- [ ] **Step 1: Update client factory and session state in `app.py`**

Set default provider to `"groq"` and default model to `"llama-3.3-70b-versatile"`.
Add `_groq_client()` factory function reading `GROQ_API_KEY` from `os.environ` or `st.secrets`.

- [ ] **Step 2: Update sidebar settings**

Add Provider selectbox (`Groq` / `Gemini`, default `Groq`).
Update model selectbox dynamically based on selected provider (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`).
Add Groq API key input in the credentials expander.
Add "Test Groq Connection" in connection tester and diagnostics tab.

- [ ] **Step 3: Update analysis pipeline**

In analysis setup, initialize Groq client when provider is Groq.
Pass Groq client and model to `classify_batch` and `summarize_comments`.

- [ ] **Step 4: Run full test suite to verify no regressions**

Run: `python -m pytest tests/`
Expected: ALL PASS

- [ ] **Step 5: Commit and push**

```bash
git add app.py
git commit -m "feat(groq): switch Streamlit UI default provider to Groq with full model management"
git push origin main
```
