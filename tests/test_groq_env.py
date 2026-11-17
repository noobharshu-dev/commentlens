import groq
from core.classifier import DEFAULT_GROQ_MODEL, SUPPORTED_GROQ_MODELS


def test_groq_sdk_available():
    assert groq.__version__ is not None


def test_groq_defaults():
    assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-120b"
    assert "openai/gpt-oss-20b" in SUPPORTED_GROQ_MODELS
    assert "qwen/qwen3.8-27b" in SUPPORTED_GROQ_MODELS
