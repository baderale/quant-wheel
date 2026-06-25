"""Provider-agnostic LLM factory tests (no network — models aren't invoked)."""

from __future__ import annotations

import pytest

from quantwheel.agents import llm


def test_default_provider_and_model(monkeypatch):
    monkeypatch.delenv("QUANTWHEEL_PROVIDER", raising=False)
    monkeypatch.delenv("QUANTWHEEL_MODEL", raising=False)
    assert llm.get_provider() == "anthropic"
    assert llm.get_model() == "claude-sonnet-4-6"


def test_env_selects_ollama(monkeypatch):
    monkeypatch.setenv("QUANTWHEEL_PROVIDER", "ollama")
    monkeypatch.delenv("QUANTWHEEL_MODEL", raising=False)
    assert llm.get_provider() == "ollama"
    assert llm.get_model() == "llama3.2"


def test_explicit_model_overrides_default(monkeypatch):
    monkeypatch.setenv("QUANTWHEEL_MODEL", "qwen2.5-coder:7b")
    assert llm.get_model("ollama") == "qwen2.5-coder:7b"


def test_make_chat_model_builds_ollama_without_network(monkeypatch):
    # Constructing the client must not require a running server.
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    model = llm.make_chat_model(provider="ollama", model="llama3.2")
    assert type(model).__name__ == "ChatOllama"


def test_missing_provider_package_raises_with_hint(monkeypatch):
    # google integration is not installed in the dev env -> helpful error.
    pytest.importorskip  # noqa: B018 — keep import side effects minimal
    try:
        import langchain_google_genai  # noqa: F401
        pytest.skip("google integration is installed; can't test the missing path")
    except ModuleNotFoundError:
        pass
    with pytest.raises(ModuleNotFoundError, match="pip install"):
        llm.make_chat_model(provider="google", model="gemini-1.5-flash")


def test_llm_available_ollama_needs_no_key(monkeypatch):
    monkeypatch.setenv("QUANTWHEEL_PROVIDER", "ollama")
    ok, msg = llm.llm_available()
    assert ok and "local" in msg.lower()


def test_llm_available_anthropic_requires_key(monkeypatch):
    monkeypatch.setenv("QUANTWHEEL_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, msg = llm.llm_available()
    assert not ok and "ANTHROPIC_API_KEY" in msg
