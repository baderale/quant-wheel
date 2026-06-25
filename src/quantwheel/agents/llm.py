"""Provider-agnostic chat-model factory.

The council and weekly agent don't care *which* LLM they talk to — local or
cloud, Ollama or Anthropic or OpenAI. You pick the provider with environment
variables (or a `.env` file), and everything downstream just works. This is what
lets anyone clone the repo and run it on their own model.

Configure via env (or `.env`):

    QUANTWHEEL_PROVIDER   anthropic | ollama | openai | google   (default: anthropic)
    QUANTWHEEL_MODEL      model id; defaults per provider if unset
    QUANTWHEEL_TEMPERATURE   float (default 0.3)

    # Ollama (fully local — no API key):
    QUANTWHEEL_PROVIDER=ollama
    QUANTWHEEL_MODEL=llama3.2          # any tool/JSON-capable local model
    OLLAMA_BASE_URL=http://localhost:11434   (optional)

    # OpenAI or any OpenAI-compatible server (vLLM, LM Studio, etc.):
    QUANTWHEEL_PROVIDER=openai
    QUANTWHEEL_MODEL=gpt-4o-mini
    OPENAI_API_KEY=...
    QUANTWHEEL_BASE_URL=http://localhost:1234/v1   (optional, for local servers)

Each provider needs its LangChain integration installed — the factory raises a
clear "pip install …" message if it's missing. Note: the council uses
**structured output**, so a local model must support tool-calling or JSON-schema
output (e.g. llama3.1/3.2, qwen2.5; tiny models like gemma3:270m will not).
"""

from __future__ import annotations

import os
from typing import Optional

# Load .env once so QUANTWHEEL_* / API keys are available everywhere.
try:  # optional dependency, but it's in the core deps
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# Sensible default model per provider (override with QUANTWHEEL_MODEL).
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
}

_INSTALL_HINT = {
    "anthropic": "pip install langchain-anthropic",
    "ollama": "pip install langchain-ollama  (and run `ollama serve`)",
    "openai": "pip install langchain-openai",
    "google": "pip install langchain-google-genai",
}


def get_provider() -> str:
    """The configured provider (lower-cased), defaulting to ``anthropic``."""
    return os.getenv("QUANTWHEEL_PROVIDER", "anthropic").strip().lower()


def get_model(provider: Optional[str] = None) -> str:
    """The configured model id, or the provider's default."""
    provider = (provider or get_provider()).lower()
    return os.getenv("QUANTWHEEL_MODEL") or DEFAULT_MODELS.get(provider, "")


def _temperature(override: Optional[float]) -> float:
    if override is not None:
        return override
    try:
        return float(os.getenv("QUANTWHEEL_TEMPERATURE", "0.3"))
    except ValueError:
        return 0.3


def make_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Return a LangChain chat model for the configured (or given) provider.

    Raises ``ModuleNotFoundError`` with an install hint if the provider's
    integration package isn't installed.
    """
    provider = (provider or get_provider()).lower()
    model = model or get_model(provider)
    temp = _temperature(temperature)

    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, temperature=temp)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            base = os.getenv("OLLAMA_BASE_URL") or os.getenv("QUANTWHEEL_BASE_URL")
            kwargs = {"model": model, "temperature": temp}
            if base:
                kwargs["base_url"] = base
            return ChatOllama(**kwargs)

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            base = os.getenv("QUANTWHEEL_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            kwargs = {"model": model, "temperature": temp}
            if base:
                kwargs["base_url"] = base
            return ChatOpenAI(**kwargs)

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=model, temperature=temp)

    except ModuleNotFoundError as exc:  # missing integration package
        hint = _INSTALL_HINT.get(provider, f"pip install langchain-{provider}")
        raise ModuleNotFoundError(
            f"Provider {provider!r} needs an extra package. Try: {hint}"
        ) from exc

    # Anything else: defer to LangChain's universal initializer.
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, model_provider=provider, temperature=temp)


def llm_available() -> tuple[bool, str]:
    """Whether the configured provider is usable, plus a human-readable reason.

    Local providers (Ollama, or OpenAI-compatible with a base URL) need no API
    key; cloud providers need their key in the environment. Used by the dashboard
    to enable/disable the council button.
    """
    provider = get_provider()
    model = get_model(provider)
    if provider == "ollama":
        return True, f"Ollama · {model} (local, no API key needed)."
    if provider == "openai" and os.getenv("QUANTWHEEL_BASE_URL"):
        return True, f"OpenAI-compatible local server · {model}."
    key = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider)
    if key and not os.getenv(key):
        return False, f"Set {key} (or switch QUANTWHEEL_PROVIDER=ollama for local)."
    return True, f"{provider} · {model} ready."
