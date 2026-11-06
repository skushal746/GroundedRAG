"""LLM client factory."""

from app.config import LLMProvider, Settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import ChatMessage, Completion, LLMClient, LLMError, Usage
from app.llm.openai_client import OpenAIClient

__all__ = ["ChatMessage", "Completion", "LLMClient", "LLMError", "Usage", "build_llm_client"]


def build_llm_client(settings: Settings) -> LLMClient:
    """Instantiate the client selected by LLM_PROVIDER / LLM_MODEL.

    If the selected cloud provider has no API key, the local Ollama server is used instead
    (see `Settings.effective_llm_provider`).
    """
    model = settings.resolved_llm_model
    provider = settings.effective_llm_provider
    if provider is LLMProvider.OLLAMA:
        # Ollama serves an OpenAI-compatible API (including JSON-schema structured outputs);
        # it ignores the API key, but the SDK requires a non-empty one.
        return OpenAIClient(
            model,
            "ollama",
            timeout_s=settings.ollama_timeout_s,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
            label="Ollama",
        )
    if provider is LLMProvider.ANTHROPIC:
        return AnthropicClient(
            model, settings.anthropic_api_key.get_secret_value(), timeout_s=settings.llm_timeout_s
        )
    if provider is LLMProvider.FAKE:
        from app.testing import FakeLLM

        return FakeLLM()
    return OpenAIClient(
        model,
        settings.openai_api_key.get_secret_value(),
        timeout_s=settings.llm_timeout_s,
        temperature=settings.llm_temperature,
    )
