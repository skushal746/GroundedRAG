"""LLM client factory."""

from app.config import LLMProvider, Settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import ChatMessage, Completion, LLMClient, LLMError, Usage
from app.llm.openai_client import OpenAIClient

__all__ = ["ChatMessage", "Completion", "LLMClient", "LLMError", "Usage", "build_llm_client"]


def build_llm_client(settings: Settings) -> LLMClient:
    """Instantiate the client selected by LLM_PROVIDER / LLM_MODEL."""
    model = settings.resolved_llm_model
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        return AnthropicClient(
            model, settings.anthropic_api_key.get_secret_value(), timeout_s=settings.llm_timeout_s
        )
    if settings.llm_provider is LLMProvider.FAKE:
        from app.testing import FakeLLM

        return FakeLLM()
    return OpenAIClient(
        model,
        settings.openai_api_key.get_secret_value(),
        timeout_s=settings.llm_timeout_s,
        temperature=settings.llm_temperature,
    )
