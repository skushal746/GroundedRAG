"""Provider-agnostic LLM interface."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        return self


@dataclass(slots=True)
class Completion:
    """Raw model output. For `complete_json` the text is a JSON document (unvalidated)."""

    text: str
    usage: Usage = field(default_factory=Usage)


class LLMError(RuntimeError):
    """Any provider failure: timeout, rate limit, auth, 5xx, refusal, empty output."""


class LLMClient(Protocol):
    """Implementations: OpenAIClient, AnthropicClient, and FakeLLM (tests / offline demo)."""

    model: str

    async def complete(
        self, system: str, messages: list[ChatMessage], *, max_tokens: int = 256
    ) -> Completion:
        """Free-text completion."""
        ...

    async def complete_json(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 800,
    ) -> Completion:
        """Completion constrained to `schema` (OpenAI structured outputs / Anthropic tool use).

        Callers must still validate the result: constrained decoding can be truncated,
        and not every model honours every schema keyword.
        """
        ...
