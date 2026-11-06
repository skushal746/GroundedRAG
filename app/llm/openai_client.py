"""OpenAI chat-completions client with JSON-schema structured outputs.

Also used for Ollama, which serves the same API (pass `base_url`)."""

from typing import Any

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.llm.base import ChatMessage, Completion, LLMError, Usage


class OpenAIClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        timeout_s: float = 30.0,
        temperature: float | None = 0.0,
        base_url: str | None = None,
        label: str = "OpenAI",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._label = label
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=1
        )

    async def _create(
        self, system: str, messages: list[ChatMessage], max_tokens: int, **extra: Any
    ) -> Completion:
        payload: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system)
        ]
        for m in messages:
            payload.append(
                ChatCompletionUserMessageParam(role="user", content=m.content)
                if m.role == "user"
                else ChatCompletionAssistantMessageParam(role="assistant", content=m.content)
            )
        if self._temperature is not None:
            extra["temperature"] = self._temperature
        try:
            response = await self._client.chat.completions.create(
                model=self.model, messages=payload, max_completion_tokens=max_tokens, **extra
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"{self._label} request failed: {type(exc).__name__}: {exc}") from exc
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"Model refused: {choice.message.refusal}")
        text = choice.message.content or ""
        if not text.strip():
            raise LLMError(f"Empty completion (finish_reason={choice.finish_reason})")
        usage = Usage()
        if response.usage is not None:
            usage = Usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return Completion(text=text, usage=usage)

    async def complete(
        self, system: str, messages: list[ChatMessage], *, max_tokens: int = 256
    ) -> Completion:
        return await self._create(system, messages, max_tokens)

    async def complete_json(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 800,
    ) -> Completion:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }
        return await self._create(system, messages, max_tokens, response_format=response_format)
