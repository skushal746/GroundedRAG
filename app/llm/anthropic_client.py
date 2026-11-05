"""Anthropic (Claude) Messages API client. Structured output is obtained by forcing a
single tool call whose `input_schema` is the target JSON schema."""

import json
from typing import Any

import anthropic
from anthropic.types import AssistantMessageParam, MessageParam, UserMessageParam

from app.llm.base import ChatMessage, Completion, LLMError, Usage


class AnthropicClient:
    def __init__(
        self, model: str, api_key: str, *, timeout_s: float = 30.0, client: Any | None = None
    ) -> None:
        self.model = model
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=1
        )

    async def _create(
        self, system: str, messages: list[ChatMessage], max_tokens: int, **extra: Any
    ) -> Any:
        params: list[MessageParam] = [
            UserMessageParam(role="user", content=m.content)
            if m.role == "user"
            else AssistantMessageParam(role="assistant", content=m.content)
            for m in messages
        ]
        try:
            return await self._client.messages.create(
                model=self.model,
                system=system,
                max_tokens=max_tokens,
                messages=params,
                **extra,
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic request failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _usage(response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(usage.input_tokens, usage.output_tokens)

    async def complete(
        self, system: str, messages: list[ChatMessage], *, max_tokens: int = 256
    ) -> Completion:
        response = await self._create(system, messages, max_tokens)
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise LLMError(f"Empty completion (stop_reason={response.stop_reason})")
        return Completion(text=text, usage=self._usage(response))

    async def complete_json(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 800,
    ) -> Completion:
        tool = {
            "name": schema_name,
            "description": "Return the final answer in this exact structure.",
            "input_schema": schema,
        }
        response = await self._create(
            system,
            messages,
            max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": schema_name},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == schema_name:
                return Completion(text=json.dumps(block.input), usage=self._usage(response))
        raise LLMError(f"No tool_use block in response (stop_reason={response.stop_reason})")
