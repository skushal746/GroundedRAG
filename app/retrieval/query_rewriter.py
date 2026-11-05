"""Rewrites follow-up messages ("what about the annual plan?") into standalone queries.

History is used *only* here, to resolve references for retrieval. It is never given to the
answering prompt, so answers can only come from retrieved sources.
"""

import logging
from dataclasses import dataclass, field

from app.generation.prompts import REWRITE_SYSTEM_PROMPT, REWRITE_USER_TEMPLATE
from app.llm.base import ChatMessage, LLMClient, LLMError, Usage

log = logging.getLogger(__name__)

_MAX_TURN_CHARS = 600


@dataclass(slots=True)
class Rewrite:
    query: str
    usage: Usage = field(default_factory=Usage)
    rewritten: bool = False


class QueryRewriter:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rewrite(self, message: str, history: list[ChatMessage]) -> Rewrite:
        """Return a standalone query; on any LLM failure, fall back to the raw message."""
        if not history:
            return Rewrite(query=message)
        transcript = "\n".join(
            f"{'Customer' if m.role == 'user' else 'Assistant'}: {m.content[:_MAX_TURN_CHARS]}"
            for m in history
        )
        prompt = REWRITE_USER_TEMPLATE.format(history=transcript, message=message)
        try:
            completion = await self._llm.complete(
                REWRITE_SYSTEM_PROMPT, [ChatMessage("user", prompt)], max_tokens=120
            )
        except LLMError as exc:
            log.warning("rewrite_failed", extra={"error": str(exc)})
            return Rewrite(query=message)
        lines = completion.text.strip().splitlines()
        query = (lines[0].strip().strip('"').strip() if lines else "") or message
        return Rewrite(query=query, usage=completion.usage, rewritten=query != message)
