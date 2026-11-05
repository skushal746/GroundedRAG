"""Deterministic, offline stand-ins for the embedder, reranker and LLM.

Used by the test suite, and selectable at runtime (`EMBEDDING_PROVIDER=fake`,
`LLM_PROVIDER=fake`) to run the full API without model downloads or API keys.
"""

import hashlib
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.llm.base import ChatMessage, Completion, LLMError, Usage

_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = (
    "a an and are as at be by can do does for from how i in is it my of on or the to what "
    "when where which who why will with you your"
)
_STOP = frozenset(_STOP_WORDS.split())


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class HashingEmbedder:
    """Bag-of-words feature hashing: texts sharing words get similar vectors."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for term in _terms(text):
            h = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class OverlapReranker:
    """Fraction of query terms present in the passage, in [0, 1]."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_terms(query))
        if not q:
            return [0.0 for _ in passages]
        return [len(q & set(_terms(p))) / len(q) for p in passages]


@dataclass(slots=True)
class RecordedCall:
    kind: str
    system: str
    messages: list[ChatMessage]


_SOURCE = re.compile(r'<source id="(S\d+)"[^\n]*\n(.*?)\n</source>', re.DOTALL)
_LATEST = re.compile(r"Latest customer message: (.*)")


class FakeLLM:
    """Scripted LLM. Queued responses are returned first; otherwise it answers from S1
    (or reports unanswerable when there are no sources) and rewrites to the latest message."""

    def __init__(
        self,
        *,
        json_responses: list[str | Exception] | None = None,
        text_responses: list[str | Exception] | None = None,
        model: str = "fake-llm",
    ) -> None:
        self.model = model
        self._json: deque[str | Exception] = deque(json_responses or [])
        self._text: deque[str | Exception] = deque(text_responses or [])
        self.calls: list[RecordedCall] = []

    @staticmethod
    def _next(queue: deque[str | Exception]) -> str | None:
        if not queue:
            return None
        item = queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    async def complete(
        self, system: str, messages: list[ChatMessage], *, max_tokens: int = 256
    ) -> Completion:
        self.calls.append(RecordedCall("complete", system, messages))
        scripted = self._next(self._text)
        if scripted is not None:
            return Completion(scripted, Usage(10, 5))
        match = _LATEST.search(messages[-1].content)
        if match is None:
            raise LLMError("fake LLM: no latest message")
        return Completion(match.group(1).strip(), Usage(10, 5))

    async def complete_json(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 800,
    ) -> Completion:
        self.calls.append(RecordedCall("complete_json", system, messages))
        scripted = self._next(self._json)
        if scripted is not None:
            return Completion(scripted, Usage(100, 20))
        sources = _SOURCE.findall(messages[-1].content)
        if not sources:
            payload = {
                "answer": "I don't know.",
                "citations": [],
                "answerable": False,
                "confidence": "low",
            }
        else:
            source_id, text = sources[0]
            sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
            payload = {
                "answer": f"{sentence} [{source_id}]",
                "citations": [source_id],
                "answerable": True,
                "confidence": "high",
            }
        return Completion(json.dumps(payload), Usage(100, 20))
