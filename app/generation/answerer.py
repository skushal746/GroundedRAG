"""Builds the numbered-source context, calls the LLM for structured output, validates it."""

import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    SOURCE_TEMPLATE,
    STANDALONE_LINE,
)
from app.llm.base import ChatMessage, LLMClient, LLMError, Usage
from app.retrieval.types import Candidate

log = logging.getLogger(__name__)


class AnswerOutput(BaseModel):
    """The JSON object the LLM must return."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Answer for the customer, with inline [S#] citations")
    citations: list[str] = Field(description="Ids of every source cited, e.g. ['S1','S3']")
    answerable: bool = Field(description="False if the sources do not contain the answer")
    confidence: Literal["high", "medium", "low"]


ANSWER_SCHEMA = AnswerOutput.model_json_schema()


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    candidate: Candidate


@dataclass(slots=True)
class GenerationOutcome:
    output: AnswerOutput | None
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    attempts: int = 0


def number_sources(candidates: list[Candidate]) -> list[Source]:
    """Assign S1..Sn in rank order."""
    return [Source(id=f"S{i}", candidate=c) for i, c in enumerate(candidates, start=1)]


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def render_sources(sources: list[Source]) -> str:
    """Render sources as tagged blocks. `<` in document text is escaped so a document
    cannot close its own <source> tag and smuggle in instructions."""
    blocks = []
    for src in sources:
        cand = src.candidate
        pages = ""
        if cand.page_start is not None:
            pages = (
                f' pages="{cand.page_start}"'
                if cand.page_start == cand.page_end
                else f' pages="{cand.page_start}-{cand.page_end}"'
            )
        section = f' section="{_escape(cand.section_path)}"' if cand.section_path else ""
        blocks.append(
            SOURCE_TEMPLATE.format(
                id=src.id,
                filename=_escape(cand.filename),
                section=section,
                pages=pages,
                text=cand.text.replace("<", "&lt;"),
            )
        )
    return "\n".join(blocks)


class Answerer:
    def __init__(self, llm: LLMClient, max_tokens: int = 800) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    async def generate(
        self, *, original: str, standalone: str, sources: list[Source]
    ) -> GenerationOutcome:
        """Call the LLM; retry once on invalid JSON; never raise."""
        prompt = ANSWER_USER_TEMPLATE.format(
            sources=render_sources(sources),
            original=original,
            standalone=STANDALONE_LINE.format(query=standalone) if standalone != original else "",
        )
        messages = [ChatMessage("user", prompt)]
        outcome = GenerationOutcome(output=None)
        for attempt in (1, 2):
            outcome.attempts = attempt
            try:
                completion = await self._llm.complete_json(
                    ANSWER_SYSTEM_PROMPT,
                    messages,
                    schema=ANSWER_SCHEMA,
                    schema_name="grounded_answer",
                    max_tokens=self._max_tokens,
                )
            except LLMError as exc:
                log.warning("llm_error", extra={"error": str(exc), "attempt": attempt})
                outcome.error = str(exc)
                return outcome
            outcome.usage += completion.usage
            try:
                outcome.output = AnswerOutput.model_validate_json(completion.text)
                outcome.error = None
                return outcome
            except ValidationError as exc:
                outcome.error = f"invalid structured output: {exc.error_count()} error(s)"
                log.warning(
                    "llm_invalid_json",
                    extra={"attempt": attempt, "raw": completion.text[:500]},
                )
        return outcome
