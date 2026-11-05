"""Confidence gating and the single fallback decision point.

Every path that can produce an unreliable answer ends here, so the API has exactly one
fallback shape with a machine-readable reason the client app can branch on (e.g. open
a support ticket on `low_retrieval_confidence`).
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from app.config import RetrievalMode
from app.generation.citations import validate_citations
from app.retrieval.types import RetrievalResult

Confidence = Literal["high", "medium", "low"]


class FallbackReason(StrEnum):
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
    MODEL_UNANSWERABLE = "model_unanswerable"
    INVALID_CITATIONS = "invalid_citations"
    LLM_ERROR = "llm_error"


@dataclass(slots=True)
class Decision:
    answer: str
    answerable: bool
    confidence: Confidence
    fallback_reason: FallbackReason | None
    cited_ids: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)


def retrieval_gate(
    result: RetrievalResult, mode: RetrievalMode, min_rerank_score: float
) -> FallbackReason | None:
    """Pre-generation gate: skip the LLM when retrieval found nothing trustworthy.

    Only the cross-encoder produces a calibrated query-passage relevance score; RRF and raw
    cosine scores are not comparable across queries, so in the ablation modes the gate only
    triggers on an empty result and the LLM's own `answerable` flag does the rest.
    """
    if not result.final:
        return FallbackReason.LOW_RETRIEVAL_CONFIDENCE
    if (
        mode is RetrievalMode.HYBRID_RERANK
        and result.top_rerank_score is not None
        and result.top_rerank_score < min_rerank_score
    ):
        return FallbackReason.LOW_RETRIEVAL_CONFIDENCE
    return None


def fallback(message: str, reason: FallbackReason) -> Decision:
    return Decision(answer=message, answerable=False, confidence="low", fallback_reason=reason)


def decide(
    *,
    answer: str,
    citations: list[str],
    answerable: bool,
    confidence: Confidence,
    provided_ids: set[str],
    fallback_message: str,
) -> Decision:
    """Post-generation gate: accept the model's answer only if it is grounded."""
    if not answerable:
        return fallback(fallback_message, FallbackReason.MODEL_UNANSWERABLE)
    check = validate_citations(answer, citations, provided_ids)
    if not check.valid:
        decision = fallback(fallback_message, FallbackReason.INVALID_CITATIONS)
        decision.invalid_citations = check.invalid
        return decision
    return Decision(
        answer=check.answer,
        answerable=True,
        confidence=confidence,
        fallback_reason=None,
        cited_ids=check.valid,
        invalid_citations=check.invalid,
    )
