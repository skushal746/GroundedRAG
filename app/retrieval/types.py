"""Retrieval data types."""

import uuid
from dataclasses import dataclass, field


@dataclass(slots=True)
class Candidate:
    """A chunk moving through the retrieval funnel, with every stage's score/rank."""

    chunk_id: int
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str
    embed_text: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    vector_score: float | None = None
    vector_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    merged_chunk_ids: list[int] = field(default_factory=list)

    @property
    def page(self) -> int | None:
        return self.page_start

    @property
    def score(self) -> float:
        """Best available relevance score (rerank > rrf > vector)."""
        for value in (self.rerank_score, self.rrf_score, self.vector_score, self.keyword_score):
            if value is not None:
                return value
        return 0.0

    def log_entry(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "vector_score": _round(self.vector_score),
            "vector_rank": self.vector_rank,
            "keyword_score": _round(self.keyword_score),
            "keyword_rank": self.keyword_rank,
            "rrf_score": _round(self.rrf_score),
            "rerank_score": _round(self.rerank_score),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


@dataclass(slots=True)
class RetrievalResult:
    query: str
    candidates: list[Candidate]
    """Everything retrieved (union of vector + keyword), for logging."""
    final: list[Candidate]
    """Top-k after fusion/rerank and de-duplication: what the LLM sees."""
    top_rerank_score: float | None
    timings_ms: dict[str, float]
