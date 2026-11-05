"""The retrieval funnel: vector (+ keyword) → RRF → cross-encoder rerank → de-dupe → top-k."""

import time

from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import RetrievalMode, Settings
from app.retrieval.dedupe import dedupe_candidates
from app.retrieval.embedder import Embedder
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.keyword import keyword_search
from app.retrieval.reranker import Reranker
from app.retrieval.types import Candidate, RetrievalResult
from app.retrieval.vector import vector_search


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        reranker: Reranker | None,
        settings: Settings,
        mode: RetrievalMode | None = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self.mode = mode or settings.retrieval_mode
        if self.mode is RetrievalMode.HYBRID_RERANK and reranker is None:
            raise ValueError("hybrid_rerank mode requires a reranker")

    async def retrieve(self, db: AsyncSession, query: str) -> RetrievalResult:
        """Run the configured funnel for `query`."""
        timings: dict[str, float] = {}
        start = time.perf_counter()
        vector = await run_in_threadpool(self.embedder.embed_query, query)
        dense = await vector_search(db, vector, self.settings.vector_top_k)
        if self.mode is RetrievalMode.VECTOR:
            candidates = dense
            ranked = dense
        else:
            lexical = await keyword_search(db, query, self.settings.keyword_top_k)
            ranked = reciprocal_rank_fusion(dense, lexical, k=self.settings.rrf_k)
            candidates = ranked
        timings["retrieve_ms"] = _ms(start)

        top_rerank: float | None = None
        if self.mode is RetrievalMode.HYBRID_RERANK and ranked:
            assert self.reranker is not None
            start = time.perf_counter()
            ranked = await self._rerank(query, ranked)
            top_rerank = ranked[0].rerank_score
            timings["rerank_ms"] = _ms(start)

        final = dedupe_candidates(ranked, limit=self.settings.final_top_k)
        return RetrievalResult(
            query=query,
            candidates=candidates,
            final=final,
            top_rerank_score=top_rerank,
            timings_ms=timings,
        )

    async def _rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        assert self.reranker is not None
        scores = await run_in_threadpool(
            self.reranker.score, query, [c.embed_text for c in candidates]
        )
        for cand, score in zip(candidates, scores, strict=True):
            cand.rerank_score = score
        return sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))
