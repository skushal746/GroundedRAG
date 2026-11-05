"""Reciprocal Rank Fusion (Cormack et al., 2009).

RRF only uses ranks, so it can combine cosine similarities and ts_rank scores that live on
incomparable scales without any normalisation or tuning: score(d) = sum_i 1 / (k + rank_i(d)).
"""

from app.retrieval.types import Candidate


def reciprocal_rank_fusion(*ranked_lists: list[Candidate], k: int = 60) -> list[Candidate]:
    """Fuse ranked lists into one, merging per-list scores onto a single Candidate per chunk."""
    fused: dict[int, Candidate] = {}
    totals: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, cand in enumerate(ranked, start=1):
            existing = fused.get(cand.chunk_id)
            if existing is None:
                fused[cand.chunk_id] = existing = cand
            else:
                existing.vector_score = existing.vector_score or cand.vector_score
                existing.vector_rank = existing.vector_rank or cand.vector_rank
                existing.keyword_score = existing.keyword_score or cand.keyword_score
                existing.keyword_rank = existing.keyword_rank or cand.keyword_rank
            totals[cand.chunk_id] = totals.get(cand.chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, cand in fused.items():
        cand.rrf_score = totals[chunk_id]
    return sorted(fused.values(), key=lambda c: (-totals[c.chunk_id], c.chunk_id))
