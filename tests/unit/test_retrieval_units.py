import uuid

import pytest

from app.retrieval.dedupe import dedupe_candidates, stitch
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.types import Candidate
from tests.conftest import make_candidate


def test_rrf_math_and_score_merging() -> None:
    a, b, c = (make_candidate(i) for i in (1, 2, 3))
    a.vector_score, a.vector_rank = 0.9, 1
    b.vector_score, b.vector_rank = 0.8, 2
    b_kw = make_candidate(2)
    b_kw.keyword_score, b_kw.keyword_rank = 0.5, 1
    c.keyword_score, c.keyword_rank = 0.4, 2

    fused = reciprocal_rank_fusion([a, b], [b_kw, c], k=60)

    assert [x.chunk_id for x in fused] == [2, 1, 3]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].rrf_score == pytest.approx(1 / 61)
    assert fused[2].rrf_score == pytest.approx(1 / 62)
    # Chunk 2 carries both its vector and keyword scores after fusion.
    assert (fused[0].vector_score, fused[0].keyword_score) == (0.8, 0.5)


def test_rrf_ties_are_deterministic() -> None:
    x, y = make_candidate(5), make_candidate(4)
    fused = reciprocal_rank_fusion([x], [y])
    assert [c.chunk_id for c in fused] == [4, 5]


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_candidate_score_precedence() -> None:
    c = make_candidate(1)
    assert c.score == 0.0
    c.keyword_score = 0.2
    assert c.score == 0.2
    c.vector_score = 0.5
    assert c.score == 0.5
    c.rrf_score = 0.03
    assert c.score == 0.03
    c.rerank_score = 0.9
    assert c.score == 0.9
    assert c.log_entry()["rerank_score"] == 0.9


def test_stitch_removes_overlap() -> None:
    first = "Alpha beta gamma. Delta epsilon zeta eta theta."
    second = "Delta epsilon zeta eta theta. Iota kappa."
    assert stitch(first, second) == "Alpha beta gamma. Delta epsilon zeta eta theta. Iota kappa."
    assert stitch("no overlap here at all", "different text") == (
        "no overlap here at all\n\ndifferent text"
    )


def _chunk(
    idx: int, text: str, doc: int = 1, section: str = "S", page: int | None = None
) -> Candidate:
    return make_candidate(
        100 + idx + doc * 1000,
        document_id=uuid.UUID(int=doc),
        chunk_index=idx,
        text=text,
        embed_text=f"Title\nSection: {section}\n\n{text}",
        section_path=section,
        page_start=page,
        page_end=page,
    )


def test_dedupe_merges_adjacent_chunks_and_drops_duplicates() -> None:
    overlap = "Shared overlapping sentence goes here."
    c1 = _chunk(1, f"First part is long enough. {overlap}", page=2)
    c2 = _chunk(2, f"{overlap} Second part.", page=3)
    c0 = _chunk(0, "Zero part. First part is long enough.", page=1)
    other_doc = _chunk(2, "Other document text.", doc=2)
    duplicate = _chunk(7, "Other document text.", doc=3)

    kept = dedupe_candidates([c1, other_doc, c2, duplicate, c0], limit=5)

    assert [k.chunk_id for k in kept] == [c1.chunk_id, other_doc.chunk_id]
    merged = kept[0]
    assert merged.text == f"Zero part. First part is long enough. {overlap} Second part."
    assert merged.embed_text == "Title\nSection: S\n\n" + merged.text
    assert merged.merged_chunk_ids == [c2.chunk_id, c0.chunk_id]
    assert (merged.page_start, merged.page_end) == (1, 3)


def test_dedupe_does_not_merge_across_sections_and_honours_limit() -> None:
    a = _chunk(1, "Section A text.", section="A")
    b = _chunk(2, "Section B text.", section="B")
    c = _chunk(5, "Far away chunk.", section="A")
    assert [k.chunk_id for k in dedupe_candidates([a, b, c], limit=2)] == [a.chunk_id, b.chunk_id]
