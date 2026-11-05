"""Dense retrieval: pgvector cosine distance over the HNSW index."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStatus
from app.retrieval.types import Candidate


def candidate_from_chunk(chunk: Chunk, filename: str) -> Candidate:
    """Build a retrieval Candidate from an ORM chunk row."""
    return Candidate(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=filename,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        embed_text=chunk.embed_text,
        section_path=chunk.section_path,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
    )


async def vector_search(db: AsyncSession, query_vector: list[float], k: int) -> list[Candidate]:
    """Top-k chunks by cosine similarity (score = 1 - cosine distance)."""
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(Chunk, Document.filename, distance)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == DocumentStatus.READY)
        .order_by(distance)
        .limit(k)
    )
    rows = (await db.execute(stmt)).all()
    out = []
    for rank, (chunk, filename, dist) in enumerate(rows, start=1):
        cand = candidate_from_chunk(chunk, filename)
        cand.vector_score, cand.vector_rank = 1.0 - float(dist), rank
        out.append(cand)
    return out
