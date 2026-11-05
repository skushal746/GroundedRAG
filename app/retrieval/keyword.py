"""Lexical retrieval: Postgres full-text search over the generated `tsv` column (GIN index).

The question is turned into an OR-query of its stemmed, stop-word-filtered lexemes, so
natural-language questions still match (plainto_tsquery ANDs every term and usually
returns nothing for a full sentence). `ts_rank_cd` then rewards chunks matching more terms
close together. This catches exact identifiers that embeddings blur: error codes,
plan names, header names like `Retry-After`.
"""

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.retrieval.types import Candidate
from app.retrieval.vector import candidate_from_chunk

_KEYWORD_SQL = text(
    """
    WITH q AS (
        SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS query
        FROM unnest(tsvector_to_array(to_tsvector('english', :q))) AS lexeme
    )
    SELECT c.id, ts_rank_cd(c.tsv, q.query, 32) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN q
    WHERE d.status = 'ready' AND c.tsv @@ q.query
    ORDER BY score DESC, c.id
    LIMIT :k
    """
)


async def keyword_search(db: AsyncSession, query: str, k: int) -> list[Candidate]:
    """Top-k chunks by full-text rank."""
    rows = (await db.execute(_KEYWORD_SQL, {"q": query, "k": k})).all()
    if not rows:
        return []
    scores = {int(chunk_id): float(score) for chunk_id, score in rows}
    stmt = (
        select(Chunk, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(scores))
    )
    by_id = {chunk.id: (chunk, filename) for chunk, filename in (await db.execute(stmt)).all()}
    out = []
    for rank, (chunk_id, score) in enumerate(scores.items(), start=1):
        chunk, filename = by_id[chunk_id]
        cand = candidate_from_chunk(chunk, filename)
        cand.keyword_score, cand.keyword_rank = score, rank
        out.append(cand)
    return out
