"""Initial schema: documents, chunks (pgvector + tsvector), sessions, messages, query logs,
feedback.

Revision ID: 0001
Revises:
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from app.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB()


def _ts(name: str = "created_at", fn: str = "now()") -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.text(fn), nullable=False)


def upgrade() -> None:
    dim = get_settings().embedding_dim
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False),
        sa.Column("title", sa.String(512)),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("num_chunks", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_pages", sa.Integer),
        _ts(),
        _ts("updated_at"),
    )
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"], unique=True)

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            UUID,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embed_text", sa.Text, nullable=False),
        sa.Column("section_path", sa.String(1024)),
        sa.Column("page_start", sa.Integer),
        sa.Column("page_end", sa.Integer),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(dim), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', embed_text)", persisted=True),
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv_gin", "chunks", ["tsv"], postgresql_using="gin")

    op.create_table("sessions", sa.Column("id", UUID, primary_key=True), _ts())

    op.create_table(
        "messages",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "session_id", UUID, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("meta", JSONB, nullable=False),
        _ts(fn="clock_timestamp()"),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    op.create_table(
        "query_logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "session_id", UUID, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "message_id",
            UUID,
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("request_id", sa.String(64)),
        sa.Column("original_query", sa.Text, nullable=False),
        sa.Column("rewritten_query", sa.Text, nullable=False),
        sa.Column("retrieval_mode", sa.String(32), nullable=False),
        sa.Column("candidates", JSONB, nullable=False),
        sa.Column("final_chunk_ids", JSONB, nullable=False),
        sa.Column("cited_source_ids", JSONB, nullable=False),
        sa.Column("top_rerank_score", sa.Float),
        sa.Column("answerable", sa.Boolean, nullable=False),
        sa.Column("confidence", sa.String(8), nullable=False),
        sa.Column("fallback_reason", sa.String(32)),
        sa.Column("llm_model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=False),
        sa.Column("completion_tokens", sa.Integer, nullable=False),
        sa.Column("latency_ms", JSONB, nullable=False),
        _ts(),
    )
    op.create_index("ix_query_logs_session_id", "query_logs", ["session_id"])
    op.create_index("ix_query_logs_fallback_reason", "query_logs", ["fallback_reason"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "message_id", UUID, sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("helpful", sa.Boolean, nullable=False),
        sa.Column("comment", sa.Text),
        _ts(),
    )
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"])


def downgrade() -> None:
    for table in ("feedback", "query_logs", "messages", "sessions", "chunks", "documents"):
        op.drop_table(table)
