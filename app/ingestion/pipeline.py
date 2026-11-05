"""parse → clean → chunk → embed → store. Idempotent by SHA-256 content hash."""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from app.db.models import Chunk, Document, DocumentStatus
from app.ingestion.chunker import Chunker
from app.ingestion.parsers import resolve_parser
from app.retrieval.embedder import Embedder

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Registration:
    document: Document
    created: bool
    """False when identical content was already ingested (the upload is a no-op)."""
    needs_processing: bool


class IngestionPipeline:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        chunker: Chunker,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._chunker = chunker

    async def register(
        self, db: AsyncSession, *, filename: str, content_type: str | None, content: bytes
    ) -> Registration:
        """Validate the format and create (or find) the document row.

        Raises UnsupportedFormatError for unknown types. Identical content returns the
        existing row; a previously *failed* upload of the same content is retried.
        """
        mime, _ = resolve_parser(filename, content_type)
        digest = hashlib.sha256(content).hexdigest()
        existing = (
            await db.execute(select(Document).where(Document.content_hash == digest))
        ).scalar_one_or_none()
        if existing is not None:
            if existing.status != DocumentStatus.FAILED:
                return Registration(existing, created=False, needs_processing=False)
            existing.status, existing.error, existing.filename = (
                DocumentStatus.PROCESSING,
                None,
                filename,
            )
            await db.commit()
            return Registration(existing, created=False, needs_processing=True)

        document = Document(
            filename=filename,
            mime_type=mime,
            content_hash=digest,
            size_bytes=len(content),
            status=DocumentStatus.PROCESSING,
        )
        db.add(document)
        await db.commit()
        return Registration(document, created=True, needs_processing=True)

    async def process(self, document_id: uuid.UUID, content: bytes) -> None:
        """Parse, chunk, embed and store. Never raises: failures are recorded on the row."""
        start = time.perf_counter()
        async with self._session_factory() as db:
            document = await db.get(Document, document_id)
            if document is None:
                log.error("ingest_missing_document", extra={"document_id": str(document_id)})
                return
            try:
                _, parser = resolve_parser(document.filename, document.mime_type)
                parsed = await run_in_threadpool(parser, content, document.filename)
                drafts = self._chunker.chunk(parsed)
                if not drafts:
                    raise ValueError("document produced no chunks")
                vectors = await run_in_threadpool(
                    self._embedder.embed_documents, [d.embed_text for d in drafts]
                )
                await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
                db.add_all(
                    Chunk(
                        document_id=document_id,
                        chunk_index=d.chunk_index,
                        text=d.text,
                        embed_text=d.embed_text,
                        section_path=d.section_path,
                        page_start=d.page_start,
                        page_end=d.page_end,
                        token_count=d.token_count,
                        embedding=vector,
                    )
                    for d, vector in zip(drafts, vectors, strict=True)
                )
                document.title = parsed.title[:512]
                document.num_pages = parsed.num_pages
                document.num_chunks = len(drafts)
                document.status = DocumentStatus.READY
                document.error = None
                await db.commit()
                log.info(
                    "ingest_ready",
                    extra={
                        "document_id": str(document_id),
                        "doc_filename": document.filename,
                        "chunks": len(drafts),
                        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    },
                )
            except Exception as exc:
                await db.rollback()
                await self._mark_failed(document_id, f"{type(exc).__name__}: {exc}")

    async def _mark_failed(self, document_id: uuid.UUID, error: str) -> None:
        log.warning("ingest_failed", extra={"document_id": str(document_id), "error": error})
        async with self._session_factory() as db:
            document = await db.get(Document, document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error = error[:2000]
                await db.commit()
