"""One chat turn, end to end: history → rewrite → retrieve → gate → generate → validate →
persist (messages + query log). This is the only place the stages are wired together."""

import logging
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import ChatSession, Message, QueryLog
from app.generation.answerer import Answerer, Source, number_sources
from app.generation.guardrails import (
    Decision,
    FallbackReason,
    decide,
    fallback,
    retrieval_gate,
)
from app.generation.prompts import ANSWER_PROMPT_VERSION
from app.llm.base import ChatMessage, Usage
from app.observability.logging import request_id_var
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.retriever import Retriever
from app.retrieval.types import RetrievalResult

log = logging.getLogger(__name__)

SNIPPET_CHARS = 300


class SessionNotFoundError(LookupError):
    pass


@dataclass(slots=True)
class ChatTurn:
    session_id: uuid.UUID
    message_id: uuid.UUID
    decision: Decision
    cited_sources: list[Source]
    retrieval: RetrievalResult
    rewritten_query: str
    usage: Usage
    latency_ms: dict[str, float] = field(default_factory=dict)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class ChatService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        rewriter: QueryRewriter,
        answerer: Answerer,
        settings: Settings,
        llm_model: str,
    ) -> None:
        self._retriever = retriever
        self._rewriter = rewriter
        self._answerer = answerer
        self._settings = settings
        self._llm_model = llm_model

    async def handle(
        self, db: AsyncSession, session_id: uuid.UUID | None, message: str
    ) -> ChatTurn:
        """Answer one user message. LLM failures become fallbacks; this never raises for them."""
        t0 = time.perf_counter()
        latency: dict[str, float] = {}
        usage = Usage()
        session = await self._get_or_create_session(db, session_id)
        history = await self._history(db, session.id)

        start = time.perf_counter()
        rewrite = await self._rewriter.rewrite(message, history)
        usage += rewrite.usage
        latency["rewrite_ms"] = _ms(start)

        retrieval = await self._retriever.retrieve(db, rewrite.query)
        latency.update(retrieval.timings_ms)

        sources = number_sources(retrieval.final)
        reason = retrieval_gate(retrieval, self._retriever.mode, self._settings.min_rerank_score)
        if reason is not None:
            decision = fallback(self._settings.fallback_message, reason)
        else:
            start = time.perf_counter()
            outcome = await self._answerer.generate(
                original=message, standalone=rewrite.query, sources=sources
            )
            latency["generate_ms"] = _ms(start)
            usage += outcome.usage
            if outcome.output is None:
                decision = fallback(self._settings.fallback_message, FallbackReason.LLM_ERROR)
            else:
                decision = decide(
                    answer=outcome.output.answer,
                    citations=outcome.output.citations,
                    answerable=outcome.output.answerable,
                    confidence=outcome.output.confidence,
                    provided_ids={s.id for s in sources},
                    fallback_message=self._settings.fallback_message,
                )
        by_id = {s.id: s for s in sources}
        cited = [by_id[i] for i in decision.cited_ids]
        latency["total_ms"] = _ms(t0)

        assistant = await self._persist(
            db, session.id, message, rewrite.query, retrieval, decision, cited, usage, latency
        )
        log.info(
            "chat_turn",
            extra={
                "session_id": str(session.id),
                "rewritten": rewrite.rewritten,
                "fallback_reason": decision.fallback_reason,
                "top_rerank_score": retrieval.top_rerank_score,
                "cited": decision.cited_ids,
                "latency_ms": latency,
            },
        )
        return ChatTurn(
            session_id=session.id,
            message_id=assistant.id,
            decision=decision,
            cited_sources=cited,
            retrieval=retrieval,
            rewritten_query=rewrite.query,
            usage=usage,
            latency_ms=latency,
        )

    async def _get_or_create_session(
        self, db: AsyncSession, session_id: uuid.UUID | None
    ) -> ChatSession:
        if session_id is not None:
            session = await db.get(ChatSession, session_id)
            if session is None:
                raise SessionNotFoundError(str(session_id))
            return session
        session = ChatSession()
        db.add(session)
        await db.flush()
        return session

    async def _history(self, db: AsyncSession, session_id: uuid.UUID) -> list[ChatMessage]:
        limit = self._settings.history_turns * 2
        if limit <= 0:
            return []
        rows = (
            await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
        ).scalars()
        return [
            ChatMessage("user" if m.role == "user" else "assistant", m.content)
            for m in reversed(list(rows))
        ]

    async def _persist(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        message: str,
        rewritten: str,
        retrieval: RetrievalResult,
        decision: Decision,
        cited: list[Source],
        usage: Usage,
        latency: dict[str, float],
    ) -> Message:
        db.add(Message(session_id=session_id, role="user", content=message, meta={}))
        await db.flush()  # separate INSERTs so clock_timestamp() orders user before assistant
        assistant = Message(
            session_id=session_id,
            role="assistant",
            content=decision.answer,
            meta={
                "answerable": decision.answerable,
                "confidence": decision.confidence,
                "fallback_reason": decision.fallback_reason,
                "sources": [source_payload(s) for s in cited],
            },
        )
        db.add(assistant)
        await db.flush()
        db.add(
            QueryLog(
                session_id=session_id,
                message_id=assistant.id,
                request_id=request_id_var.get(),
                original_query=message,
                rewritten_query=rewritten,
                retrieval_mode=self._retriever.mode.value,
                candidates=[c.log_entry() for c in retrieval.candidates],
                final_chunk_ids=[[c.chunk_id, *c.merged_chunk_ids] for c in retrieval.final],
                cited_source_ids=decision.cited_ids,
                top_rerank_score=retrieval.top_rerank_score,
                answerable=decision.answerable,
                confidence=decision.confidence,
                fallback_reason=decision.fallback_reason,
                llm_model=self._llm_model,
                prompt_version=ANSWER_PROMPT_VERSION,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency,
            )
        )
        await db.commit()
        return assistant


def source_payload(source: Source) -> dict[str, object]:
    """Public representation of a cited source (stored on the message and returned by the API)."""
    cand = source.candidate
    snippet = cand.text if len(cand.text) <= SNIPPET_CHARS else cand.text[:SNIPPET_CHARS] + "…"
    return {
        "id": source.id,
        "document_id": str(cand.document_id),
        "filename": cand.filename,
        "page": cand.page_start,
        "page_end": cand.page_end,
        "section": cand.section_path,
        "snippet": snippet,
        "score": round(cand.score, 4),
    }
