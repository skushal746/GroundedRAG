"""Chat endpoint: the integration surface for the client's application."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import DbDep, get_chat_service, require_api_key
from app.api.schemas import ChatRequest, ChatResponse, FeedbackOut, FeedbackRequest, SourceOut
from app.db.models import Feedback, Message
from app.services.chat import ChatService, SessionNotFoundError, source_payload

router = APIRouter(prefix="/v1/chat", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question",
    responses={404: {"description": "session_id does not exist"}},
)
async def chat(
    body: ChatRequest, db: DbDep, service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Answers from the knowledge base with citations. When no reliable answer exists the
    response is the configured fallback message with `answerable=false` and a
    `fallback_reason`; LLM outages also produce a fallback (never a 5xx)."""
    try:
        turn = await service.handle(db, body.session_id, body.message)
    except SessionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from exc
    d = turn.decision
    return ChatResponse(
        session_id=turn.session_id,
        message_id=turn.message_id,
        answer=d.answer,
        answerable=d.answerable,
        confidence=d.confidence,
        fallback_reason=d.fallback_reason,
        sources=[SourceOut.model_validate(source_payload(s)) for s in turn.cited_sources],
        latency_ms=int(turn.latency_ms["total_ms"]),
    )


@router.post(
    "/{message_id}/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
    summary="Thumbs up / down on an answer",
    responses={404: {"description": "Assistant message not found"}},
)
async def feedback(message_id: uuid.UUID, body: FeedbackRequest, db: DbDep) -> FeedbackOut:
    message = await db.get(Message, message_id)
    if message is None or message.role != "assistant":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Assistant message not found")
    row = Feedback(message_id=message_id, helpful=body.helpful, comment=body.comment)
    db.add(row)
    await db.commit()
    return FeedbackOut(id=row.id, message_id=message_id, helpful=row.helpful)
