"""Conversation history."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import DbDep, require_api_key
from app.api.schemas import MessageOut, SessionMessages
from app.db.models import ChatSession, Message

router = APIRouter(
    prefix="/v1/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)]
)


@router.get(
    "/{session_id}/messages",
    response_model=SessionMessages,
    summary="Conversation history",
    responses={404: {"description": "Session not found"}},
)
async def get_messages(session_id: uuid.UUID, db: DbDep) -> SessionMessages:
    if await db.get(ChatSession, session_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    rows = (
        await db.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
        )
    ).scalars()
    return SessionMessages(
        session_id=session_id, messages=[MessageOut.model_validate(m) for m in rows]
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session, its messages, logs and feedback",
    responses={404: {"description": "Session not found"}},
)
async def delete_session(session_id: uuid.UUID, db: DbDep) -> None:
    session = await db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    await db.delete(session)
    await db.commit()
