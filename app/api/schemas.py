"""Public request/response models (these define the OpenAPI contract)."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.generation.guardrails import FallbackReason

_EXAMPLE_DOC_ID = "6f1c2a1e-3b7d-4c55-9a40-2d1f8e6b9c10"
_EXAMPLE_SESSION_ID = "0b6f3c9e-8f21-4a7e-b1d2-5c3e9a7f4d21"


class DocumentOut(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": _EXAMPLE_DOC_ID,
                    "filename": "billing_and_refunds.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                    "title": "Billing & Refunds Policy",
                    "status": "ready",
                    "error": None,
                    "num_chunks": 14,
                    "num_pages": None,
                    "size_bytes": 39012,
                    "created_at": "2026-11-05T10:12:00Z",
                }
            ]
        },
    )

    id: uuid.UUID
    filename: str
    mime_type: str
    title: str | None
    status: Literal["processing", "ready", "failed"]
    error: str | None
    num_chunks: int
    num_pages: int | None
    size_bytes: int
    created_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"message": "How long do I have to request a refund on an annual plan?"},
                {"session_id": _EXAMPLE_SESSION_ID, "message": "What about monthly plans?"},
            ]
        }
    )

    session_id: uuid.UUID | None = Field(
        default=None, description="Omit to start a new conversation; reuse to continue one."
    )
    message: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    id: str = Field(description="Citation id used inline in the answer, e.g. S1")
    document_id: uuid.UUID
    filename: str
    page: int | None = Field(description="First page of the cited passage (PDF only)")
    page_end: int | None = None
    section: str | None = Field(description='Heading path, e.g. "Billing > Refunds"')
    snippet: str
    score: float = Field(description="Relevance score (cross-encoder probability in [0,1])")


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": _EXAMPLE_SESSION_ID,
                    "message_id": "9a3c2b1d-0e4f-4a5b-8c7d-6e5f4a3b2c1d",
                    "answer": "Annual plans can be refunded in full within 30 days of the "
                    "purchase or renewal date [S1].",
                    "answerable": True,
                    "confidence": "high",
                    "fallback_reason": None,
                    "sources": [
                        {
                            "id": "S1",
                            "document_id": _EXAMPLE_DOC_ID,
                            "filename": "billing_and_refunds.docx",
                            "page": None,
                            "page_end": None,
                            "section": "Refunds > Annual plans",
                            "snippet": "Annual subscriptions are eligible for a full refund...",
                            "score": 0.91,
                        }
                    ],
                    "latency_ms": 1234,
                }
            ]
        }
    )

    session_id: uuid.UUID
    message_id: uuid.UUID = Field(description="Id of the assistant message (use for feedback)")
    answer: str
    answerable: bool
    confidence: Literal["high", "medium", "low"]
    fallback_reason: FallbackReason | None = Field(
        description="Set whenever the fallback message was returned instead of an answer"
    )
    sources: list[SourceOut]
    latency_ms: int


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    meta: dict[str, Any]
    created_at: datetime


class SessionMessages(BaseModel):
    session_id: uuid.UUID
    messages: list[MessageOut]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"helpful": False, "comment": "Wrong refund window"}]}
    )

    helpful: bool
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackOut(BaseModel):
    id: int
    message_id: uuid.UUID
    helpful: bool


class HealthOut(BaseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, bool] = Field(default_factory=dict)
