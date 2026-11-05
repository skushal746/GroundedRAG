"""Document upload / listing / deletion."""

import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, select

from app.api.deps import DbDep, get_pipeline, get_settings, require_api_key
from app.api.schemas import DocumentList, DocumentOut
from app.config import Settings
from app.db.models import Document
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.types import UnsupportedFormatError

router = APIRouter(
    prefix="/v1/documents", tags=["documents"], dependencies=[Depends(require_api_key)]
)


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document (PDF, DOCX, MD, TXT, HTML)",
    responses={
        200: {"model": DocumentOut, "description": "Identical content already ingested (no-op)"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)
async def upload_document(
    file: UploadFile,
    background: BackgroundTasks,
    response: Response,
    db: DbDep,
    pipeline: IngestionPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> Document:
    """Accepts the file and ingests it in the background. Poll `GET /v1/documents/{id}`
    until `status` is `ready` (or `failed`, with `error` explaining why)."""
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")
    try:
        reg = await pipeline.register(
            db,
            filename=file.filename or "upload",
            content_type=file.content_type,
            content=content,
        )
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    if reg.needs_processing:
        background.add_task(pipeline.process, reg.document.id, content)
    else:
        response.status_code = status.HTTP_200_OK
    return reg.document


@router.get("", response_model=DocumentList, summary="List documents")
async def list_documents(db: DbDep, limit: int = 100, offset: int = 0) -> DocumentList:
    rows = (
        await db.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
    ).scalars()
    total = (await db.execute(select(func.count()).select_from(Document))).scalar_one()
    return DocumentList(items=[DocumentOut.model_validate(r) for r in rows], total=total)


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Get a document (poll for ingestion status)",
    responses={404: {"description": "Not found"}},
)
async def get_document(document_id: uuid.UUID, db: DbDep) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and all of its chunks",
    responses={404: {"description": "Not found"}},
)
async def delete_document(document_id: uuid.UUID, db: DbDep) -> None:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await db.delete(document)
    await db.commit()
