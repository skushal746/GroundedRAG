"""FastAPI dependencies. Everything model- or network-backed is resolved here, so tests can
swap any of them via `app.dependency_overrides` or by passing fake Components."""

import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.components import Components
from app.config import Settings
from app.generation.answerer import Answerer
from app.ingestion.chunker import Chunker
from app.ingestion.pipeline import IngestionPipeline
from app.llm import LLMClient
from app.retrieval.embedder import Embedder
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.retriever import Retriever
from app.services.chat import ChatService

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_components(request: Request) -> Components:
    components: Components = request.app.state.components
    return components


ComponentsDep = Annotated[Components, Depends(get_components)]


def get_settings(components: ComponentsDep) -> Settings:
    return components.settings


async def get_db(components: ComponentsDep) -> AsyncIterator[AsyncSession]:
    async with components.session_factory() as session:
        yield session


def get_embedder(components: ComponentsDep) -> Embedder:
    return components.embedder


def get_llm(components: ComponentsDep) -> LLMClient:
    return components.llm


def get_retriever(
    components: ComponentsDep, embedder: Annotated[Embedder, Depends(get_embedder)]
) -> Retriever:
    return Retriever(embedder, components.reranker, components.settings)


def get_pipeline(
    components: ComponentsDep, embedder: Annotated[Embedder, Depends(get_embedder)]
) -> IngestionPipeline:
    settings = components.settings
    chunker = Chunker(settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    return IngestionPipeline(components.session_factory, embedder, chunker)


def get_chat_service(
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> ChatService:
    return ChatService(
        retriever=retriever,
        rewriter=QueryRewriter(llm),
        answerer=Answerer(llm, settings.llm_max_tokens),
        settings=settings,
        llm_model=llm.model,
    )


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    api_key: Annotated[str | None, Security(_api_key_header)],
) -> None:
    """Service-to-service auth via the `X-API-Key` header (constant-time comparison)."""
    valid = settings.api_key_set
    if not api_key or not any(secrets.compare_digest(api_key, k) for k in valid):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )


DbDep = Annotated[AsyncSession, Depends(get_db)]
