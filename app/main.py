"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, documents, health, sessions
from app.components import Components, build_components
from app.config import Settings, get_settings
from app.db.session import check_database
from app.observability.logging import configure_logging
from app.observability.middleware import RequestContextMiddleware

log = logging.getLogger(__name__)

DESCRIPTION = """
Customer-support knowledge assistant that answers **only** from uploaded documents,
cites its sources, and returns a structured fallback when it can't answer reliably.

Authenticate with the `X-API-Key` header. Start with `POST /v1/documents`, then `POST /v1/chat`.
"""


def create_app(settings: Settings | None = None, components: Components | None = None) -> FastAPI:
    """Build the app. Pass `components` to inject fakes (tests); otherwise models are loaded
    once in the lifespan handler."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = components or build_components(settings)
        app.state.components = built
        db_ok = await check_database(built.engine)
        log.info(
            "startup",
            extra={
                "database_ok": db_ok,
                "llm_provider": settings.llm_provider,
                "llm_model": built.llm.model,
                "retrieval_mode": settings.retrieval_mode,
                "min_rerank_score": settings.min_rerank_score,
            },
        )
        if not settings.api_key_set:
            log.warning("no_api_keys_configured: every authenticated request will be rejected")
        yield
        if components is None:
            await built.engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "chat", "description": "Ask questions; give feedback on answers"},
            {"name": "documents", "description": "Manage the knowledge base"},
            {"name": "sessions", "description": "Conversation history"},
            {"name": "health", "description": "Liveness / readiness probes"},
        ],
    )
    app.add_middleware(RequestContextMiddleware)
    for module in (chat, documents, sessions, health):
        app.include_router(module.router)
    return app
