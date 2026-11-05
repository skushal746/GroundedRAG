"""Shared fixtures. Unit tests are offline; integration tests start a pgvector container."""

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from app.config import EmbeddingProvider, LLMProvider, Settings
from app.retrieval.types import Candidate

# Never let tests pick up a developer's real .env (API keys, DB URL).
os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "api_keys": "test-key",
        "llm_provider": LLMProvider.FAKE,
        "embedding_provider": EmbeddingProvider.FAKE,
        "min_rerank_score": 0.3,
        "history_turns": 3,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


def make_candidate(chunk_id: int, **kw: object) -> Candidate:
    defaults: dict[str, object] = {
        "chunk_id": chunk_id,
        "document_id": kw.pop("document_id", uuid.UUID(int=1)),
        "filename": "doc.md",
        "chunk_index": chunk_id,
        "text": f"text {chunk_id}",
        "embed_text": f"Doc\n\ntext {chunk_id}",
        "section_path": "Section",
        "page_start": None,
        "page_end": None,
    }
    defaults.update(kw)
    return Candidate(**defaults)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- integration


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """Start pgvector/pgvector:pg16 once per session and apply Alembic migrations."""
    testcontainers = pytest.importorskip("testcontainers.postgres")
    try:
        container = testcontainers.PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg")
        container.start()
    except Exception as exc:  # Docker not available
        pytest.skip(f"Docker unavailable for integration tests: {exc}")
    url = container.get_connection_url()
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    try:
        yield url
    finally:
        container.stop()


@pytest.fixture
async def app_client(pg_url: str) -> AsyncIterator[tuple[object, object]]:
    """(app, httpx.AsyncClient) wired to the test DB with fake models. Tables are emptied
    after each test."""
    import httpx
    from sqlalchemy import text

    from app.components import Components
    from app.db.session import create_engine, create_session_factory
    from app.main import create_app
    from app.testing import FakeLLM, HashingEmbedder, OverlapReranker

    settings = make_settings(database_url=pg_url)
    engine = create_engine(settings)
    components = Components(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        embedder=HashingEmbedder(settings.embedding_dim),
        reranker=OverlapReranker(),
        llm=FakeLLM(),
    )
    app = create_app(settings, components)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
        ) as client,
    ):
        yield app, client
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE documents, sessions RESTART IDENTITY CASCADE"))
    await engine.dispose()
