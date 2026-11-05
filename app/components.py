"""Long-lived components, built once at startup and shared by all requests."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import EmbeddingProvider, RetrievalMode, Settings
from app.db.session import create_engine, create_session_factory
from app.llm import LLMClient, build_llm_client
from app.retrieval.embedder import Embedder, build_embedder
from app.retrieval.reranker import CrossEncoderReranker, Reranker


@dataclass(slots=True)
class Components:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    embedder: Embedder
    reranker: Reranker | None
    llm: LLMClient


def build_components(settings: Settings) -> Components:
    """Load models and clients. Slow (model load), so call once per process."""
    engine = create_engine(settings)
    reranker: Reranker | None = None
    if settings.retrieval_mode is RetrievalMode.HYBRID_RERANK:
        if settings.embedding_provider is EmbeddingProvider.FAKE:
            from app.testing import OverlapReranker

            reranker = OverlapReranker()
        else:
            reranker = CrossEncoderReranker(settings.reranker_model)
    return Components(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        embedder=build_embedder(settings),
        reranker=reranker,
        llm=build_llm_client(settings),
    )
