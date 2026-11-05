"""Async engine / session factory helpers."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Create the async engine (asyncpg driver)."""
    return create_async_engine(
        settings.database_url, pool_size=settings.db_pool_size, pool_pre_ping=True
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def check_database(engine: AsyncEngine) -> bool:
    """True when the DB is reachable and the pgvector extension is installed."""
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            )
            return bool(result.scalar_one())
    except Exception:
        return False
