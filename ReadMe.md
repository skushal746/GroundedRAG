# Build: "GroundedRAG": a reliable customer-support RAG assistant (Python / FastAPI)

## Goal
Build a production-quality, portfolio-grade RAG knowledge assistant that answers customer
questions ONLY from uploaded documents, cites its sources, refuses when it can't find a
reliable answer, and includes an evaluation harness with measured results. The point of the
repo is to show that I understand how to make RAG *reliable*, not just how to connect an LLM to
a vector DB. Code quality, tests and a clear README matter as much as the features.

Build the whole project in one go, run it end to end, run the tests and the eval, and fix
anything that fails before you finish.

## Tech stack (use exactly this unless something is blocked; if so, tell me why)
- Python 3.12, managed with `uv` (pyproject.toml, lockfile)
- FastAPI + Pydantic v2 + pydantic-settings
- PostgreSQL 16 + **pgvector** (one DB for vectors, full-text search, sessions, and query logs)
- SQLAlchemy 2.0 (async) + asyncpg, Alembic mig