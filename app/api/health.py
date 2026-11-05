"""Liveness and readiness probes (unauthenticated)."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.deps import ComponentsDep
from app.api.schemas import HealthOut
from app.db.session import check_database

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Liveness")
async def health() -> HealthOut:
    return HealthOut(status="ok")


@router.get(
    "/ready",
    response_model=HealthOut,
    summary="Readiness: DB reachable, pgvector installed, models loaded",
    responses={503: {"model": HealthOut}},
)
async def ready(components: ComponentsDep) -> JSONResponse:
    checks = {
        "database": await check_database(components.engine),
        "embedder": components.embedder is not None,
        "llm_client": components.llm is not None,
    }
    body = HealthOut(status="ok" if all(checks.values()) else "unavailable", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if body.status == "ok" else 503)
