"""End-to-end API tests against a real Postgres + pgvector (testcontainers), with the
embedder, reranker and LLM replaced by deterministic fakes."""

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select

from app.db.models import Chunk, QueryLog
from app.llm.base import LLMError
from app.testing import FakeLLM
from tests.conftest import SAMPLE_DOCS

pytestmark = pytest.mark.integration

Client = tuple[FastAPI, httpx.AsyncClient]

FAQ = b"""# Help Center

## Billing

### Can I get a refund?

Annual plans can be refunded within 30 days of purchase. Monthly plans are not refundable.

## Developers

### Rate limits

The Team plan allows 300 API requests per minute. The Business plan allows 1000 requests
per minute. Exceeding it returns error ORB-4091 with a Retry-After header.
"""


async def _upload(client: httpx.AsyncClient, name: str, content: bytes) -> httpx.Response:
    return await client.post("/v1/documents", files={"file": (name, content)})


async def _ingest_faq(client: httpx.AsyncClient) -> dict[str, Any]:
    resp = await _upload(client, "faq.md", FAQ)
    assert resp.status_code == 202, resp.text
    doc = (await client.get(f"/v1/documents/{resp.json()['id']}")).json()
    assert doc["status"] == "ready", doc
    return doc


def _set_llm(app: FastAPI, llm: FakeLLM) -> FakeLLM:
    app.state.components.llm = llm
    return llm


def _answer(answer: str, cites: list[str], answerable: bool = True) -> str:
    return json.dumps(
        {"answer": answer, "citations": cites, "answerable": answerable, "confidence": "high"}
    )


async def test_health_ready_and_auth(app_client: Client) -> None:
    _, client = app_client
    assert (await client.get("/health")).json() == {"status": "ok", "checks": {}}
    ready = await client.get("/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] is True
    assert "X-Request-ID" in ready.headers
    for headers in ({"X-API-Key": "wrong"}, {"X-API-Key": ""}):
        resp = await client.get("/v1/documents", headers=headers)
        assert resp.status_code == 401


async def test_upload_validation(app_client: Client) -> None:
    _, client = app_client
    resp = await _upload(client, "malware.exe", b"MZ...")
    assert resp.status_code == 415 and ".exe" in resp.json()["detail"]
    assert (await _upload(client, "empty.md", b"")).status_code == 400


async def test_ingest_is_idempotent_and_deletable(app_client: Client) -> None:
    app, client = app_client
    doc = await _ingest_faq(client)
    assert doc["num_chunks"] == 2 and doc["title"] == "Help Center"

    again = await _upload(client, "faq-copy.md", FAQ)
    assert again.status_code == 200 and again.json()["id"] == doc["id"]
    listing = (await client.get("/v1/documents")).json()
    assert listing["total"] == 1

    assert (await client.delete(f"/v1/documents/{doc['id']}")).status_code == 204
    assert (await client.get(f"/v1/documents/{doc['id']}")).status_code == 404
    assert (await client.delete(f"/v1/documents/{doc['id']}")).status_code == 404
    async with app.state.components.session_factory() as db:
        assert (await db.execute(select(func.count()).select_from(Chunk))).scalar_one() == 0


async def test_failed_ingestion_is_recorded_and_retried(app_client: Client) -> None:
    _, client = app_client
    resp = await _upload(client, "broken.pdf", b"%PDF-1.4 definitely not a pdf")
    assert resp.status_code == 202
    doc = (await client.get(f"/v1/documents/{resp.json()['id']}")).json()
    assert doc["status"] == "failed" and "PDF" in doc["error"]
    retry = await _upload(client, "broken.pdf", b"%PDF-1.4 definitely not a pdf")
    # Same content as a *failed* document is accepted again for reprocessing.
    assert retry.status_code == 202 and retry.json()["id"] == doc["id"]
    assert retry.json()["status"] == "processing"


async def test_sample_documents_all_formats(app_client: Client) -> None:
    _, client = app_client
    for path in sorted(SAMPLE_DOCS.iterdir()):
        resp = await _upload(client, path.name, path.read_bytes())
        assert resp.status_code == 202, path.name
        doc = (await client.get(f"/v1/documents/{resp.json()['id']}")).json()
        assert doc["status"] == "ready", (path.name, doc["error"])
        assert doc["num_chunks"] > 0
    pdf = next(
        d
        for d in (await client.get("/v1/documents")).json()["items"]
        if d["filename"].endswith(".pdf")
    )
    assert pdf["num_pages"] == 6


async def test_chat_answer_with_citations_and_query_log(app_client: Client) -> None:
    app, client = app_client
    doc = await _ingest_faq(client)
    resp = await client.post(
        "/v1/chat",
        json={"message": "Can I get a refund on annual plans?"},
        headers={"X-Request-ID": "req-abc"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answerable"] is True and body["fallback_reason"] is None
    assert "30 days" in body["answer"] and "[S1]" in body["answer"]
    (source,) = body["sources"]
    assert source["id"] == "S1" and source["document_id"] == doc["id"]
    assert source["filename"] == "faq.md"
    assert source["section"] == "Billing > Can I get a refund?"
    assert 0 < source["score"] <= 1
    assert resp.headers["X-Request-ID"] == "req-abc"

    async with app.state.components.session_factory() as db:
        log = (await db.execute(select(QueryLog))).scalar_one()
    assert log.request_id == "req-abc"
    assert log.original_query == log.rewritten_query == "Can I get a refund on annual plans?"
    assert log.retrieval_mode == "hybrid_rerank" and log.prompt_version == "answer-v1"
    assert log.cited_source_ids == ["S1"] and log.top_rerank_score is not None
    assert {"rewrite_ms", "retrieve_ms", "rerank_ms", "generate_ms", "total_ms"} <= set(
        log.latency_ms
    )
    assert any(c["keyword_rank"] for c in log.candidates)
    assert any(c["vector_rank"] for c in log.candidates)
    assert log.prompt_tokens > 0


async def test_keyword_search_finds_exact_error_code(app_client: Client) -> None:
    _, client = app_client
    await _ingest_faq(client)
    body = (await client.post("/v1/chat", json={"message": "ORB-4091"})).json()
    assert body["sources"][0]["section"] == "Developers > Rate limits"


async def test_fallback_low_retrieval_confidence_skips_llm(app_client: Client) -> None:
    app, client = app_client
    await _ingest_faq(client)
    llm = _set_llm(app, FakeLLM())
    body = (await client.post("/v1/chat", json={"message": "Who is the CEO?"})).json()
    assert body["fallback_reason"] == "low_retrieval_confidence"
    assert body["answerable"] is False and body["sources"] == []
    assert body["answer"] == app.state.components.settings.fallback_message
    assert not [c for c in llm.calls if c.kind == "complete_json"], "LLM must not be called"


async def test_fallback_when_knowledge_base_empty(app_client: Client) -> None:
    _, client = app_client
    body = (await client.post("/v1/chat", json={"message": "refund?"})).json()
    assert body["fallback_reason"] == "low_retrieval_confidence"


@pytest.mark.parametrize(
    ("responses", "reason"),
    [
        ([_answer("Not in the sources.", [], answerable=False)], "model_unanswerable"),
        ([_answer("Refunds are instant [S7].", ["S7"])], "invalid_citations"),
        ([LLMError("timeout")], "llm_error"),
        (["{not json", '{"answer": 1}'], "llm_error"),
    ],
)
async def test_fallback_paths(app_client: Client, responses: list[Any], reason: str) -> None:
    app, client = app_client
    await _ingest_faq(client)
    _set_llm(app, FakeLLM(json_responses=responses))
    resp = await client.post("/v1/chat", json={"message": "Can I get a refund on annual plans?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["fallback_reason"] == reason and body["answerable"] is False
    assert body["sources"] == [] and body["confidence"] == "low"


async def test_retry_after_malformed_json_succeeds(app_client: Client) -> None:
    app, client = app_client
    await _ingest_faq(client)
    _set_llm(app, FakeLLM(json_responses=["{oops", _answer("30 days [S1].", ["S1"])]))
    body = (await client.post("/v1/chat", json={"message": "annual refund window?"})).json()
    assert body["answerable"] is True and body["answer"] == "30 days [S1]."


async def test_session_continuity_and_history(app_client: Client) -> None:
    app, client = app_client
    await _ingest_faq(client)
    llm = _set_llm(
        app, FakeLLM(text_responses=["What is the API rate limit on the Business plan?"])
    )
    first = (await client.post("/v1/chat", json={"message": "What's the Team rate limit?"})).json()
    session_id = first["session_id"]
    second = (
        await client.post("/v1/chat", json={"session_id": session_id, "message": "And Business?"})
    ).json()
    assert second["session_id"] == session_id
    assert second["sources"][0]["section"] == "Developers > Rate limits"

    rewrite_calls = [c for c in llm.calls if c.kind == "complete"]
    assert len(rewrite_calls) == 1, "first turn has no history, so no rewrite"
    assert "What's the Team rate limit?" in rewrite_calls[0].messages[0].content
    answer_prompt = [c for c in llm.calls if c.kind == "complete_json"][-1].messages[0].content
    assert "What is the API rate limit on the Business plan?" in answer_prompt

    history = (await client.get(f"/v1/sessions/{session_id}/messages")).json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[1]["meta"]["sources"][0]["id"] == "S1"

    async with app.state.components.session_factory() as db:
        logs = (await db.execute(select(QueryLog).order_by(QueryLog.id))).scalars().all()
    assert logs[1].rewritten_query == "What is the API rate limit on the Business plan?"
    assert logs[1].original_query == "And Business?"

    assert (await client.delete(f"/v1/sessions/{session_id}")).status_code == 204
    assert (await client.get(f"/v1/sessions/{session_id}/messages")).status_code == 404
    assert (await client.delete(f"/v1/sessions/{session_id}")).status_code == 404


async def test_unknown_session_is_404(app_client: Client) -> None:
    _, client = app_client
    resp = await client.post(
        "/v1/chat",
        json={"session_id": "00000000-0000-0000-0000-000000000000", "message": "hi"},
    )
    assert resp.status_code == 404


async def test_feedback(app_client: Client) -> None:
    _, client = app_client
    await _ingest_faq(client)
    body = (await client.post("/v1/chat", json={"message": "annual refund?"})).json()
    fb = await client.post(
        f"/v1/chat/{body['message_id']}/feedback", json={"helpful": False, "comment": "wrong"}
    )
    assert fb.status_code == 201 and fb.json()["helpful"] is False
    missing = await client.post(
        "/v1/chat/00000000-0000-0000-0000-000000000000/feedback", json={"helpful": True}
    )
    assert missing.status_code == 404


async def test_vector_only_mode(app_client: Client) -> None:
    from app.config import RetrievalMode
    from app.retrieval.retriever import Retriever

    app, client = app_client
    await _ingest_faq(client)
    components = app.state.components
    retriever = Retriever(components.embedder, None, components.settings, mode=RetrievalMode.VECTOR)
    async with components.session_factory() as db:
        result = await retriever.retrieve(db, "annual refund")
    assert result.final and result.top_rerank_score is None
    assert all(c.keyword_rank is None for c in result.candidates)
    with pytest.raises(ValueError, match="reranker"):
        Retriever(components.embedder, None, components.settings)


async def test_openapi_is_documented(app_client: Client) -> None:
    _, client = app_client
    spec = (await client.get("/openapi.json")).json()
    assert {"/v1/chat", "/v1/documents", "/v1/sessions/{session_id}/messages"} <= set(spec["paths"])
    chat = spec["paths"]["/v1/chat"]["post"]
    assert chat["tags"] == ["chat"]
    assert "examples" in spec["components"]["schemas"]["ChatResponse"]
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]
