"""LLM clients, embedders and reranker wrappers, tested against fake SDK objects."""

import json
import logging
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import numpy as np
import openai
import pytest

from app.config import EmbeddingProvider, LLMProvider
from app.llm import build_llm_client
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import ChatMessage, LLMError
from app.llm.openai_client import OpenAIClient
from app.observability.logging import JsonFormatter, configure_logging, request_id_var
from app.retrieval.embedder import (
    BGE_QUERY_INSTRUCTION,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import CrossEncoderReranker
from app.testing import FakeLLM, HashingEmbedder, OverlapReranker
from tests.conftest import make_settings

MSGS = [ChatMessage("user", "hi"), ChatMessage("assistant", "hello"), ChatMessage("user", "q")]


class _Recorder:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response, self.error, self.kwargs = response, error, {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _openai(content: str | None, refusal: str | None = None) -> tuple[OpenAIClient, _Recorder]:
    msg = SimpleNamespace(content=content, refusal=refusal)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )
    rec = _Recorder(resp)
    client = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    return OpenAIClient("gpt-x", "k", client=client, temperature=0.0), rec


async def test_openai_complete_json_sends_strict_schema() -> None:
    client, rec = _openai('{"a": 1}')
    out = await client.complete_json("sys", MSGS, schema={"type": "object"}, schema_name="s")
    assert out.text == '{"a": 1}' and out.usage.prompt_tokens == 11
    assert rec.kwargs["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in rec.kwargs["messages"]] == ["system", "user", "assistant", "user"]
    assert rec.kwargs["temperature"] == 0.0


async def test_openai_omits_temperature_when_none() -> None:
    client, rec = _openai("x")
    client._temperature = None
    await client.complete("sys", MSGS)
    assert "temperature" not in rec.kwargs


async def test_openai_errors_become_llm_errors() -> None:
    client, rec = _openai("x")
    rec.error = openai.APITimeoutError(request=httpx.Request("POST", "https://x"))
    with pytest.raises(LLMError, match="APITimeoutError"):
        await client.complete("sys", MSGS)
    client, _ = _openai("", None)
    with pytest.raises(LLMError, match="Empty"):
        await client.complete("sys", MSGS)
    client, _ = _openai(None, refusal="no")
    with pytest.raises(LLMError, match="refused"):
        await client.complete("sys", MSGS)


def _anthropic(blocks: list[Any]) -> tuple[AnthropicClient, _Recorder]:
    resp = SimpleNamespace(
        content=blocks,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=20, output_tokens=5),
    )
    rec = _Recorder(resp)
    return AnthropicClient("claude-x", "k", client=SimpleNamespace(messages=rec)), rec


async def test_anthropic_tool_use_structured_output() -> None:
    block = SimpleNamespace(type="tool_use", name="s", input={"answer": "x"})
    client, rec = _anthropic([block])
    out = await client.complete_json("sys", MSGS, schema={"type": "object"}, schema_name="s")
    assert json.loads(out.text) == {"answer": "x"}
    assert rec.kwargs["tool_choice"] == {"type": "tool", "name": "s"}
    assert rec.kwargs["system"] == "sys"
    assert out.usage.completion_tokens == 5


async def test_anthropic_text_and_errors() -> None:
    client, _ = _anthropic([SimpleNamespace(type="text", text="hello")])
    assert (await client.complete("sys", MSGS)).text == "hello"
    client, _ = _anthropic([SimpleNamespace(type="text", text="")])
    with pytest.raises(LLMError, match="Empty"):
        await client.complete("sys", MSGS)
    client, _ = _anthropic([SimpleNamespace(type="text", text="no tool")])
    with pytest.raises(LLMError, match="tool_use"):
        await client.complete_json("sys", MSGS, schema={}, schema_name="s")
    client, rec = _anthropic([])
    rec.error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    with pytest.raises(LLMError, match="APIConnectionError"):
        await client.complete("sys", MSGS)


def test_build_llm_client_by_provider() -> None:
    assert isinstance(build_llm_client(make_settings()), FakeLLM)
    oa = build_llm_client(make_settings(llm_provider=LLMProvider.OPENAI, openai_api_key="k"))
    assert isinstance(oa, OpenAIClient) and oa.model == "gpt-4.1-mini"
    an = build_llm_client(
        make_settings(
            llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key="k", llm_model="claude-haiku-5-5"
        )
    )
    assert isinstance(an, AnthropicClient) and an.model == "claude-haiku-5-5"


async def test_query_rewriter() -> None:
    rewriter = QueryRewriter(FakeLLM(text_responses=['"What is the annual refund window?"\nextra']))
    assert (await rewriter.rewrite("and annual?", [])).query == "and annual?"
    out = await rewriter.rewrite("and annual?", MSGS)
    assert out.query == "What is the annual refund window?" and out.rewritten
    failing = QueryRewriter(FakeLLM(text_responses=[LLMError("down")]))
    assert (await failing.rewrite("and annual?", MSGS)).query == "and annual?"


# ------------------------------------------------------------------- embedders / reranker


class _FakeST:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        self.inputs.extend(texts)
        return np.ones((len(texts), 3), dtype=np.float32)


def test_sentence_transformer_embedder_uses_bge_query_prefix() -> None:
    model = _FakeST()
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", model=model)
    assert emb.dim == 3
    assert emb.embed_documents(["a", "b"]) == [[1.0, 1.0, 1.0]] * 2
    assert emb.embed_query("q") == [1.0, 1.0, 1.0]
    assert model.inputs == ["a", "b", BGE_QUERY_INSTRUCTION + "q"]


def test_openai_embedder_batches() -> None:
    calls: list[int] = []

    def create(model: str, input: list[str], dimensions: int) -> Any:
        calls.append(len(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * dimensions) for _ in input])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    emb = OpenAIEmbedder("text-embedding-3-small", 4, "k", client=client)
    assert len(emb.embed_documents(["x"] * 300)) == 300
    assert calls == [256, 44]
    assert emb.embed_query("q") == [0.1] * 4


def test_build_embedder_checks_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(build_embedder(make_settings()), HashingEmbedder)
    assert isinstance(
        build_embedder(
            make_settings(embedding_provider=EmbeddingProvider.OPENAI, openai_api_key="k")
        ),
        OpenAIEmbedder,
    )
    monkeypatch.setattr(
        "app.retrieval.embedder.SentenceTransformerEmbedder",
        lambda name: SentenceTransformerEmbedder(name, model=_FakeST()),  # 3-dim model
    )
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        build_embedder(make_settings(embedding_provider=EmbeddingProvider.LOCAL))


def test_cross_encoder_reranker_wrapper() -> None:
    class _CE:
        def predict(self, pairs: list[tuple[str, str]], **kw: Any) -> list[float]:
            assert kw["activation_fn"] is not None
            return [0.25 for _ in pairs]

    reranker = CrossEncoderReranker("x", model=_CE())
    assert reranker.score("q", ["a", "b"]) == [0.25, 0.25]
    assert reranker.score("q", []) == []


def test_fake_models() -> None:
    emb = HashingEmbedder(16)
    v1, v2 = emb.embed_query("refund annual plan"), emb.embed_documents(["annual plan refund"])[0]
    assert pytest.approx(sum(a * b for a, b in zip(v1, v2, strict=True))) == 1.0
    assert OverlapReranker().score("refund policy", ["refund", "nothing", ""]) == [0.5, 0.0, 0.0]
    assert OverlapReranker().score("the a", ["x"]) == [0.0]


# ----------------------------------------------------------------------------- logging


def test_json_formatter_includes_request_id_and_extras() -> None:
    token = request_id_var.set("req-123")
    try:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("w",), None)
        record.stage = "rerank"
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["msg"] == "hello w" and line["request_id"] == "req-123"
    assert line["stage"] == "rerank" and line["level"] == "INFO"


def test_json_formatter_exceptions_and_configure() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "err", (), sys.exc_info())
    assert "boom" in json.loads(JsonFormatter().format(record))["exc"]
    configure_logging("debug")
    assert logging.getLogger().level == logging.DEBUG
    configure_logging("INFO")


def test_settings_helpers() -> None:
    s = make_settings(api_keys=" a, b ,,", llm_temperature="")
    assert s.api_key_set == frozenset({"a", "b"})
    assert s.llm_temperature is None
    assert s.resolved_llm_model == "fake-llm"
