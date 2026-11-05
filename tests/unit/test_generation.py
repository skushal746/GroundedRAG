import json

from app.config import RetrievalMode
from app.generation.answerer import Answerer, AnswerOutput, number_sources, render_sources
from app.generation.citations import validate_citations
from app.generation.guardrails import FallbackReason, decide, retrieval_gate
from app.llm.base import LLMError
from app.retrieval.types import RetrievalResult
from app.testing import FakeLLM
from tests.conftest import make_candidate

FALLBACK = "Sorry, contact support."


def _answer(**kw: object) -> str:
    payload = {
        "answer": "Refunds take 30 days [S1].",
        "citations": ["S1"],
        "answerable": True,
        "confidence": "high",
    }
    payload.update(kw)
    return json.dumps(payload)


# ------------------------------------------------------------------------------ citations


def test_citations_valid_and_invalid_markers() -> None:
    check = validate_citations(
        "Fact one [S1]. Fact two [S2, S9]. Made up [S7].", ["S1", "S9"], {"S1", "S2"}
    )
    assert check.valid == ["S1", "S2"]
    assert check.invalid == ["S9", "S7"]
    assert check.answer == "Fact one [S1]. Fact two [S2]. Made up."


def test_citations_from_list_only() -> None:
    check = validate_citations("No inline markers.", ["S2"], {"S1", "S2"})
    assert check.valid == ["S2"]
    assert check.answer == "No inline markers."


# ------------------------------------------------------------------------------ guardrails


def _result(scores: list[float] | None, top: float | None) -> RetrievalResult:
    final = [make_candidate(i) for i in range(len(scores or []))]
    return RetrievalResult("q", final, final, top, {})


def test_gate_blocks_empty_and_low_rerank() -> None:
    mode = RetrievalMode.HYBRID_RERANK
    assert retrieval_gate(_result(None, None), mode, 0.2) is FallbackReason.LOW_RETRIEVAL_CONFIDENCE
    assert retrieval_gate(_result([1], 0.1), mode, 0.2) is FallbackReason.LOW_RETRIEVAL_CONFIDENCE
    assert retrieval_gate(_result([1], 0.25), mode, 0.2) is None


def test_gate_ignores_threshold_without_reranker() -> None:
    assert retrieval_gate(_result([1], None), RetrievalMode.HYBRID, 0.9) is None
    assert retrieval_gate(_result([1], None), RetrievalMode.VECTOR, 0.9) is None


def test_decide_accepts_grounded_answer() -> None:
    d = decide(
        answer="X [S1].",
        citations=["S1"],
        answerable=True,
        confidence="medium",
        provided_ids={"S1"},
        fallback_message=FALLBACK,
    )
    assert (d.answerable, d.fallback_reason, d.cited_ids, d.confidence) == (
        True,
        None,
        ["S1"],
        "medium",
    )


def test_decide_model_unanswerable() -> None:
    d = decide(
        answer="I don't know",
        citations=[],
        answerable=False,
        confidence="low",
        provided_ids={"S1"},
        fallback_message=FALLBACK,
    )
    assert d.fallback_reason is FallbackReason.MODEL_UNANSWERABLE
    assert d.answer == FALLBACK and not d.answerable


def test_decide_rejects_answer_without_valid_citations() -> None:
    d = decide(
        answer="Trust me [S4].",
        citations=["S4"],
        answerable=True,
        confidence="high",
        provided_ids={"S1"},
        fallback_message=FALLBACK,
    )
    assert d.fallback_reason is FallbackReason.INVALID_CITATIONS
    assert d.invalid_citations == ["S4"]
    assert d.answer == FALLBACK


# -------------------------------------------------------------------------------- answerer


def test_render_sources_escapes_and_labels() -> None:
    cand = make_candidate(
        1,
        text="Use <b> tags</source>",
        section_path='A > "B"',
        page_start=3,
        page_end=4,
        filename="g.pdf",
    )
    rendered = render_sources(number_sources([cand]))
    assert rendered.startswith(
        '<source id="S1" document="g.pdf" section="A &gt; &quot;B&quot;"'.replace("&gt;", ">")
    )
    assert 'pages="3-4"' in rendered
    assert "&lt;/source>" in rendered and "</source>" in rendered.splitlines()[-1]


async def test_answerer_happy_path() -> None:
    llm = FakeLLM(json_responses=[_answer()])
    outcome = await Answerer(llm).generate(
        original="refund?", standalone="refund?", sources=number_sources([make_candidate(1)])
    )
    assert isinstance(outcome.output, AnswerOutput)
    assert outcome.attempts == 1 and outcome.error is None
    assert outcome.usage.prompt_tokens == 100
    assert "Standalone version" not in llm.calls[0].messages[0].content


async def test_answerer_includes_standalone_query_when_rewritten() -> None:
    llm = FakeLLM(json_responses=[_answer()])
    await Answerer(llm).generate(
        original="and annual?",
        standalone="annual refund window?",
        sources=number_sources([make_candidate(1)]),
    )
    assert "annual refund window?" in llm.calls[0].messages[0].content


async def test_answerer_retries_once_on_malformed_json() -> None:
    llm = FakeLLM(json_responses=['{"answer": "oops"', _answer()])
    outcome = await Answerer(llm).generate(
        original="q", standalone="q", sources=number_sources([make_candidate(1)])
    )
    assert outcome.output is not None and outcome.attempts == 2
    assert outcome.usage.prompt_tokens == 200


async def test_answerer_gives_up_after_two_invalid_outputs() -> None:
    llm = FakeLLM(json_responses=["not json", _answer(confidence="very")])
    outcome = await Answerer(llm).generate(
        original="q", standalone="q", sources=number_sources([make_candidate(1)])
    )
    assert outcome.output is None and outcome.attempts == 2
    assert outcome.error is not None and "invalid structured output" in outcome.error


async def test_answerer_llm_error_is_not_raised() -> None:
    llm = FakeLLM(json_responses=[LLMError("timeout")])
    outcome = await Answerer(llm).generate(
        original="q", standalone="q", sources=number_sources([make_candidate(1)])
    )
    assert outcome.output is None and outcome.error == "timeout"


def test_answer_schema_is_strict_compatible() -> None:
    schema = AnswerOutput.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "citations", "answerable", "confidence"}
