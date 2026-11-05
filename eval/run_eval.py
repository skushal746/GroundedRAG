"""Run the eval set through the real pipeline (same ChatService the API uses) and write a
Markdown + JSON report.

    uv run python -m eval.run_eval                  # production config only
    uv run python -m eval.run_eval --ablation       # vector vs hybrid vs hybrid+rerank

Requires the sample docs to be ingested (`make ingest-samples`) and an LLM API key.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import delete

from app.components import Components, build_components
from app.config import RetrievalMode, Settings, get_settings
from app.db.models import ChatSession
from app.generation.answerer import Answerer, number_sources, render_sources
from app.generation.prompts import ANSWER_PROMPT_VERSION
from app.llm import ChatMessage, LLMClient, LLMError, build_llm_client
from app.observability.logging import configure_logging
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.retriever import Retriever
from app.retrieval.types import Candidate
from app.services.chat import ChatService, ChatTurn
from eval.metrics import (
    SourceRef,
    any_match,
    first_hit_rank,
    gate_sweep,
    mean,
    percentile,
    reciprocal_rank,
)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset.yaml"
REPORTS = ROOT / "reports"
THRESHOLDS = [0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

JUDGE_SYSTEM = """\
You are a strict evaluator of a customer-support assistant. You are given a question, a
reference answer written by a human, the source passages the assistant was shown, and the
assistant's answer. Score two things on a 1-5 scale:

correctness: does the answer convey the same key facts as the reference answer?
  5 = all key facts correct, 4 = correct with a minor omission, 3 = partially correct,
  2 = mostly wrong or missing the main fact, 1 = wrong or contradicts the reference.
faithfulness: is every claim in the answer supported by the source passages?
  5 = fully supported, 3 = some unsupported detail, 1 = mainly unsupported / invented.

Judge facts, not wording or style. Give a one-sentence reason."""


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    correctness: Literal[1, 2, 3, 4, 5]
    faithfulness: Literal[1, 2, 3, 4, 5]
    reason: str


@dataclass(slots=True)
class QuestionResult:
    id: str
    category: str
    question: str
    expected_answerable: bool
    rewritten_query: str
    answer: str
    answerable: bool
    confidence: str
    fallback_reason: str | None
    top_rerank_score: float | None
    retrieved: list[dict[str, Any]]
    cited: list[dict[str, Any]]
    hit_rank: int | None
    citation_hit: bool | None
    latency_ms: float
    tokens: int
    correctness: int | None = None
    faithfulness: int | None = None
    judge_reason: str | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def refusal_correct(self) -> bool:
        return self.answerable == self.expected_answerable


def _ref(c: Candidate) -> SourceRef:
    return SourceRef(c.filename, c.section_path, c.page_start, c.page_end)


def _ref_dict(c: Candidate) -> dict[str, Any]:
    return {"file": c.filename, "section": c.section_path, "pages": [c.page_start, c.page_end]}


async def judge(
    llm: LLMClient, item: dict[str, Any], turn: ChatTurn
) -> tuple[int, int, str] | None:
    sources = [s.candidate for s in turn.cited_sources] or turn.retrieval.final
    prompt = (
        f"Question: {item['question']}\n"
        f"Standalone question: {turn.rewritten_query}\n\n"
        f"Reference answer: {item['expected_answer']}\n\n"
        f"Source passages:\n{render_sources(number_sources(sources))}\n\n"
        f"Assistant answer: {turn.decision.answer}"
    )
    for _ in range(2):
        try:
            completion = await llm.complete_json(
                JUDGE_SYSTEM,
                [ChatMessage("user", prompt)],
                schema=JudgeOutput.model_json_schema(),
                schema_name="judgement",
                max_tokens=300,
            )
            out = JudgeOutput.model_validate_json(completion.text)
            return out.correctness, out.faithfulness, out.reason
        except (LLMError, ValidationError):
            continue
    return None


async def run_config(
    components: Components,
    settings: Settings,
    mode: RetrievalMode,
    items: list[dict[str, Any]],
    judge_llm: LLMClient | None,
) -> list[QuestionResult]:
    retriever = Retriever(components.embedder, components.reranker, settings, mode=mode)
    service = ChatService(
        retriever=retriever,
        rewriter=QueryRewriter(components.llm),
        answerer=Answerer(components.llm, settings.llm_max_tokens),
        settings=settings,
        llm_model=components.llm.model,
    )
    results: list[QuestionResult] = []
    session_ids = []
    for item in items:
        async with components.session_factory() as db:
            session_id = None
            for previous in item.get("history", []):
                session_id = (await service.handle(db, session_id, previous)).session_id
            turn = await service.handle(db, session_id, item["question"])
            session_ids.append(turn.session_id)
        expected = item.get("expected_sources") or []
        retrieved = [_ref(c) for c in turn.retrieval.final]
        cited = [_ref(s.candidate) for s in turn.cited_sources]
        result = QuestionResult(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            expected_answerable=item["answerable"],
            rewritten_query=turn.rewritten_query,
            answer=turn.decision.answer,
            answerable=turn.decision.answerable,
            confidence=turn.decision.confidence,
            fallback_reason=turn.decision.fallback_reason,
            top_rerank_score=turn.retrieval.top_rerank_score,
            retrieved=[_ref_dict(c) for c in turn.retrieval.final],
            cited=[_ref_dict(s.candidate) for s in turn.cited_sources],
            hit_rank=first_hit_rank(expected, retrieved) if expected else None,
            citation_hit=any_match(expected, cited)
            if expected and turn.decision.answerable
            else None,
            latency_ms=turn.latency_ms["total_ms"],
            tokens=turn.usage.prompt_tokens + turn.usage.completion_tokens,
            stage_ms=turn.latency_ms,
        )
        if judge_llm is not None and item["answerable"] and turn.decision.answerable:
            verdict = await judge(judge_llm, item, turn)
            if verdict is not None:
                result.correctness, result.faithfulness, result.judge_reason = verdict
        results.append(result)
        status = result.fallback_reason or f"answered(c={result.correctness})"
        print(f"  [{mode.value:>13}] {item['id']:<7} rank={result.hit_rank} {status}", flush=True)

    async with components.session_factory() as db:  # eval sessions shouldn't pollute logs
        await db.execute(delete(ChatSession).where(ChatSession.id.in_(session_ids)))
        await db.commit()
    return results


def summarize(results: list[QuestionResult]) -> dict[str, float]:
    answerable = [r for r in results if r.expected_answerable]
    unanswerable = [r for r in results if not r.expected_answerable]
    answered = [r for r in answerable if r.answerable]
    judged = [r for r in answered if r.correctness is not None]
    cited = [r for r in answered if r.citation_hit is not None]
    latencies = [r.latency_ms for r in results]
    return {
        "n": len(results),
        "hit_rate_at_1": mean([1.0 if r.hit_rank == 1 else 0.0 for r in answerable]),
        "hit_rate_at_5": mean([1.0 if r.hit_rank else 0.0 for r in answerable]),
        "mrr": mean([reciprocal_rank(r.hit_rank) for r in answerable]),
        "citation_accuracy": mean([1.0 if r.citation_hit else 0.0 for r in cited]),
        "correctness_mean": mean([float(r.correctness or 0) for r in judged]),
        "faithfulness_mean": mean([float(r.faithfulness or 0) for r in judged]),
        # Correct (judge >= 4) over ALL answerable questions: false refusals count as wrong.
        "answer_accuracy": mean(
            [1.0 if (r.answerable and (r.correctness or 0) >= 4) else 0.0 for r in answerable]
        ),
        "correct_refusal_rate": mean([1.0 if not r.answerable else 0.0 for r in unanswerable]),
        "false_refusal_rate": mean([1.0 if not r.answerable else 0.0 for r in answerable]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "avg_tokens": mean([float(r.tokens) for r in results]),
    }


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _score(v: float | None) -> str:
    """Readable rerank score: tiny values in scientific notation (the gate lives there)."""
    if v is None:
        return ""
    return f"{v:.1e}" if v < 0.01 else f"{v:.3f}"


def render_markdown(
    meta: dict[str, Any],
    summaries: dict[str, dict[str, float]],
    details: dict[str, list[QuestionResult]],
    primary: str,
) -> str:
    lines = [
        "# GroundedRAG evaluation report",
        "",
        f"- Date: {meta['date']}",
        f"- LLM: `{meta['llm_model']}` · judge: `{meta['judge_model']}` · prompt: "
        f"`{meta['prompt_version']}`",
        f"- Embeddings: `{meta['embedding_model']}` · reranker: `{meta['reranker_model']}`",
        f"- Confidence gate: `MIN_RERANK_SCORE={meta['min_rerank_score']}` (hybrid_rerank only)",
        f"- Questions: {meta['n_questions']} ({meta['n_answerable']} answerable, "
        f"{meta['n_questions'] - meta['n_answerable']} should be refused)",
        "",
        "## Retrieval ablation",
        "",
        "| Metric | " + " | ".join(summaries) + " |",
        "|---" * (len(summaries) + 1) + "|",
    ]
    rows = [
        ("Hit rate@1", "hit_rate_at_1", _pct),
        ("Hit rate@5", "hit_rate_at_5", _pct),
        ("MRR", "mrr", lambda v: f"{v:.3f}"),
        ("Citation accuracy", "citation_accuracy", _pct),
        ("Answer accuracy (judge ≥4, all answerable)", "answer_accuracy", _pct),
        ("Correctness (1-5, answered)", "correctness_mean", lambda v: f"{v:.2f}"),
        ("Faithfulness (1-5, answered)", "faithfulness_mean", lambda v: f"{v:.2f}"),
        ("Correct refusals (unanswerable)", "correct_refusal_rate", _pct),
        ("False refusals (answerable)", "false_refusal_rate", _pct),
        ("Latency p50", "latency_p50_ms", lambda v: f"{v / 1000:.2f}s"),
        ("Latency p95", "latency_p95_ms", lambda v: f"{v / 1000:.2f}s"),
        ("Avg tokens / query", "avg_tokens", lambda v: f"{v:.0f}"),
    ]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt(s[key]) for s in summaries.values()) + " |")

    results = details[primary]
    lines += [
        "",
        f"## By category ({primary})",
        "",
        "| Category | n | Hit@5 | Correct ≥4 / refused ok |",
        "|---|---|---|---|",
    ]
    for cat in dict.fromkeys(r.category for r in results):
        rs = [r for r in results if r.category == cat]
        hit = [r for r in rs if r.expected_answerable]
        ok = sum(
            1
            for r in rs
            if (r.expected_answerable and r.answerable and (r.correctness or 0) >= 4)
            or (not r.expected_answerable and not r.answerable)
        )
        hit_s = _pct(mean([1.0 if r.hit_rank else 0.0 for r in hit])) if hit else "n/a"
        lines.append(f"| {cat} | {len(rs)} | {hit_s} | {ok}/{len(rs)} |")

    lines += [
        "",
        f"## Per question ({primary})",
        "",
        "| id | category | rank | cited ok | outcome | correct | faithful | top rerank |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        outcome = r.fallback_reason or "answered"
        score = _score(r.top_rerank_score)
        lines.append(
            f"| {r.id} | {r.category} | {r.hit_rank or '-'} | "
            f"{'' if r.citation_hit is None else ('yes' if r.citation_hit else 'no')} | "
            f"{outcome} | {r.correctness or ''} | {r.faithfulness or ''} | {score} |"
        )

    rerank = details.get(RetrievalMode.HYBRID_RERANK.value)
    if rerank:
        lines += [
            "",
            "## Confidence-gate calibration (hybrid_rerank)",
            "",
            "Top cross-encoder score per question, and what each threshold would block:",
            "",
            "| Threshold | Answerable blocked (false refusals) | Unanswerable blocked |",
            "|---|---|---|",
        ]
        for row in gate_sweep(
            [(r.top_rerank_score, r.expected_answerable) for r in rerank], THRESHOLDS
        ):
            lines.append(
                f"| {row['threshold']:g} | {int(row['answerable_blocked'])} "
                f"({_pct(row['answerable_blocked_pct'])}) | {int(row['unanswerable_blocked'])} "
                f"({_pct(row['unanswerable_blocked_pct'])}) |"
            )
        ans = sorted(r.top_rerank_score or 0 for r in rerank if r.expected_answerable)
        una = sorted(r.top_rerank_score or 0 for r in rerank if not r.expected_answerable)
        lines += [
            "",
            f"- Lowest answerable top score: **{_score(ans[0])}** "
            f"(next: {', '.join(_score(s) for s in ans[1:4])})",
            f"- Should-refuse top scores: {', '.join(_score(s) for s in una)}",
            f"- Configured gate: `MIN_RERANK_SCORE={meta['min_rerank_score']:g}`",
        ]

    issues = [
        r
        for r in results
        if (
            r.expected_answerable
            and (not r.answerable or (r.correctness or 5) < 4 or not r.hit_rank)
        )
        or (not r.expected_answerable and r.answerable)
    ]
    lines += ["", f"## Failures and partial answers ({primary})", ""]
    if not issues:
        lines.append("None.")
    for r in issues:
        lines += [
            f"### {r.id}: {r.question}",
            "",
            f"- Rewritten query: {r.rewritten_query}",
            f"- Outcome: {r.fallback_reason or 'answered'} · rank: {r.hit_rank} · "
            f"correctness: {r.correctness} · faithfulness: {r.faithfulness}",
            f"- Answer: {r.answer}",
            f"- Judge: {r.judge_reason or '-'}",
            f"- Retrieved: {', '.join(f'{d["file"]} [{d["section"]}]' for d in r.retrieved)}",
            "",
        ]
    return "\n".join(lines) + "\n"


def render_saved(path: Path) -> None:
    """Re-render a Markdown report from its JSON (no LLM calls)."""
    data = json.loads(path.read_text())
    details = {m: [QuestionResult(**r) for r in rs] for m, rs in data["results"].items()}
    summaries = {m: summarize(rs) for m, rs in details.items()}
    md = render_markdown(data["meta"], summaries, details, list(details)[-1])
    path.with_suffix(".md").write_text(md)
    path.write_text(json.dumps({**data, "summary": summaries}, indent=2, default=str))
    print(f"Re-rendered {path.with_suffix('.md')}")


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging("WARNING")
    logging.getLogger("app").setLevel(logging.WARNING)
    if args.min_rerank_score is not None:
        settings = settings.model_copy(update={"min_rerank_score": args.min_rerank_score})
    items = yaml.safe_load(DATASET.read_text())["questions"]
    if args.only:
        items = [i for i in items if i["id"] in args.only.split(",")]

    components = build_components(
        settings.model_copy(update={"retrieval_mode": RetrievalMode.HYBRID_RERANK})
    )
    judge_settings = settings.model_copy(update={"llm_model": args.judge_model})
    judge_llm = None if args.no_judge else build_llm_client(judge_settings)

    modes = (
        [RetrievalMode.VECTOR, RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK]
        if args.ablation
        else [settings.retrieval_mode]
    )
    details: dict[str, list[QuestionResult]] = {}
    started = time.perf_counter()
    for mode in modes:
        print(f"Running {len(items)} questions with retrieval_mode={mode.value}")
        details[mode.value] = await run_config(components, settings, mode, items, judge_llm)
    await components.engine.dispose()

    summaries = {mode: summarize(rs) for mode, rs in details.items()}
    primary = modes[-1].value
    meta = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "llm_model": components.llm.model,
        "judge_model": judge_llm.model if judge_llm else None,
        "prompt_version": ANSWER_PROMPT_VERSION,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "min_rerank_score": settings.min_rerank_score,
        "n_questions": len(items),
        "n_answerable": sum(1 for i in items if i["answerable"]),
        "runtime_s": round(time.perf_counter() - started, 1),
    }
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = args.name or f"report-{stamp}"
    md = render_markdown(meta, summaries, details, primary)
    (REPORTS / f"{name}.md").write_text(md)
    (REPORTS / f"{name}.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "summary": summaries,
                "results": {m: [asdict(r) for r in rs] for m, rs in details.items()},
            },
            indent=2,
            default=str,
        )
    )
    print("\n" + md.split("## By category")[0])
    print(f"Report written to {REPORTS / (name + '.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true", help="run vector, hybrid, hybrid_rerank")
    parser.add_argument("--judge-model", default="gpt-4.1", help="LLM-as-judge model")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--min-rerank-score", type=float, help="override the confidence gate")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--name", help="report file name (default: timestamped)")
    parser.add_argument("--render", type=Path, help="re-render a saved JSON report and exit")
    args = parser.parse_args()
    if args.render:
        render_saved(args.render)
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
