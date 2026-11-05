"""Pure metric functions for the eval harness (no I/O, unit-tested)."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where a retrieved/cited passage came from."""

    filename: str
    section: str | None
    page_start: int | None
    page_end: int | None


def matches(expected: dict[str, Any], ref: SourceRef) -> bool:
    """True if `ref` satisfies one expected-source spec: same file, page inside the
    passage's page range (if given) and section substring match (if given)."""
    if expected["file"] != ref.filename:
        return False
    page = expected.get("page")
    if page is not None and not (
        ref.page_start is not None and ref.page_start <= page <= (ref.page_end or ref.page_start)
    ):
        return False
    section = expected.get("section")
    return section is None or section.lower() in (ref.section or "").lower()


def first_hit_rank(expected: Sequence[dict[str, Any]], ranked: Sequence[SourceRef]) -> int | None:
    """1-based rank of the first passage matching any expected source, else None."""
    for rank, ref in enumerate(ranked, start=1):
        if any(matches(e, ref) for e in expected):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def any_match(expected: Sequence[dict[str, Any]], refs: Sequence[SourceRef]) -> bool:
    return any(matches(e, r) for e in expected for r in refs)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def gate_sweep(
    scores: Sequence[tuple[float | None, bool]], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """For each threshold: how many answerable questions the gate would wrongly block and how
    many unanswerable ones it would correctly block. `scores` = (top_rerank_score, answerable)."""
    rows = []
    n_ans = sum(1 for _, a in scores if a) or 1
    n_un = sum(1 for _, a in scores if not a) or 1
    for t in thresholds:
        blocked_ans = sum(1 for s, a in scores if a and (s or 0.0) < t)
        blocked_un = sum(1 for s, a in scores if not a and (s or 0.0) < t)
        rows.append(
            {
                "threshold": t,
                "answerable_blocked": blocked_ans,
                "answerable_blocked_pct": blocked_ans / n_ans,
                "unanswerable_blocked": blocked_un,
                "unanswerable_blocked_pct": blocked_un / n_un,
            }
        )
    return rows
