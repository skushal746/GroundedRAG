from eval.metrics import (
    SourceRef,
    any_match,
    first_hit_rank,
    gate_sweep,
    matches,
    mean,
    percentile,
    reciprocal_rank,
)

GUIDE = SourceRef("guide.pdf", "Plan limits", 5, 5)
FAQ = SourceRef("faq.md", "Billing > Can I get a refund?", None, None)


def test_matches_file_page_and_section() -> None:
    assert matches({"file": "guide.pdf", "page": 5, "section": "plan"}, GUIDE)
    assert not matches({"file": "guide.pdf", "page": 6}, GUIDE)
    assert not matches({"file": "faq.md"}, GUIDE)
    assert matches({"file": "faq.md", "section": "refund"}, FAQ)
    assert not matches({"file": "faq.md", "page": 1}, FAQ)
    assert matches({"file": "x", "page": 3}, SourceRef("x", None, 2, 4))


def test_rank_metrics() -> None:
    expected = [{"file": "faq.md"}]
    assert first_hit_rank(expected, [GUIDE, FAQ]) == 2
    assert first_hit_rank(expected, [GUIDE]) is None
    assert reciprocal_rank(2) == 0.5 and reciprocal_rank(None) == 0.0
    assert any_match(expected, [FAQ]) and not any_match(expected, [])


def test_percentile_and_mean() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50 and percentile(values, 95) == 95
    assert percentile([], 50) == 0.0
    assert mean([]) == 0.0 and mean([1.0, 3.0]) == 2.0


def test_gate_sweep() -> None:
    rows = gate_sweep([(0.9, True), (0.05, True), (0.01, False), (None, False)], [0.0, 0.1])
    assert rows[0]["answerable_blocked"] == 0 and rows[0]["unanswerable_blocked"] == 0
    assert rows[1]["answerable_blocked"] == 1 and rows[1]["unanswerable_blocked"] == 2
    assert rows[1]["unanswerable_blocked_pct"] == 1.0
