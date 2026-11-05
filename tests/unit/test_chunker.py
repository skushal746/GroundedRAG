from itertools import pairwise

import pytest

from app.ingestion.chunker import Chunker, approx_token_count, contextual_header, section_path
from app.ingestion.types import Block, ParsedDocument


def _sentences(n: int, prefix: str = "Sentence") -> str:
    return " ".join(f"{prefix} number {i} has some words in it." for i in range(n))


def test_token_count_is_words_plus_punctuation() -> None:
    assert approx_token_count("Hello, world!") == 4


def test_chunks_never_straddle_sections() -> None:
    doc = ParsedDocument(
        title="Doc",
        blocks=[
            Block("Alpha text.", headings=("A",)),
            Block("More alpha.", headings=("A",)),
            Block("Beta text.", headings=("B", "B1")),
        ],
    )
    chunks = Chunker().chunk(doc)
    assert [c.section_path for c in chunks] == ["A", "B > B1"]
    assert chunks[0].text == "Alpha text.\n\nMore alpha."
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_long_section_is_split_with_bounded_size_and_overlap() -> None:
    doc = ParsedDocument(title="Doc", blocks=[Block(_sentences(60), headings=("S",))])
    chunker = Chunker(max_tokens=100, overlap_tokens=25)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 3
    assert all(c.token_count <= 100 for c in chunks)
    for prev, nxt in pairwise(chunks):
        overlap = nxt.text[: nxt.text.index(".") + 1]
        assert overlap in prev.text, "next chunk should start with a sentence of the previous"


def test_overlap_respects_budget() -> None:
    doc = ParsedDocument(title="Doc", blocks=[Block(_sentences(30), headings=("S",))])
    chunks = Chunker(max_tokens=60, overlap_tokens=12).chunk(doc)
    first_sentence_tokens = approx_token_count("Sentence number 0 has some words in it.")
    for prev, nxt in pairwise(chunks):
        shared = [s for s in nxt.text.split(". ") if s in prev.text]
        assert sum(approx_token_count(s) for s in shared) <= 12 + first_sentence_tokens


def test_page_range_and_contextual_header() -> None:
    doc = ParsedDocument(
        title="Guide",
        blocks=[
            Block("Page one text.", headings=("Guide", "Limits"), page=4),
            Block("Page two text.", headings=("Guide", "Limits"), page=5),
        ],
    )
    (chunk,) = Chunker().chunk(doc)
    assert (chunk.page_start, chunk.page_end) == (4, 5)
    assert chunk.section_path == "Limits"  # leading heading equal to the title is dropped
    assert chunk.embed_text == "Guide\nSection: Limits\n\n" + chunk.text
    assert "Section" not in chunk.text


def test_big_table_is_split_by_rows_repeating_header() -> None:
    rows = "\n".join(f"| plan {i} | {i * 10} GB |" for i in range(80))
    table = "| Plan | Storage |\n|---|---|\n" + rows
    doc = ParsedDocument(title="T", blocks=[Block(table, headings=("Limits",), kind="table")])
    chunks = Chunker(max_tokens=120, overlap_tokens=20).chunk(doc)
    assert len(chunks) > 1
    assert all(c.text.startswith("| Plan | Storage |\n|---|---|") for c in chunks)
    all_rows = [line for c in chunks for line in c.text.splitlines()[2:]]
    assert all_rows == rows.splitlines(), "tables are not overlapped; every row exactly once"


def test_run_on_text_is_hard_split() -> None:
    doc = ParsedDocument(title="T", blocks=[Block("word " * 400)])
    chunks = Chunker(max_tokens=100, overlap_tokens=10).chunk(doc)
    assert all(c.token_count <= 100 for c in chunks)
    assert chunks[0].section_path is None
    assert chunks[0].embed_text.startswith("T\n\n")


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Chunker(max_tokens=50, overlap_tokens=50)


def test_section_path_helpers() -> None:
    assert section_path((), "T") is None
    assert section_path(("T", "A", "B"), "t") == "A > B"
    assert contextual_header("T", None) == "T\n\n"
