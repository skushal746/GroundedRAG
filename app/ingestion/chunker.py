"""Structure-aware chunker.

1. Blocks are grouped into sections by heading path, so a chunk never straddles two
   sections (keeps citations precise: "Billing > Refunds", not "Billing > Refunds / Taxes").
2. Inside a section, paragraphs are split into sentences (tables into row groups that
   repeat the header) and greedily packed up to `max_tokens`.
3. Consecutive chunks in a section overlap by up to `overlap_tokens` of trailing sentences.
4. Each chunk gets a contextual header (document title + section path) in `embed_text`,
   which is what gets embedded and keyword-indexed; `text` stays clean for display.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import groupby

from app.ingestion.types import Block, ChunkDraft, ParsedDocument

TokenCounter = Callable[[str], int]

_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
SECTION_SEPARATOR = " > "


def approx_token_count(text: str) -> int:
    """Cheap, deterministic token estimate (words + punctuation).

    Within ~15% of WordPiece/BPE counts for English prose, which is plenty for sizing
    chunks, and it keeps the chunker independent of any particular model's tokenizer.
    """
    return len(_TOKEN.findall(text))


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    page: int | None
    tokens: int
    new_paragraph: bool


class Chunker:
    """Splits a ParsedDocument into ChunkDrafts. See module docstring for the algorithm."""

    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 80,
        token_counter: TokenCounter = approx_token_count,
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.count = token_counter

    def chunk(self, doc: ParsedDocument) -> list[ChunkDraft]:
        """Chunk a parsed document; chunk_index is sequential across the whole document."""
        drafts: list[ChunkDraft] = []
        for headings, section_blocks in groupby(doc.blocks, key=lambda b: b.headings):
            section = section_path(headings, doc.title)
            units = list(self._units(section_blocks))
            for group in self._pack(units):
                text = self._join(group)
                pages = [u.page for u in group if u.page is not None]
                drafts.append(
                    ChunkDraft(
                        chunk_index=len(drafts),
                        text=text,
                        embed_text=contextual_header(doc.title, section) + text,
                        section_path=section,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        token_count=self.count(text),
                    )
                )
        return drafts

    # -- internals -----------------------------------------------------------------

    def _units(self, blocks: Iterable[Block]) -> Iterable[_Unit]:
        for block in blocks:
            pieces = (
                self._split_table(block.text)
                if block.kind == "table"
                else [block.text]
                if block.kind in ("list", "code") and self.count(block.text) <= self.max_tokens
                else self._split_sentences(block.text)
            )
            for i, piece in enumerate(pieces):
                yield _Unit(piece, block.page, self.count(piece), new_paragraph=i == 0)

    def _split_sentences(self, text: str) -> list[str]:
        out: list[str] = []
        for sentence in _SENTENCE_END.split(text):
            if self.count(sentence) <= self.max_tokens:
                out.append(sentence)
                continue
            words, current = sentence.split(), []
            for word in words:  # pathological run-on text: hard split on words
                current.append(word)
                if self.count(" ".join(current)) >= self.max_tokens - 10:
                    out.append(" ".join(current))
                    current = []
            if current:
                out.append(" ".join(current))
        return out

    def _split_table(self, table: str) -> list[str]:
        """Keep small tables whole; split big ones into row groups that repeat the header."""
        if self.count(table) <= self.max_tokens:
            return [table]
        lines = table.splitlines()
        header, rows = lines[:2], lines[2:]
        budget = self.max_tokens - self.count("\n".join(header))
        groups: list[list[str]] = [[]]
        for row in rows:
            if groups[-1] and self.count("\n".join([*groups[-1], row])) > budget:
                groups.append([])
            groups[-1].append(row)
        return ["\n".join(header + g) for g in groups if g]

    def _pack(self, units: list[_Unit]) -> Iterable[list[_Unit]]:
        current: list[_Unit] = []
        size = 0
        for unit in units:
            if current and size + unit.tokens > self.max_tokens:
                yield current
                current = self._overlap(current, budget=self.max_tokens - unit.tokens)
                size = sum(u.tokens for u in current)
            current.append(unit)
            size += unit.tokens
        if current:
            yield current

    def _overlap(self, previous: list[_Unit], budget: int) -> list[_Unit]:
        """Trailing non-table units of `previous` totalling <= overlap_tokens."""
        carried: list[_Unit] = []
        total = 0
        limit = min(self.overlap_tokens, budget)
        for unit in reversed(previous):
            if unit.text.startswith("|") or total + unit.tokens > limit:
                break
            carried.insert(0, unit)
            total += unit.tokens
        if carried and len(carried) == len(previous):
            return []  # never duplicate an entire chunk
        return carried

    @staticmethod
    def _join(units: list[_Unit]) -> str:
        parts: list[str] = []
        for i, unit in enumerate(units):
            if i:
                parts.append("\n\n" if unit.new_paragraph else " ")
            parts.append(unit.text)
        return "".join(parts)


def section_path(headings: tuple[str, ...], title: str) -> str | None:
    """Join the heading path, dropping a leading heading that just repeats the title."""
    if headings and headings[0].strip().lower() == title.strip().lower():
        headings = headings[1:]
    return SECTION_SEPARATOR.join(headings) or None


def contextual_header(title: str, section: str | None) -> str:
    """Header prepended to chunk text before embedding / FTS indexing."""
    return f"{title}\nSection: {section}\n\n" if section else f"{title}\n\n"
