"""Intermediate representation shared by all parsers and the chunker."""

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["paragraph", "table", "list", "code"]


@dataclass(frozen=True, slots=True)
class Block:
    """A contiguous piece of body text with the structure it was found under.

    `headings` is the full heading path, e.g. ("Billing", "Refunds").
    `page` is 1-based and only known for paginated formats (PDF).
    """

    text: str
    headings: tuple[str, ...] = ()
    page: int | None = None
    kind: BlockKind = "paragraph"


@dataclass(slots=True)
class ParsedDocument:
    title: str
    blocks: list[Block] = field(default_factory=list)
    num_pages: int | None = None


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk ready to embed. `embed_text` carries a contextual header (title + section)."""

    chunk_index: int
    text: str
    embed_text: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    token_count: int


class UnsupportedFormatError(ValueError):
    """Raised for file types we don't parse (mapped to HTTP 415)."""


class ParseError(ValueError):
    """Raised when a supported file cannot be parsed (corrupt, empty, encrypted...)."""
