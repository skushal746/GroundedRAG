"""DOCX parser (python-docx). Walks the body in order, keeping Heading N structure and
rendering tables as Markdown so row/column relationships survive chunking."""

import io
import re

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.cleaning import normalize_text, rows_to_markdown
from app.ingestion.types import Block, BlockKind, ParsedDocument, ParseError

_HEADING_STYLE = re.compile(r"^Heading\s*(\d)$", re.IGNORECASE)


def _table_to_markdown(table: Table) -> str:
    return rows_to_markdown([[cell.text for cell in row.cells] for row in table.rows])


def parse_docx(content: bytes, filename: str) -> ParsedDocument:
    """Parse a DOCX into heading-aware blocks."""
    try:
        document = docx.Document(io.BytesIO(content))
    except Exception as exc:
        raise ParseError(f"Could not open DOCX: {exc}") from exc

    title = normalize_text(document.core_properties.title or "")
    headings: list[str] = []
    blocks: list[Block] = []

    for item in document.iter_inner_content():
        if isinstance(item, Table):
            md = _table_to_markdown(item)
            if md:
                blocks.append(Block(text=md, headings=tuple(headings), kind="table"))
            continue
        assert isinstance(item, Paragraph)
        text = normalize_text(item.text)
        if not text:
            continue
        style = item.style.name if item.style is not None else ""
        if style == "Title":
            title = title or text
            continue
        match = _HEADING_STYLE.match(style or "")
        if match:
            level = int(match.group(1))
            headings = [*headings[: level - 1], text]
            continue
        kind: BlockKind = "list" if "List" in (style or "") else "paragraph"
        if kind == "list":
            text = f"- {text}"
        blocks.append(Block(text=text, headings=tuple(headings), kind=kind))

    if not blocks:
        raise ParseError("DOCX contains no text")
    return ParsedDocument(title=title or (headings[0] if headings else filename), blocks=blocks)
