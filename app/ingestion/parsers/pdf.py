"""PDF parser (PyMuPDF). Keeps page numbers, infers headings from font size, extracts
tables as Markdown and strips repeated page headers/footers."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymupdf

from app.ingestion.cleaning import normalize_text, rows_to_markdown, unwrap_lines
from app.ingestion.types import Block, ParsedDocument, ParseError

_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
_HEADING_SIZE_RATIO = 1.15
_MAX_HEADING_CHARS = 120


@dataclass(slots=True)
class _Line:
    text: str
    size: float
    page: int
    y: float


def _iter_lines(page: Any, page_no: int, skip_rects: list[Any]) -> list[_Line]:
    lines: list[_Line] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            bbox = pymupdf.Rect(line["bbox"])
            if any(bbox.intersects(r) for r in skip_rects):
                continue
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            size = max(s["size"] for s in spans)
            lines.append(_Line(text, round(size, 1), page_no, bbox.y0))
    return lines


def _table_markdown(page: Any, table: Any) -> str:
    """Build cell text from the words whose centre lies inside each cell.

    (`Table.to_markdown()` can mistake the paragraph above a table for an external header
    row, `Table.extract()` drops spaces for some PDF producers, and clipping by cell box
    picks up glyphs of neighbouring rows that touch the shared border.)
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell is None:
                cells.append("")
                continue
            x0, y0, x1, y1 = cell
            inside = [
                w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1
            ]
            inside.sort(key=lambda w: (w[5], w[6], w[7]))
            cells.append(" ".join(w[4] for w in inside))
        rows.append(cells)
    return rows_to_markdown(rows)


def _normalize_running(text: str) -> str:
    """Mask digits so 'Guide - Page 2' and 'Guide - Page 3' count as the same running line."""
    return re.sub(r"\d+", "#", text)


def _repeated_lines(pages: list[list[_Line]]) -> set[str]:
    """Lines that occur on most pages (running headers / footers)."""
    if len(pages) < 3:
        return set()
    seen: Counter[str] = Counter()
    for page_lines in pages:
        seen.update({_normalize_running(ln.text) for ln in page_lines})
    return {text for text, n in seen.items() if n >= max(3, int(len(pages) * 0.6))}


def _heading_levels(lines: list[_Line], body_size: float) -> dict[float, int]:
    sizes = sorted(
        {ln.size for ln in lines if ln.size >= body_size * _HEADING_SIZE_RATIO}, reverse=True
    )
    return {size: level for level, size in enumerate(sizes, start=1)}


def parse_pdf(content: bytes, filename: str) -> ParsedDocument:
    """Parse a PDF into heading-aware blocks with page numbers."""
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ParseError(f"Could not open PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise ParseError("PDF is password protected")
        pages: list[list[_Line]] = []
        tables: dict[int, list[tuple[float, str]]] = {}
        for page_no in range(1, doc.page_count + 1):
            page = doc[page_no - 1]
            found = page.find_tables().tables
            tables[page_no] = [(t.bbox[1], _table_markdown(page, t)) for t in found]
            pages.append(_iter_lines(page, page_no, [pymupdf.Rect(t.bbox) for t in found]))
        num_pages = doc.page_count
        meta_title = (doc.metadata or {}).get("title") or ""

    all_lines = [ln for page_lines in pages for ln in page_lines]
    if not all_lines and not any(tables.values()):
        raise ParseError("PDF contains no extractable text (scanned PDFs need OCR)")

    repeated = _repeated_lines(pages)
    body_size = Counter(ln.size for ln in all_lines).most_common(1)[0][0] if all_lines else 0.0
    levels = _heading_levels(all_lines, body_size)

    blocks: list[Block] = []
    headings: list[str] = []
    paragraph: list[str] = []
    para_page: int | None = None
    first_heading: str | None = None

    def flush() -> None:
        nonlocal paragraph, para_page
        text = unwrap_lines("\n".join(paragraph))
        if text:
            blocks.append(Block(text=text, headings=tuple(headings), page=para_page))
        paragraph, para_page = [], None

    for page_no, page_lines in enumerate(pages, start=1):
        page_tables = sorted(tables.get(page_no, []))
        items: list[tuple[float, _Line | str]] = [(ln.y, ln) for ln in page_lines]
        items += [(y, md) for y, md in page_tables]
        items.sort(key=lambda item: item[0])
        prev_y: float | None = None
        for y, item in items:
            if isinstance(item, str):
                flush()
                blocks.append(
                    Block(text=item, headings=tuple(headings), page=page_no, kind="table")
                )
                prev_y = None
                continue
            line = item
            if _normalize_running(line.text) in repeated or _PAGE_NUMBER.match(line.text):
                continue
            level = levels.get(line.size)
            if level is not None and len(line.text) <= _MAX_HEADING_CHARS:
                flush()
                if first_heading is None:
                    first_heading = line.text
                headings = [*headings[: level - 1], line.text]
                prev_y = None
                continue
            # A vertical gap bigger than ~1 line height starts a new paragraph.
            if prev_y is not None and y - prev_y > line.size * 1.8:
                flush()
            paragraph.append(line.text)
            para_page = para_page or page_no
            prev_y = y
        flush()

    title = normalize_text(meta_title) or first_heading or filename
    return ParsedDocument(title=title, blocks=blocks, num_pages=num_pages)
