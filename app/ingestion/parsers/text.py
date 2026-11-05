"""Markdown, plain-text and HTML parsers."""

import re

from bs4 import BeautifulSoup, Tag

from app.ingestion.cleaning import normalize_text, rows_to_markdown, unwrap_lines
from app.ingestion.types import Block, BlockKind, ParsedDocument, ParseError

_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_FENCE = re.compile(r"^(```|~~~)")


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _classify(text: str) -> BlockKind:
    lines = text.splitlines()
    if all(line.lstrip().startswith("|") for line in lines):
        return "table"
    if all(re.match(r"^\s*([-*+]|\d+[.)])\s+", line) for line in lines):
        return "list"
    return "paragraph"


def _parse_lines(text: str, *, markdown: bool) -> tuple[list[Block], str | None]:
    """Shared line-based parser. Markdown gets ATX (#) headings and code fences;
    both formats get setext (underlined) headings."""
    blocks: list[Block] = []
    headings: list[str] = []
    buffer: list[str] = []
    first_heading: str | None = None
    in_fence = False

    def flush() -> None:
        raw = "\n".join(buffer).strip()
        buffer.clear()
        if not raw:
            return
        kind = "code" if raw.startswith(("```", "~~~")) else _classify(raw)
        body = normalize_text(raw) if kind in ("table", "list", "code") else unwrap_lines(raw)
        blocks.append(Block(text=body, headings=tuple(headings), kind=kind))

    def set_heading(level: int, title: str) -> None:
        nonlocal headings, first_heading
        flush()
        title = normalize_text(title)
        first_heading = first_heading or title
        headings = [*headings[: level - 1], title]

    for line in text.splitlines():
        if markdown and _FENCE.match(line.strip()):
            in_fence = not in_fence
            buffer.append(line)
            if not in_fence:
                flush()
            continue
        if in_fence:
            buffer.append(line)
            continue
        atx = _ATX.match(line) if markdown else None
        if atx:
            set_heading(len(atx.group(1)), atx.group(2))
            continue
        if _SETEXT.match(line) and len(buffer) == 1 and buffer[0].strip():
            title = buffer.pop()
            set_heading(1 if line.strip().startswith("=") else 2, title)
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return blocks, first_heading


def parse_markdown(content: bytes, filename: str) -> ParsedDocument:
    """Parse Markdown, using `#` headings as the section structure."""
    blocks, first_heading = _parse_lines(_decode(content), markdown=True)
    if not blocks:
        raise ParseError("Markdown file contains no text")
    return ParsedDocument(title=first_heading or filename, blocks=blocks)


def parse_text(content: bytes, filename: str) -> ParsedDocument:
    """Parse plain text; underlined (setext-style) lines are treated as headings."""
    blocks, first_heading = _parse_lines(_decode(content), markdown=False)
    if not blocks:
        raise ParseError("Text file is empty")
    return ParsedDocument(title=first_heading or filename, blocks=blocks)


_HTML_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_HTML_BLOCKS = {"p", "li", "pre", "table", "dt", "dd", "blockquote"}


def _html_table(table: Tag) -> str:
    return rows_to_markdown(
        [[c.get_text(" ") for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
    )


def parse_html(content: bytes, filename: str) -> ParsedDocument:
    """Parse HTML in document order: h1-h6 build the section path, p/li/pre/table become blocks."""
    soup = BeautifulSoup(_decode(content), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    title = normalize_text(soup.title.get_text()) if soup.title else ""
    root = soup.body or soup
    headings: list[str] = []
    blocks: list[Block] = []

    for el in root.find_all([*_HTML_HEADINGS, *_HTML_BLOCKS]):
        # Skip elements nested in another captured block (e.g. <p> inside <li>, <td>).
        if el.find_parent([*_HTML_BLOCKS - {"blockquote"}]) is not None:
            continue
        if el.name in _HTML_HEADINGS:
            level = _HTML_HEADINGS[el.name]
            text = normalize_text(el.get_text(" "))
            if text:
                headings = [*headings[: level - 1], text]
                title = title or text
            continue
        if el.name == "table":
            text, kind = _html_table(el), "table"
        elif el.name == "pre":
            text, kind = normalize_text(el.get_text()), "code"
        elif el.name == "li":
            text, kind = "- " + unwrap_lines(el.get_text()), "list"
        else:
            text, kind = unwrap_lines(el.get_text()), "paragraph"
        if text.strip("- "):
            blocks.append(Block(text=text, headings=tuple(headings), kind=kind))  # type: ignore[arg-type]

    if not blocks:
        raise ParseError("HTML contains no text")
    return ParsedDocument(title=title or filename, blocks=blocks)
