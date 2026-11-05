"""Parser registry: picks a parser by MIME type, falling back to the file extension."""

from collections.abc import Callable
from pathlib import PurePath

from app.ingestion.parsers.docx import parse_docx
from app.ingestion.parsers.pdf import parse_pdf
from app.ingestion.parsers.text import parse_html, parse_markdown, parse_text
from app.ingestion.types import ParsedDocument, UnsupportedFormatError

Parser = Callable[[bytes, str], ParsedDocument]

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_BY_MIME: dict[str, tuple[str, Parser]] = {
    "application/pdf": ("application/pdf", parse_pdf),
    DOCX_MIME: (DOCX_MIME, parse_docx),
    "text/markdown": ("text/markdown", parse_markdown),
    "text/x-markdown": ("text/markdown", parse_markdown),
    "text/plain": ("text/plain", parse_text),
    "text/html": ("text/html", parse_html),
    "application/xhtml+xml": ("text/html", parse_html),
}

_BY_EXTENSION: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": DOCX_MIME,
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
}

SUPPORTED_EXTENSIONS = tuple(_BY_EXTENSION)


def resolve_parser(filename: str, content_type: str | None) -> tuple[str, Parser]:
    """Return `(canonical_mime_type, parser)`.

    The extension wins over generic MIME types such as `application/octet-stream` or
    `text/plain` (browsers and curl often send those for .md files).
    """
    ext = PurePath(filename).suffix.lower()
    mime = (content_type or "").split(";")[0].strip().lower()
    if ext in _BY_EXTENSION:
        return _BY_MIME[_BY_EXTENSION[ext]]
    if mime in _BY_MIME:
        return _BY_MIME[mime]
    raise UnsupportedFormatError(
        f"Unsupported file type '{ext or mime or 'unknown'}'. "
        f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
