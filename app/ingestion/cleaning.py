"""Text normalisation applied by every parser."""

import re
import unicodedata
from collections.abc import Sequence

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<!\n)\n(?!\n)")
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """NFKC-normalise, drop control chars and collapse whitespace while keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines inside a paragraph (PDF/TXT line wrapping)."""
    return _SOFT_BREAK.sub(" ", normalize_text(text))


def rows_to_markdown(rows: Sequence[Sequence[str | None]]) -> str:
    """Render table rows as a Markdown table (first row = header). Cells are normalised and
    flattened to one line so the row structure survives chunking and embedding."""
    clean = [[normalize_text(c or "").replace("\n", " ") for c in row] for row in rows]
    clean = [row for row in clean if any(row)]
    if not clean:
        return ""
    header, *body = clean
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)
