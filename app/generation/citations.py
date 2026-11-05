"""Post-generation citation validation.

The model is told which source ids exist, but nothing forces it to only use those; it can
invent `[S7]` or cite nothing at all. Every citation is checked against the set of ids
actually sent, invalid markers are removed from the text, and an "answerable" answer that
ends up with no valid citation is treated as ungrounded.
"""

import re
from dataclasses import dataclass

_MARKER = re.compile(r"\s?\[(S\d+(?:\s*,\s*S\d+)*)\]")
_ID = re.compile(r"S\d+")


@dataclass(frozen=True, slots=True)
class CitationCheck:
    valid: list[str]
    """Cited ids that exist in the context, in first-mention order."""
    invalid: list[str]
    """Ids the model cited that were never provided (hallucinated references)."""
    answer: str
    """Answer text with invalid inline markers removed."""


def validate_citations(answer: str, cited: list[str], provided_ids: set[str]) -> CitationCheck:
    """Reconcile the `citations` field with inline [S#] markers against the provided ids."""
    ordered: list[str] = []
    invalid: list[str] = []

    def note(source_id: str) -> None:
        bucket = ordered if source_id in provided_ids else invalid
        if source_id not in bucket:
            bucket.append(source_id)

    def clean(match: re.Match[str]) -> str:
        ids = _ID.findall(match.group(1))
        for source_id in ids:
            note(source_id)
        keep = [i for i in ids if i in provided_ids]
        return f" [{', '.join(keep)}]" if keep else ""

    cleaned = _MARKER.sub(clean, answer).strip()
    for source_id in cited:
        note(source_id.strip())
    return CitationCheck(valid=ordered, invalid=invalid, answer=cleaned)
