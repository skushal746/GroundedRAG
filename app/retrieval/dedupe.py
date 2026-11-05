"""Context de-duplication.

Adjacent chunks of the same section share ~80 tokens of overlap; sending both wastes
context and makes the model cite the same fact twice under different IDs. We merge
adjacent chunks into one source (stitching out the overlap) and drop exact duplicates.
"""

import re

from app.retrieval.types import Candidate

_WS = re.compile(r"\s+")
_MIN_OVERLAP_CHARS = 20


def stitch(first: str, second: str) -> str:
    """Concatenate two consecutive chunks, removing the longest suffix/prefix overlap."""
    for size in range(min(len(first), len(second)), _MIN_OVERLAP_CHARS - 1, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return first + "\n\n" + second


def dedupe_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Walk candidates in rank order and return up to `limit` distinct sources.

    Exact duplicate texts are dropped; a chunk adjacent to an already-kept chunk of the same
    section is merged into it (it doesn't use up a slot). A merged source keeps the position
    and scores of its highest-ranked member. Stops once `limit` sources are kept, so only
    chunks ranked above the last kept source can be merged in.
    """
    kept: list[Candidate] = []
    spans: list[set[int]] = []  # chunk_index values merged into kept[i]
    seen: set[str] = set()
    for cand in candidates:
        if len(kept) == limit:
            break
        key = _WS.sub(" ", cand.text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        for target, span in zip(kept, spans, strict=True):
            if (
                target.document_id == cand.document_id
                and target.section_path == cand.section_path
                and (cand.chunk_index - 1 in span or cand.chunk_index + 1 in span)
            ):
                _merge_into(target, cand, prepend=cand.chunk_index < min(span))
                span.add(cand.chunk_index)
                break
        else:
            kept.append(cand)
            spans.append({cand.chunk_index})
    return kept


def _merge_into(target: Candidate, other: Candidate, *, prepend: bool) -> None:
    header = target.embed_text[: len(target.embed_text) - len(target.text)]
    target.text = stitch(other.text, target.text) if prepend else stitch(target.text, other.text)
    target.embed_text = header + target.text
    target.merged_chunk_ids.append(other.chunk_id)
    pages = [p for p in (target.page_start, target.page_end, other.page_start, other.page_end) if p]
    if pages:
        target.page_start, target.page_end = min(pages), max(pages)
