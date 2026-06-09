"""Deterministic recursive-character chunking for the Lottie knowledge layer.

Pure Python, no LLM. Uses a sliding window with boundary snapping so chunks
always end on a natural separator (double-newline, newline, sentence boundary,
word boundary) rather than in the middle of a token.

Public API::

    from lottie.knowledge.chunking import ChunkConfig, chunk_document
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from lottie.knowledge.schema import Chunk, Document


class ChunkConfig(BaseModel):
    """Configuration for the sliding-window chunker."""

    size: int = 1000
    overlap: int = 200
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]

    @model_validator(mode="after")
    def _validate_window(self) -> ChunkConfig:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if not (0 <= self.overlap < self.size):
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        return self


def _find_boundary(
    text: str,
    start: int,
    end: int,
    separators: list[str],
) -> int | None:
    """Return the index in *text* just after the last occurrence of any separator
    within ``text[start:end]``, scanning *separators* in priority order.

    The empty-string sentinel is skipped (it would always match at every
    position and is meaningless as a boundary).  Returns ``None`` when no
    non-empty separator appears in the window.
    """
    window = text[start:end]
    for sep in separators:
        if not sep:  # skip empty-string sentinel
            continue
        local_index = window.rfind(sep)
        if local_index != -1:
            return start + local_index + len(sep)
    return None


def chunk_document(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    """Slice *doc.content* into overlapping :class:`Chunk` objects.

    Algorithm
    ---------
    Sliding window of width ``cfg.size`` advances by ``cfg.size - cfg.overlap``
    characters per step.  Before emitting each non-final chunk the window end
    is snapped to the last natural separator boundary found within the window,
    so chunk boundaries align with paragraph breaks, sentence ends, or at least
    word breaks whenever possible.

    The algorithm is fully deterministic: identical inputs always produce
    identical outputs.
    """
    text = doc.content
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    idx = 0

    while start < n:
        end = min(start + cfg.size, n)

        if end < n:
            # Not the final chunk — try to snap to a natural boundary.
            # Only accept the snap if the resulting chunk is substantial (at
            # least size // 4 characters).  A tiny snap near the window start
            # would produce a chunk that is a strict subset of the previous
            # chunk, yielding garbage embeddings downstream.
            snapped = _find_boundary(text, start, end, cfg.separators)
            snap_floor = max(1, cfg.size // 4)
            if snapped is not None and snapped - start >= snap_floor:
                end = snapped

        chunk = Chunk(
            id=f"{doc.id}#{idx}",
            doc_id=doc.id,
            index=idx,
            text=text[start:end],
            start=start,
            end=end,
            metadata={"layer": doc.layer.value, "doc_id": doc.id},
        )
        chunks.append(chunk)
        idx += 1

        if end >= n:
            break

        # Guarantee forward progress even when overlap equals size.
        start = max(end - cfg.overlap, start + 1)

    return chunks
