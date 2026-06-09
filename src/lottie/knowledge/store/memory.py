"""In-memory cosine-similarity vector store.

This is the reference implementation of ``VectorStore`` for development,
testing, and small corpora (rule of thumb: < ~200 chunks).  It keeps all
``EmbeddedChunk`` objects in a plain Python list and scores queries with
O(n) brute-force cosine similarity.

Design notes
------------
Dimension mismatch
    If a stored chunk's embedding has a different vector length than the query
    embedding, that chunk is **skipped** (excluded from candidates).  This is
    preferable to crashing because corpora may be built incrementally with
    different embedding models; the caller can inspect ``count()`` vs. len(hits)
    to detect silent exclusions if needed.  The alternative (scoring 0.0 and
    including) was rejected because it would pollute ranked results with
    spurious hits.

Tie-breaking
    When two candidates share the same cosine score, they are ordered by
    ``chunk.id`` ascending (lexicographic).  This makes ``query`` output fully
    deterministic regardless of insertion order, which matters for reproducible
    evaluation and snapshot tests.

Tag parsing (forward-looking)
    Phase 1 chunks don't carry structured tags — they live in
    ``metadata["tags"]`` as a comma-separated string.  The split + strip logic
    here is intentionally forward-looking: once chunks gain a structured
    ``tags: list[str]`` field (Phase 2+), the parsing can be replaced with a
    direct attribute access.
"""

from __future__ import annotations

import math

from lottie.knowledge.schema import (
    EmbeddedChunk,
    Embedding,
    KnowledgeLayer,
    RetrievalHit,
)
from lottie.knowledge.store.base import VectorStore

__all__ = ["InMemoryVectorStore"]


def _cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between vectors *a* and *b*.

    Uses ``math.fsum`` for numerically-stable dot product and norm calculations.
    Returns 0.0 if either vector is the zero vector (norm == 0).

    Precondition: ``len(a) == len(b)`` — callers must guard against mismatch.
    """
    dot: float = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm_a: float = math.sqrt(math.fsum(x * x for x in a))
    norm_b: float = math.sqrt(math.fsum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_tags(chunk: EmbeddedChunk) -> set[str]:
    """Extract the tag set from a chunk's metadata.

    Reads ``chunk.chunk.metadata.get("tags", "")`` and splits on ``","``
    with stripping.  Empty strings after stripping are dropped.
    """
    raw: str = chunk.chunk.metadata.get("tags", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


class InMemoryVectorStore(VectorStore):
    """Brute-force in-memory vector store backed by a plain ``list``."""

    def __init__(self) -> None:
        self._chunks: list[EmbeddedChunk] = []

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, items: list[EmbeddedChunk]) -> None:
        """Append *items* to the store (no deduplication)."""
        self._chunks.extend(items)

    def count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    def clear(self) -> None:
        """Remove all chunks from the store."""
        self._chunks.clear()

    def query(
        self,
        embedding: Embedding,
        k: int,
        *,
        layers: list[KnowledgeLayer] | None = None,
        tags: list[str] | None = None,
    ) -> list[RetrievalHit]:
        """Return the top-*k* chunks by cosine similarity, with optional filters.

        See ``VectorStore.query`` for the full contract.
        """
        if k <= 0:
            return []

        query_vec: list[float] = embedding.vector
        query_dim: int = len(query_vec)

        # Build allowed-layer set for O(1) membership tests
        layer_values: set[str] | None = (
            {layer.value for layer in layers} if layers else None
        )
        # Build required-tag set
        tag_set: set[str] | None = (
            set(tags) if tags else None
        )

        scored: list[tuple[float, str, EmbeddedChunk]] = []

        for item in self._chunks:
            # --- dimension guard ---
            if len(item.embedding.vector) != query_dim:
                # Skip silently; see module docstring for rationale.
                continue

            # --- layer filter ---
            if layer_values is not None:
                chunk_layer: str = item.chunk.metadata.get("layer", "")
                if chunk_layer not in layer_values:
                    continue

            # --- tag filter ---
            if tag_set is not None and not _chunk_tags(item).intersection(tag_set):
                continue

            score: float = _cosine(query_vec, item.embedding.vector)
            # Primary sort key: descending score → negate.
            # Secondary (tie-break): ascending chunk.id (lexicographic).
            scored.append((-score, item.chunk.id, item))

        # Sort: (-score, chunk.id) — stable, deterministic.
        scored.sort(key=lambda t: (t[0], t[1]))

        top = scored[:k]
        return [RetrievalHit(chunk=item.chunk, score=-neg_score) for neg_score, _id, item in top]
