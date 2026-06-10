"""Abstract base class for vector stores used in the Lottie knowledge subsystem.

Golden Rule: agents never call the store directly.  They go through a
RetrievalSkill that owns the store reference and enforces policy (security,
layer/tag ACLs, cost limits).  The store itself has no knowledge of agents.

Concrete implementations:
- ``InMemoryVectorStore``  — development / testing, in-process, no persistence.
- (Task 8) ChromaStore     — persistent, suited for corpora > ~200 files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottie.knowledge.schema import (
    EmbeddedChunk,
    Embedding,
    KnowledgeLayer,
    RetrievalHit,
)

__all__ = ["VectorStore"]


class VectorStore(ABC):
    """Protocol that every vector-store backend must satisfy.

    Parameters accepted by ``query`` follow *keyword-only* defaults that are
    ``None`` rather than mutable containers (``[]``) to avoid Python's
    mutable-default pitfall.  ``None`` means "no filter" (all candidates pass).
    An explicit empty list ``[]`` is also treated as "no filter" by convention —
    callers that want to guarantee zero results should use ``k=0``.
    """

    @abstractmethod
    def add(self, items: list[EmbeddedChunk]) -> None:
        """Persist *items* to the store.

        Implementations may deduplicate by ``chunk.id`` or simply append;
        callers are responsible for not adding the same chunk twice.
        """

    @abstractmethod
    def query(
        self,
        embedding: Embedding,
        k: int,
        *,
        layers: list[KnowledgeLayer] | None = None,
        tags: list[str] | None = None,
    ) -> list[RetrievalHit]:
        """Return the top-*k* chunks most similar to *embedding*.

        Parameters
        ----------
        embedding:
            Query vector.  Behavior on dimension mismatch between the query
            and stored vectors is backend-defined.
        k:
            Maximum number of hits to return.  ``k <= 0`` always returns ``[]``.
        layers:
            If non-empty, only chunks whose ``metadata["layer"]`` value matches
            one of the provided ``KnowledgeLayer`` values are considered.
            ``None`` or ``[]`` disables the filter.
        tags:
            If non-empty, only chunks whose ``metadata["tags"]`` CSV string
            intersects the provided tags are considered.
            ``None`` or ``[]`` disables the filter.
            Note: backends that store tags as CSV metadata (e.g. Chroma) cannot
            rely on server-side substring filtering — the ``tags`` filter must
            be applied client-side after fetching candidates.

        Returns
        -------
        list[RetrievalHit]
            Hits sorted by descending cosine score.  Ties are broken
            deterministically (ascending ``chunk.id``).
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently in the store."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from the store."""
