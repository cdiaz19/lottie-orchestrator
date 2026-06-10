"""Vector store abstraction, in-memory reference, and optional ChromaDB backend.

Exported names
--------------
VectorStore          — ABC that all backends must implement.
InMemoryVectorStore  — Brute-force in-memory cosine store (dev / testing).
build_vector_store   — Factory: returns the right backend for a given ``kind``.

``ChromaVectorStore`` is **not** re-exported at package level because it
requires the optional ``[chroma]`` extra.  Import it directly when needed::

    from lottie.knowledge.store.chroma import ChromaVectorStore

Factory usage
-------------
>>> from lottie.knowledge.store import build_vector_store
>>> store = build_vector_store("memory", root=Path("."))   # InMemory (always available)
>>> store = build_vector_store("chroma", root=Path("."))   # Chroma (needs [chroma] extra)

Golden Rule: agents never import from this package directly.  All retrieval
goes through a RetrievalSkill that owns the store and enforces policy.
"""

from __future__ import annotations

from pathlib import Path

from lottie.knowledge.store.base import VectorStore
from lottie.knowledge.store.memory import InMemoryVectorStore

__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "build_vector_store",
]


def build_vector_store(kind: str, root: Path) -> VectorStore:
    """Return a ``VectorStore`` instance for the given *kind*.

    Parameters
    ----------
    kind:
        Backend identifier.  Supported values:

        - ``"memory"`` — ``InMemoryVectorStore`` (no persistence, for testing).
        - ``"chroma"`` — ``ChromaVectorStore`` (persistent, ChromaDB).
          Requires the ``[chroma]`` optional extra:
          ``pip install lottie-orchestrator[chroma]``.
    root:
        Project root directory.  Passed to backends that need a filesystem
        path (e.g. ``ChromaVectorStore`` stores data under
        ``root / ".lottie" / "chroma"``).  Ignored by ``InMemoryVectorStore``.

    Raises
    ------
    ValueError
        If *kind* is not a recognised backend identifier.
    ImportError
        If *kind* is ``"chroma"`` and the ``[chroma]`` extra is not installed.
    """
    if kind == "memory":
        return InMemoryVectorStore()
    if kind == "chroma":
        try:
            from lottie.knowledge.store.chroma import ChromaVectorStore
        except ImportError as exc:
            raise ImportError(
                "The 'chroma' backend requires extra dependencies. "
                "Install with: pip install lottie-orchestrator[chroma]"
            ) from exc
        return ChromaVectorStore(root)
    raise ValueError(f"unknown vector store kind: {kind!r}")
