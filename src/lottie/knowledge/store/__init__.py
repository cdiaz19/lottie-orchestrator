"""Vector store abstraction, in-memory reference, and ChromaDB persistent backend.

Exported names
--------------
VectorStore          — ABC that all backends must implement.
InMemoryVectorStore  — Brute-force in-memory cosine store (dev / testing).
ChromaVectorStore    — Persistent ChromaDB backend for corpora > ~200 files.
build_vector_store   — Factory: returns the right backend for a given ``kind``.

Factory usage
-------------
>>> from lottie.knowledge.store import build_vector_store
>>> store = build_vector_store("memory", root=Path("."))   # InMemory
>>> store = build_vector_store("chroma", root=Path("."))   # Chroma

Golden Rule: agents never import from this package directly.  All retrieval
goes through a RetrievalSkill that owns the store and enforces policy.
"""

from __future__ import annotations

from pathlib import Path

from lottie.knowledge.store.base import VectorStore
from lottie.knowledge.store.chroma import ChromaVectorStore
from lottie.knowledge.store.memory import InMemoryVectorStore

__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
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
    root:
        Project root directory.  Passed to backends that need a filesystem
        path (e.g. ``ChromaVectorStore`` stores data under
        ``root / ".lottie" / "chroma"``).  Ignored by ``InMemoryVectorStore``.

    Raises
    ------
    ValueError
        If *kind* is not a recognised backend identifier.
    """
    if kind == "memory":
        return InMemoryVectorStore()
    if kind == "chroma":
        return ChromaVectorStore(root)
    raise ValueError(f"unknown vector store kind: {kind!r}")
