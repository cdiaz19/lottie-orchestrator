"""Vector store abstraction and in-memory reference implementation.

Exported names
--------------
VectorStore          — ABC that all backends must implement.
InMemoryVectorStore  — Brute-force in-memory cosine store (dev / testing).

Not included here (YAGNI)
-------------------------
- ChromaStore / persistent backend  → Task 8
- store factory function            → Task 8 (after Chroma backend exists)

Golden Rule: agents never import from this package directly.  All retrieval
goes through a RetrievalSkill that owns the store and enforces policy.
"""

from __future__ import annotations

from lottie.knowledge.store.base import VectorStore
from lottie.knowledge.store.memory import InMemoryVectorStore

__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
]
