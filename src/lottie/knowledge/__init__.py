"""Lottie knowledge subsystem — schemas, frontmatter parsing, manifest loader, and graph store.

Public surface::

    from lottie.knowledge import (
        KnowledgeManifest,
        KnowledgeLayer,
        DocStatus,
        Document,
        Chunk,
        Embedding,
        EmbeddedChunk,
        RetrievalQuery,
        RetrievalHit,
        RetrievalResult,
        GraphStore,
        build_knowledge_graph,
    )
"""

from __future__ import annotations

from lottie.knowledge.graph import GraphStore, build_knowledge_graph
from lottie.knowledge.manifest import KnowledgeManifest
from lottie.knowledge.schema import (
    Chunk,
    DocStatus,
    Document,
    EmbeddedChunk,
    Embedding,
    KnowledgeLayer,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
)

__all__ = [
    "KnowledgeManifest",
    "KnowledgeLayer",
    "DocStatus",
    "Document",
    "Chunk",
    "Embedding",
    "EmbeddedChunk",
    "RetrievalQuery",
    "RetrievalHit",
    "RetrievalResult",
    "GraphStore",
    "build_knowledge_graph",
]
