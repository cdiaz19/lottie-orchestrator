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
        IngestSource,
        DocumentIngestInput,
        DocumentIngestOutput,
        DocumentIngestSkill,
        index_manifest,
        resolve_embedding_settings,
    )
"""

from __future__ import annotations

import os

from lottie.knowledge.graph import GraphStore, build_knowledge_graph
from lottie.knowledge.index import index_manifest
from lottie.knowledge.ingest import (
    DocumentIngestInput,
    DocumentIngestOutput,
    DocumentIngestSkill,
    IngestSource,
)
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


def resolve_embedding_settings() -> tuple[str, str]:
    """Return ``(embedding_model, vector_store_kind)`` from env with Phase-1 defaults.

    Defaults: ``mock/embed`` (no API key) and ``memory`` (in-process store).
    Production deployments override via ``LOTTIE_EMBEDDING_MODEL`` and
    ``LOTTIE_VECTOR_STORE``.  CLI flags (``--embedder``, ``--store``) take
    precedence — callers pass explicit values directly and ignore this helper.
    """
    return (
        os.environ.get("LOTTIE_EMBEDDING_MODEL", "mock/embed"),
        os.environ.get("LOTTIE_VECTOR_STORE", "memory"),
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
    "IngestSource",
    "DocumentIngestInput",
    "DocumentIngestOutput",
    "DocumentIngestSkill",
    "index_manifest",
    "resolve_embedding_settings",
]
