"""Tests for lottie.knowledge.index — index_manifest helper.

TDD Red-Green: tests written before implementation.
No real LLM, no real embedder, no network (CLAUDE.md rule 5).
"""

from __future__ import annotations

from lottie.knowledge.embeddings import MockEmbeddingProvider
from lottie.knowledge.index import index_manifest
from lottie.knowledge.manifest import KnowledgeManifest
from lottie.knowledge.schema import DocStatus, Document, KnowledgeLayer
from lottie.knowledge.store import InMemoryVectorStore


def _make_manifest(n_docs: int) -> KnowledgeManifest:
    docs = [
        Document(
            id=f"doc/{i}",
            source=f"knowledge/global/doc{i}.md",
            layer=KnowledgeLayer.GLOBAL,
            content=f"This is content for document number {i}. " * 10,
            status=DocStatus.CURATED,
        )
        for i in range(n_docs)
    ]
    return KnowledgeManifest(documents=docs)


def test_index_manifest_returns_positive_chunk_count() -> None:
    """index_manifest over a small manifest returns > 0 chunks."""
    manifest = _make_manifest(2)
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    count = index_manifest(manifest, embedder, store)

    assert count > 0


def test_index_manifest_store_count_matches_return_value() -> None:
    """store.count() equals the value returned by index_manifest."""
    manifest = _make_manifest(2)
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    count = index_manifest(manifest, embedder, store)

    assert store.count() == count


def test_index_manifest_query_returns_chunk_from_doc0() -> None:
    """A subsequent store.query for doc0's text returns a chunk from doc0."""
    manifest = _make_manifest(2)
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    index_manifest(manifest, embedder, store)

    doc0 = manifest.documents[0]
    # embed the doc0 content to produce a query
    query_embedding = embedder.embed([doc0.content])[0]
    hits = store.query(query_embedding, k=1)

    assert len(hits) == 1
    assert hits[0].chunk.doc_id == doc0.id


def test_index_manifest_empty_manifest_returns_zero() -> None:
    """Empty manifest produces 0 chunks and nothing in the store."""
    manifest = KnowledgeManifest(documents=[])
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    count = index_manifest(manifest, embedder, store)

    assert count == 0
    assert store.count() == 0


def test_index_manifest_respects_custom_chunk_config() -> None:
    """Passing a ChunkConfig with small size produces more chunks than the default."""
    from lottie.knowledge.chunking import ChunkConfig

    manifest = _make_manifest(1)
    embedder = MockEmbeddingProvider()

    store_default = InMemoryVectorStore()
    store_small = InMemoryVectorStore()

    # Default config (size=1000) vs tiny config (size=50, overlap=0)
    index_manifest(manifest, embedder, store_default)
    index_manifest(manifest, embedder, store_small, config=ChunkConfig(size=50, overlap=0))

    # Small chunks → more of them
    assert store_small.count() > store_default.count()
