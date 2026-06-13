"""Tests for ChromaVectorStore — TDD (RED → GREEN).

Uses ``tmp_path`` for each test so the persistent Chroma database is fully
isolated.  ``MockEmbeddingProvider`` produces deterministic 16-dim L2-normalised
vectors, which makes similarity assertions meaningful without real models.

Guard
-----
The entire module is skipped if chromadb is somehow absent so that CI
never fails on an environment where chromadb isn't installed.
"""

from __future__ import annotations

from typing import Any

import pytest

chromadb = pytest.importorskip("chromadb")  # skip module if chromadb absent

from pathlib import Path  # noqa: E402

from lottie.knowledge.embeddings.mock import MockEmbeddingProvider  # noqa: E402
from lottie.knowledge.schema import (  # noqa: E402
    Chunk,
    EmbeddedChunk,
    Embedding,
    KnowledgeLayer,
    RetrievalHit,
)
from lottie.knowledge.store import VectorStore  # noqa: E402
from lottie.knowledge.store.chroma import ChromaVectorStore  # noqa: E402

_PROVIDER = MockEmbeddingProvider(dim=16)


def _embed(text: str) -> Embedding:
    """Return a deterministic 16-dim L2-normalised embedding for *text*."""
    return _PROVIDER.embed([text])[0]


def make_chunk(
    chunk_id: str,
    text: str,
    layer: str = "global",
    tags: str = "",
    doc_id: str | None = None,
) -> EmbeddedChunk:
    """Build a minimal ``EmbeddedChunk`` for testing."""
    _doc_id = doc_id if doc_id is not None else f"doc_{chunk_id}"
    metadata: dict[str, str] = {"layer": layer, "doc_id": _doc_id}
    if tags:
        metadata["tags"] = tags
    return EmbeddedChunk(
        chunk=Chunk(
            id=chunk_id,
            doc_id=_doc_id,
            index=0,
            text=text,
            start=0,
            end=len(text),
            metadata=metadata,
        ),
        embedding=_embed(text),
    )



class TestChromaIsVectorStore:
    """ChromaVectorStore must satisfy the VectorStore ABC."""

    def test_is_subclass(self) -> None:
        assert issubclass(ChromaVectorStore, VectorStore)

    def test_instance_is_vector_store(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        assert isinstance(store, VectorStore)



class TestBasicQuery:
    """add → query → correct top hit, lossless Chunk reconstruction."""

    def test_top_hit_is_apple_and_round_trip(self, tmp_path: Path) -> None:
        """Top hit for 'apple' query is the apple chunk; reconstructed Chunk is lossless."""
        store = ChromaVectorStore(tmp_path)
        apple = make_chunk("apple", "apple", doc_id="doc_apple")
        banana = make_chunk("banana", "banana", doc_id="doc_banana")
        cherry = make_chunk("cherry", "cherry", doc_id="doc_cherry")
        store.add([apple, banana, cherry])

        hits = store.query(_embed("apple"), k=2)

        assert len(hits) == 2
        assert isinstance(hits[0], RetrievalHit)

        top = hits[0]
        # Top hit must be apple
        assert top.chunk.id == "apple"

        # Lossless round-trip: all Chunk fields must match exactly
        original_chunk = apple.chunk
        assert top.chunk.doc_id == original_chunk.doc_id
        assert top.chunk.text == original_chunk.text
        assert top.chunk.start == original_chunk.start
        assert top.chunk.end == original_chunk.end
        assert top.chunk.index == original_chunk.index
        assert top.chunk.metadata == original_chunk.metadata

    def test_returns_list_of_retrieval_hits(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple"), make_chunk("b", "banana")])
        hits = store.query(_embed("apple"), k=2)
        assert isinstance(hits, list)
        assert all(isinstance(h, RetrievalHit) for h in hits)

    def test_scores_descending(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("apple", "apple"),
            make_chunk("banana", "banana"),
            make_chunk("cherry", "cherry"),
        ])
        hits = store.query(_embed("apple"), k=3)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_self_match_scores_approximately_one(self, tmp_path: Path) -> None:
        """Querying a chunk's own text should return a score ≈ 1.0."""
        store = ChromaVectorStore(tmp_path)
        emb = _embed("apple")
        chunk = make_chunk("apple", "apple")
        store.add([chunk])
        hits = store.query(emb, k=1)
        assert len(hits) == 1
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    def test_k_larger_than_corpus_returns_all(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple"), make_chunk("b", "banana")])
        hits = store.query(_embed("apple"), k=100)
        assert len(hits) == 2



class TestCountAndClear:
    """count() and clear() contract."""

    def test_empty_count_is_zero(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        assert store.count() == 0

    def test_count_after_add(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple"), make_chunk("b", "banana")])
        assert store.count() == 2

    def test_count_accumulates(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple")])
        store.add([make_chunk("b", "banana")])
        assert store.count() == 2

    def test_clear_resets_count(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple"), make_chunk("b", "banana")])
        store.clear()
        assert store.count() == 0

    def test_query_after_clear_returns_empty(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple")])
        store.clear()
        hits = store.query(_embed("apple"), k=5)
        assert hits == []

    def test_add_empty_list_does_not_raise(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([])  # must not call Chroma (Chroma errors on empty batch)
        assert store.count() == 0



class TestKEdgeCases:
    def test_k_zero_returns_empty(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple")])
        assert store.query(_embed("apple"), k=0) == []

    def test_k_negative_returns_empty(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple")])
        assert store.query(_embed("apple"), k=-1) == []



class TestLayerFilter:
    """query(layers=[...]) applies a server-side Chroma where filter."""

    def test_global_only(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("g1", "global context", layer="global"),
            make_chunk("d1", "draft context", layer="draft"),
        ])
        hits = store.query(_embed("context"), k=10, layers=[KnowledgeLayer.GLOBAL])
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"g1"}

    def test_draft_only(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("g1", "global context", layer="global"),
            make_chunk("d1", "draft context", layer="draft"),
        ])
        hits = store.query(_embed("context"), k=10, layers=[KnowledgeLayer.DRAFT])
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"d1"}

    def test_multiple_layers(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("g1", "global context", layer="global"),
            make_chunk("d1", "draft context", layer="draft"),
        ])
        hits = store.query(
            _embed("context"),
            k=10,
            layers=[KnowledgeLayer.GLOBAL, KnowledgeLayer.DRAFT],
        )
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"g1", "d1"}

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("g1", "global context", layer="global")])
        hits = store.query(_embed("context"), k=10, layers=[KnowledgeLayer.PLATFORM])
        assert hits == []

    def test_none_returns_all(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("g1", "global context", layer="global"),
            make_chunk("d1", "draft context", layer="draft"),
        ])
        hits = store.query(_embed("context"), k=10, layers=None)
        assert len(hits) == 2

    def test_empty_list_returns_all(self, tmp_path: Path) -> None:
        """layers=[] is treated as no filter."""
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("g1", "global context", layer="global"),
            make_chunk("d1", "draft context", layer="draft"),
        ])
        hits = store.query(_embed("context"), k=10, layers=[])
        assert len(hits) == 2



class TestTagFilter:
    """tags filter is applied client-side after Chroma fetch."""

    def test_jwt_includes_auth_chunk(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("auth1", "auth jwt token", tags="auth,jwt"),
            make_chunk("bill1", "billing invoice", tags="billing,payments"),
        ])
        hits = store.query(_embed("auth"), k=10, tags=["jwt"])
        chunk_ids = {h.chunk.id for h in hits}
        assert "auth1" in chunk_ids
        assert "bill1" not in chunk_ids

    def test_billing_includes_billing_chunk(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("auth1", "auth jwt token", tags="auth,jwt"),
            make_chunk("bill1", "billing invoice", tags="billing,payments"),
        ])
        hits = store.query(_embed("billing"), k=10, tags=["billing"])
        chunk_ids = {h.chunk.id for h in hits}
        assert "bill1" in chunk_ids

    def test_unknown_tag_returns_empty(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("auth1", "auth jwt token", tags="auth,jwt")])
        hits = store.query(_embed("auth"), k=10, tags=["unknown"])
        assert hits == []

    def test_no_tag_chunk_excluded_when_filter_active(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("auth1", "auth text", tags="auth"),
            make_chunk("notag", "plain text"),
        ])
        hits = store.query(_embed("text"), k=10, tags=["auth"])
        chunk_ids = {h.chunk.id for h in hits}
        assert "notag" not in chunk_ids

    def test_tags_none_returns_all(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("auth1", "auth text", tags="auth"),
            make_chunk("notag", "plain text"),
        ])
        hits = store.query(_embed("text"), k=10, tags=None)
        assert len(hits) == 2

    def test_tags_empty_list_returns_all(self, tmp_path: Path) -> None:
        """tags=[] is treated as no filter."""
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("auth1", "auth text", tags="auth"),
            make_chunk("notag", "plain text"),
        ])
        hits = store.query(_embed("text"), k=10, tags=[])
        assert len(hits) == 2

    def test_tag_whitespace_stripped(self, tmp_path: Path) -> None:
        """Tags in metadata may have spaces around commas — they should still match."""
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("ws1", "whitespace tags", tags=" auth , jwt ")])
        hits = store.query(_embed("auth"), k=5, tags=["jwt"])
        assert len(hits) == 1

    def test_empty_string_tag_is_no_filter(self, tmp_path: Path) -> None:
        """tags=[''] must be treated the same as tags=None."""
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("auth1", "auth text", tags="auth"),
            make_chunk("notag", "plain text"),
        ])
        hits_none = store.query(_embed("text"), k=10, tags=None)
        hits_empty = store.query(_embed("text"), k=10, tags=[""])
        assert {h.chunk.id for h in hits_empty} == {h.chunk.id for h in hits_none}



class TestCombinedFilters:
    def test_layer_and_tag_combined(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("g-auth", "global auth", layer="global", tags="auth"),
            make_chunk("d-auth", "draft auth", layer="draft", tags="auth"),
            make_chunk("g-plain", "global plain", layer="global"),
        ])
        hits = store.query(
            _embed("auth"), k=10, layers=[KnowledgeLayer.GLOBAL], tags=["auth"]
        )
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"g-auth"}


class TestScores:
    def test_scores_are_similarity_not_distance(self, tmp_path: Path) -> None:
        """Chroma returns distance; we must convert to similarity = 1 - distance."""
        store = ChromaVectorStore(tmp_path)
        emb = _embed("apple")
        store.add([make_chunk("apple", "apple")])
        hits = store.query(emb, k=1)
        # Self-match cosine similarity should be close to 1.0, not 0.0
        assert hits[0].score > 0.9

    def test_scores_roughly_in_valid_range(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        store.add([
            make_chunk("a", "apple"),
            make_chunk("b", "banana"),
            make_chunk("c", "cherry"),
        ])
        hits = store.query(_embed("apple"), k=3)
        for h in hits:
            # Cosine similarity is in [-1, 1]; allow small floating-point slack
            assert -1.0 - 1e-6 <= h.score <= 1.0 + 1e-6


class TestCosineSpaceVerification:
    """_open_collection() must assert hnsw:space == 'cosine'."""

    def test_normal_construct_yields_cosine_collection(self, tmp_path: Path) -> None:
        """A freshly created store must have a cosine collection."""
        store = ChromaVectorStore(tmp_path)
        assert (store._collection.metadata or {}).get("hnsw:space") == "cosine"

    def test_clear_preserves_cosine_space(self, tmp_path: Path) -> None:
        """After clear(), the recreated collection must still be cosine."""
        store = ChromaVectorStore(tmp_path)
        store.add([make_chunk("a", "apple")])
        store.clear()
        assert (store._collection.metadata or {}).get("hnsw:space") == "cosine"

    def test_pre_existing_l2_collection_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        """Constructing a ChromaVectorStore against a pre-existing non-cosine
        collection must raise RuntimeError with a helpful message."""
        import chromadb as _chromadb

        persist_dir = tmp_path / ".lottie" / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        # Create the collection with l2 space directly via the raw client.
        raw_client: Any = _chromadb.PersistentClient(path=str(persist_dir))
        raw_client.create_collection(
            name="lottie_knowledge",
            metadata={"hnsw:space": "l2"},
        )

        with pytest.raises(RuntimeError, match="hnsw:space=.l2."):
            ChromaVectorStore(tmp_path)


class TestReservedKeyGuard:
    """add() must reject chunk.metadata containing '_chunk_json'."""

    def test_reserved_key_raises_value_error(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path)
        bad_item = make_chunk("x", "text")
        # Inject the reserved key directly into the chunk's metadata.
        bad_item.chunk.metadata["_chunk_json"] = "should not be allowed"
        with pytest.raises(ValueError, match="reserved key '_chunk_json'"):
            store.add([bad_item])

    def test_normal_metadata_does_not_raise(self, tmp_path: Path) -> None:
        """Metadata without the reserved key must be accepted normally."""
        store = ChromaVectorStore(tmp_path)
        item = make_chunk("y", "text", tags="safe")
        store.add([item])  # must not raise
        assert store.count() == 1


class TestBuildVectorStore:
    """build_vector_store factory returns the right backend."""

    def test_memory_kind(self, tmp_path: Path) -> None:
        from lottie.knowledge.store import InMemoryVectorStore, build_vector_store

        store = build_vector_store("memory", tmp_path)
        assert isinstance(store, InMemoryVectorStore)

    def test_chroma_kind(self, tmp_path: Path) -> None:
        from lottie.knowledge.store import VectorStore, build_vector_store

        store = build_vector_store("chroma", tmp_path)
        assert isinstance(store, VectorStore)
        assert isinstance(store, ChromaVectorStore)

    def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        from lottie.knowledge.store import build_vector_store

        with pytest.raises(ValueError, match="unknown vector store kind"):
            build_vector_store("redis", tmp_path)
