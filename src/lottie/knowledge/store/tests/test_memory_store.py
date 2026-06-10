"""Tests for InMemoryVectorStore — TDD (RED → GREEN).

All tests are written first; implementation follows only after these are observed
to fail with ImportError / AttributeError (not with silent passes).

Helper convention
-----------------
``make_chunk(id, text, layer, tags)`` builds a minimal ``EmbeddedChunk`` whose
embedding vector comes from ``MockEmbeddingProvider``.  We keep dim=16 throughout
so that dimension-mismatch tests can craft a mismatched embedding by hand.
"""

from __future__ import annotations

import math

import pytest

from lottie.knowledge.embeddings.mock import MockEmbeddingProvider
from lottie.knowledge.schema import (
    Chunk,
    EmbeddedChunk,
    Embedding,
    KnowledgeLayer,
    RetrievalHit,
)
from lottie.knowledge.store import InMemoryVectorStore, VectorStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PROVIDER = MockEmbeddingProvider(dim=16)


def _embed(text: str) -> Embedding:
    """Return a deterministic 16-dim L2-normalised embedding for *text*."""
    return _PROVIDER.embed([text])[0]


def make_chunk(
    chunk_id: str,
    text: str,
    layer: str = "global",
    tags: str = "",
) -> EmbeddedChunk:
    """Build an ``EmbeddedChunk`` with the given id, text, layer, and tags metadata."""
    metadata: dict[str, str] = {"layer": layer, "doc_id": f"doc_{chunk_id}"}
    if tags:
        metadata["tags"] = tags
    return EmbeddedChunk(
        chunk=Chunk(
            id=chunk_id,
            doc_id=f"doc_{chunk_id}",
            index=0,
            text=text,
            start=0,
            end=len(text),
            metadata=metadata,
        ),
        embedding=_embed(text),
    )


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreIsVectorStore:
    """InMemoryVectorStore must satisfy the VectorStore ABC."""

    def test_is_subclass(self) -> None:
        assert issubclass(InMemoryVectorStore, VectorStore)

    def test_instance_is_vector_store(self) -> None:
        assert isinstance(InMemoryVectorStore(), VectorStore)


class TestCountAndClear:
    """count() and clear() basic contract."""

    def setup_method(self) -> None:
        self.store = InMemoryVectorStore()

    def test_empty_store_count_is_zero(self) -> None:
        assert self.store.count() == 0

    def test_count_after_add(self) -> None:
        self.store.add([make_chunk("a", "apple"), make_chunk("b", "banana")])
        assert self.store.count() == 2

    def test_count_accumulates_across_adds(self) -> None:
        self.store.add([make_chunk("a", "apple")])
        self.store.add([make_chunk("b", "banana")])
        assert self.store.count() == 2

    def test_clear_resets_count(self) -> None:
        self.store.add([make_chunk("a", "apple"), make_chunk("b", "banana")])
        self.store.clear()
        assert self.store.count() == 0

    def test_query_on_empty_store_returns_empty(self) -> None:
        result = self.store.query(_embed("anything"), k=5)
        assert result == []


# ---------------------------------------------------------------------------
# Basic query
# ---------------------------------------------------------------------------


class TestBasicQuery:
    """Smoke tests for query without filters."""

    def setup_method(self) -> None:
        self.store = InMemoryVectorStore()
        self.chunks = [
            make_chunk("apple", "apple"),
            make_chunk("banana", "banana"),
            make_chunk("cherry", "cherry"),
        ]
        self.store.add(self.chunks)

    def test_query_returns_list_of_retrieval_hits(self) -> None:
        hits = self.store.query(_embed("apple"), k=2)
        assert isinstance(hits, list)
        assert all(isinstance(h, RetrievalHit) for h in hits)

    def test_query_returns_k_hits_when_corpus_is_large_enough(self) -> None:
        hits = self.store.query(_embed("apple"), k=2)
        assert len(hits) == 2

    def test_top_hit_is_apple_when_querying_apple(self) -> None:
        """The most similar chunk to 'apple' embedding should be the 'apple' chunk."""
        hits = self.store.query(_embed("apple"), k=2)
        assert hits[0].chunk.id == "apple"

    def test_scores_are_in_descending_order(self) -> None:
        hits = self.store.query(_embed("apple"), k=3)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_in_valid_cosine_range(self) -> None:
        """Cosine similarity is in [-1, 1]."""
        hits = self.store.query(_embed("apple"), k=3)
        for h in hits:
            assert -1.0 - 1e-9 <= h.score <= 1.0 + 1e-9

    def test_k_larger_than_corpus_returns_all_chunks(self) -> None:
        hits = self.store.query(_embed("apple"), k=100)
        assert len(hits) == 3

    def test_k_zero_returns_empty(self) -> None:
        hits = self.store.query(_embed("apple"), k=0)
        assert hits == []

    def test_k_negative_returns_empty(self) -> None:
        hits = self.store.query(_embed("apple"), k=-1)
        assert hits == []

    def test_query_after_clear_returns_empty(self) -> None:
        self.store.clear()
        hits = self.store.query(_embed("apple"), k=5)
        assert hits == []


# ---------------------------------------------------------------------------
# Layer filter
# ---------------------------------------------------------------------------


class TestLayerFilter:
    """query(layers=[...]) must restrict results to matching layer values."""

    def setup_method(self) -> None:
        self.store = InMemoryVectorStore()
        self.global_chunk = make_chunk("g1", "global context", layer="global")
        self.draft_chunk = make_chunk("d1", "draft context", layer="draft")
        self.store.add([self.global_chunk, self.draft_chunk])

    def test_layer_filter_global_returns_only_global(self) -> None:
        hits = self.store.query(_embed("context"), k=10, layers=[KnowledgeLayer.GLOBAL])
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"g1"}

    def test_layer_filter_draft_returns_only_draft(self) -> None:
        hits = self.store.query(_embed("context"), k=10, layers=[KnowledgeLayer.DRAFT])
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"d1"}

    def test_layer_filter_multiple_layers(self) -> None:
        hits = self.store.query(
            _embed("context"), k=10, layers=[KnowledgeLayer.GLOBAL, KnowledgeLayer.DRAFT]
        )
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"g1", "d1"}

    def test_layer_filter_no_match_returns_empty(self) -> None:
        hits = self.store.query(_embed("context"), k=10, layers=[KnowledgeLayer.PLATFORM])
        assert hits == []

    def test_layer_filter_none_returns_all(self) -> None:
        """layers=None means no filter — all chunks are candidates."""
        hits = self.store.query(_embed("context"), k=10, layers=None)
        assert len(hits) == 2

    def test_layer_filter_empty_list_returns_all(self) -> None:
        """layers=[] is treated the same as layers=None (no filter)."""
        hits = self.store.query(_embed("context"), k=10, layers=[])
        assert len(hits) == 2


# ---------------------------------------------------------------------------
# Tag filter
# ---------------------------------------------------------------------------


class TestTagFilter:
    """query(tags=[...]) must restrict results to chunks whose tags intersect."""

    def setup_method(self) -> None:
        self.store = InMemoryVectorStore()
        self.auth_chunk = make_chunk("auth1", "auth jwt token", tags="auth,jwt")
        self.billing_chunk = make_chunk("bill1", "billing invoice", tags="billing,payments")
        self.no_tag_chunk = make_chunk("notag", "plain text")
        self.store.add([self.auth_chunk, self.billing_chunk, self.no_tag_chunk])

    def test_tag_filter_jwt_includes_auth_chunk(self) -> None:
        hits = self.store.query(_embed("auth"), k=10, tags=["jwt"])
        chunk_ids = {h.chunk.id for h in hits}
        assert "auth1" in chunk_ids

    def test_tag_filter_jwt_excludes_billing_chunk(self) -> None:
        hits = self.store.query(_embed("auth"), k=10, tags=["jwt"])
        chunk_ids = {h.chunk.id for h in hits}
        assert "bill1" not in chunk_ids

    def test_tag_filter_billing_includes_billing_chunk(self) -> None:
        hits = self.store.query(_embed("billing"), k=10, tags=["billing"])
        chunk_ids = {h.chunk.id for h in hits}
        assert "bill1" in chunk_ids

    def test_tag_filter_unknown_tag_returns_empty(self) -> None:
        hits = self.store.query(_embed("auth"), k=10, tags=["unknown-tag"])
        assert hits == []

    def test_tag_filter_no_tag_chunk_excluded_when_filter_active(self) -> None:
        hits = self.store.query(_embed("text"), k=10, tags=["auth"])
        chunk_ids = {h.chunk.id for h in hits}
        assert "notag" not in chunk_ids

    def test_tag_filter_none_returns_all(self) -> None:
        hits = self.store.query(_embed("text"), k=10, tags=None)
        assert len(hits) == 3

    def test_tag_filter_empty_list_returns_all(self) -> None:
        """tags=[] is treated the same as tags=None (no filter)."""
        hits = self.store.query(_embed("text"), k=10, tags=[])
        assert len(hits) == 3

    def test_tags_are_stripped_of_whitespace(self) -> None:
        """Tags in metadata may have spaces around commas — they should still match."""
        chunk = make_chunk("ws1", "whitespace tags", tags=" auth , jwt ")
        store = InMemoryVectorStore()
        store.add([chunk])
        hits = store.query(_embed("auth"), k=5, tags=["jwt"])
        assert len(hits) == 1

    def test_tag_filter_empty_string_is_no_filter(self) -> None:
        """tags=[''] must be treated the same as tags=None — no filter applied."""
        hits_none = self.store.query(_embed("text"), k=10, tags=None)
        hits_empty_str = self.store.query(_embed("text"), k=10, tags=[""])
        assert {h.chunk.id for h in hits_empty_str} == {h.chunk.id for h in hits_none}

    def test_tag_filter_whitespace_only_strings_are_no_filter(self) -> None:
        """tags=['  '] must be treated the same as tags=None — no filter applied."""
        hits_none = self.store.query(_embed("text"), k=10, tags=None)
        hits_ws = self.store.query(_embed("text"), k=10, tags=["  "])
        assert {h.chunk.id for h in hits_ws} == {h.chunk.id for h in hits_none}


# ---------------------------------------------------------------------------
# Combined layer + tag filter
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    """Both filters must apply simultaneously (AND semantics)."""

    def setup_method(self) -> None:
        self.store = InMemoryVectorStore()
        # global layer, auth tags
        self.g_auth = make_chunk("g-auth", "global auth", layer="global", tags="auth")
        # draft layer, auth tags
        self.d_auth = make_chunk("d-auth", "draft auth", layer="draft", tags="auth")
        # global layer, no matching tags
        self.g_plain = make_chunk("g-plain", "global plain", layer="global")
        self.store.add([self.g_auth, self.d_auth, self.g_plain])

    def test_layer_and_tag_filter(self) -> None:
        hits = self.store.query(
            _embed("auth"), k=10, layers=[KnowledgeLayer.GLOBAL], tags=["auth"]
        )
        chunk_ids = {h.chunk.id for h in hits}
        assert chunk_ids == {"g-auth"}


# ---------------------------------------------------------------------------
# Dimension mismatch
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    """Chunks with mismatched embedding dimension must be skipped, not crash."""

    def test_mismatch_does_not_raise(self) -> None:
        store = InMemoryVectorStore()
        # Build a chunk with a 4-dim embedding manually (provider uses dim=16)
        mismatched_emb = Embedding(vector=[0.5, 0.5, 0.5, 0.5], model="mock/embed", dim=4)
        mismatch_chunk = EmbeddedChunk(
            chunk=Chunk(
                id="mis",
                doc_id="doc_mis",
                index=0,
                text="mismatch",
                start=0,
                end=8,
                metadata={"layer": "global", "doc_id": "doc_mis"},
            ),
            embedding=mismatched_emb,
        )
        store.add([mismatch_chunk])
        # 16-dim query against 4-dim stored chunk — must not raise
        result = store.query(_embed("anything"), k=5)
        # The mismatched chunk is excluded from results
        assert all(h.chunk.id != "mis" for h in result)

    def test_mismatch_chunk_excluded_other_chunks_returned(self) -> None:
        store = InMemoryVectorStore()
        good = make_chunk("good", "apple")
        mismatched_emb = Embedding(vector=[1.0, 0.0], model="mock/embed", dim=2)
        bad = EmbeddedChunk(
            chunk=Chunk(
                id="bad",
                doc_id="doc_bad",
                index=0,
                text="bad",
                start=0,
                end=3,
                metadata={"layer": "global", "doc_id": "doc_bad"},
            ),
            embedding=mismatched_emb,
        )
        store.add([good, bad])
        hits = store.query(_embed("apple"), k=5)
        # good chunk is returned, bad is skipped
        assert len(hits) == 1
        assert hits[0].chunk.id == "good"


# ---------------------------------------------------------------------------
# Determinism (tie-break)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Identical-score chunks must return in a stable, defined order."""

    def test_identical_score_chunks_stable_order(self) -> None:
        """Two chunks with the same embedding vector appear in chunk.id ascending order."""
        # Craft two chunks with the exact same vector so cosine = 1.0 for both
        shared_vec = [1.0] + [0.0] * 15  # unit vector along dim-0
        shared_emb_a = Embedding(vector=shared_vec, model="mock/embed", dim=16)
        shared_emb_b = Embedding(vector=shared_vec, model="mock/embed", dim=16)
        chunk_a = EmbeddedChunk(
            chunk=Chunk(
                id="a_chunk",
                doc_id="doc_a",
                index=0,
                text="text_a",
                start=0,
                end=6,
                metadata={"layer": "global", "doc_id": "doc_a"},
            ),
            embedding=shared_emb_a,
        )
        chunk_b = EmbeddedChunk(
            chunk=Chunk(
                id="b_chunk",
                doc_id="doc_b",
                index=0,
                text="text_b",
                start=0,
                end=6,
                metadata={"layer": "global", "doc_id": "doc_b"},
            ),
            embedding=shared_emb_b,
        )
        # Same query vector
        query_emb = Embedding(vector=shared_vec, model="mock/embed", dim=16)

        store1 = InMemoryVectorStore()
        store1.add([chunk_b, chunk_a])  # add in reverse order to confirm sorting
        hits1 = store1.query(query_emb, k=2)

        store2 = InMemoryVectorStore()
        store2.add([chunk_a, chunk_b])
        hits2 = store2.query(query_emb, k=2)

        # Both runs must yield the same order (a_chunk before b_chunk — id ascending)
        ids1 = [h.chunk.id for h in hits1]
        ids2 = [h.chunk.id for h in hits2]
        assert ids1 == ids2
        assert ids1 == ["a_chunk", "b_chunk"]


# ---------------------------------------------------------------------------
# Cosine similarity correctness
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Spot-check cosine calculation with hand-crafted vectors."""

    def test_identical_unit_vectors_score_one(self) -> None:
        """cos(v, v) = 1.0 for any non-zero v."""
        store = InMemoryVectorStore()
        vec = [1.0] + [0.0] * 15
        emb = Embedding(vector=vec, model="mock/embed", dim=16)
        chunk = EmbeddedChunk(
            chunk=Chunk(
                id="same",
                doc_id="d",
                index=0,
                text="t",
                start=0,
                end=1,
                metadata={"layer": "global", "doc_id": "d"},
            ),
            embedding=emb,
        )
        store.add([chunk])
        hits = store.query(emb, k=1)
        assert hits[0].score == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors_score_zero(self) -> None:
        """cos(e1, e2) = 0.0 for orthogonal unit vectors."""
        store = InMemoryVectorStore()
        vec_a = [1.0] + [0.0] * 15
        vec_b = [0.0, 1.0] + [0.0] * 14
        emb_a = Embedding(vector=vec_a, model="mock/embed", dim=16)
        emb_b = Embedding(vector=vec_b, model="mock/embed", dim=16)
        chunk = EmbeddedChunk(
            chunk=Chunk(
                id="ortho",
                doc_id="d",
                index=0,
                text="t",
                start=0,
                end=1,
                metadata={"layer": "global", "doc_id": "d"},
            ),
            embedding=emb_b,
        )
        store.add([chunk])
        hits = store.query(emb_a, k=1)
        assert hits[0].score == pytest.approx(0.0, abs=1e-9)

    def test_zero_norm_candidate_scores_zero(self) -> None:
        """A zero-vector candidate is included in results with score 0.0, ranking last."""
        store = InMemoryVectorStore()
        zero_vec = [0.0] * 16
        emb_zero = Embedding(vector=zero_vec, model="mock/embed", dim=16)
        chunk_zero = EmbeddedChunk(
            chunk=Chunk(
                id="zero",
                doc_id="d",
                index=0,
                text="t",
                start=0,
                end=1,
                metadata={"layer": "global", "doc_id": "d"},
            ),
            embedding=emb_zero,
        )
        store.add([chunk_zero])
        # Query with a proper unit vector
        query_emb = Embedding(vector=[1.0] + [0.0] * 15, model="mock/embed", dim=16)
        hits = store.query(query_emb, k=5)
        # The zero-norm chunk is included (not skipped) with score 0.0.
        assert isinstance(hits, list)
        zero_hits = [h for h in hits if h.chunk.id == "zero"]
        assert len(zero_hits) == 1
        assert zero_hits[0].score == pytest.approx(0.0)

    def test_zero_query_norm_returns_all_zero_scores(self) -> None:
        """A zero-query vector produces score 0.0 for all candidates (not a crash)."""
        store = InMemoryVectorStore()
        store.add([make_chunk("x", "apple")])
        zero_query = Embedding(vector=[0.0] * 16, model="mock/embed", dim=16)
        hits = store.query(zero_query, k=5)
        assert isinstance(hits, list)
        for h in hits:
            assert h.score == pytest.approx(0.0, abs=1e-9)

    def test_cosine_is_numerically_stable_with_fsum(self) -> None:
        """math.fsum prevents catastrophic cancellation for near-unit vectors."""
        store = InMemoryVectorStore()
        # Build a near-unit vector with many small values (stress test fsum)
        n = 16
        val = 1.0 / math.sqrt(n)
        vec = [val] * n
        emb = Embedding(vector=vec, model="mock/embed", dim=n)
        chunk = EmbeddedChunk(
            chunk=Chunk(
                id="stable",
                doc_id="d",
                index=0,
                text="t",
                start=0,
                end=1,
                metadata={"layer": "global", "doc_id": "d"},
            ),
            embedding=emb,
        )
        store.add([chunk])
        hits = store.query(emb, k=1)
        assert hits[0].score == pytest.approx(1.0, abs=1e-9)
