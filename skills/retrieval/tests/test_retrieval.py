"""Unit tests for RetrievalSkill — embed-query → vector-store retrieve.

All tests are deterministic: MockEmbeddingProvider + InMemoryVectorStore, no
real LLM, no network.  CLAUDE.md rule 5.
"""

from __future__ import annotations

from lottie.knowledge.embeddings import MockEmbeddingProvider
from lottie.knowledge.schema import (
    Chunk,
    EmbeddedChunk,
    KnowledgeLayer,
    RetrievalQuery,
)
from lottie.knowledge.store import InMemoryVectorStore
from skills.retrieval.schema import RetrievalSkillInput, RetrievalSkillOutput
from skills.retrieval.skill import RetrievalSkill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEXTS = ["apple pie", "banana bread", "cherry tart"]


def _make_store_and_embedder() -> tuple[InMemoryVectorStore, MockEmbeddingProvider]:
    """Return a fresh (store, embedder) pair pre-seeded with three global chunks."""
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    chunks: list[EmbeddedChunk] = []
    for i, text in enumerate(_TEXTS):
        chunk = Chunk(
            id=f"d{i}#0",
            doc_id=f"d{i}",
            index=0,
            text=text,
            start=0,
            end=len(text),
            metadata={"layer": "global", "doc_id": f"d{i}"},
        )
        emb = embedder.embed([text])[0]
        chunks.append(EmbeddedChunk(chunk=chunk, embedding=emb))

    store.add(chunks)
    return store, embedder


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_returns_retrieval_skill_output_type() -> None:
    """run() returns an instance of RetrievalSkillOutput."""
    store, embedder = _make_store_and_embedder()
    skill = RetrievalSkill(embedder, store)
    out = skill.run(RetrievalSkillInput(query=RetrievalQuery(text="apple pie", k=2)))
    assert isinstance(out, RetrievalSkillOutput)


def test_top_hit_is_self_match() -> None:
    """The query text 'apple pie' should rank its own chunk first (cosine == 1.0)."""
    store, embedder = _make_store_and_embedder()
    skill = RetrievalSkill(embedder, store)
    out = skill.run(RetrievalSkillInput(query=RetrievalQuery(text="apple pie", k=2)))
    assert len(out.result.hits) == 2
    assert out.result.hits[0].chunk.text == "apple pie"
    assert out.result.hits[0].chunk.id == "d0#0"


def test_k_limits_number_of_hits() -> None:
    """k=1 returns exactly 1 hit even when the store has more chunks."""
    store, embedder = _make_store_and_embedder()
    skill = RetrievalSkill(embedder, store)
    out = skill.run(RetrievalSkillInput(query=RetrievalQuery(text="cherry tart", k=1)))
    assert len(out.result.hits) == 1


def test_k_equal_to_store_size_returns_all() -> None:
    """k >= store size returns all available chunks."""
    store, embedder = _make_store_and_embedder()
    skill = RetrievalSkill(embedder, store)
    out = skill.run(RetrievalSkillInput(query=RetrievalQuery(text="banana bread", k=10)))
    assert len(out.result.hits) == 3  # only 3 chunks in store


# ---------------------------------------------------------------------------
# Metrics instrumentation
# ---------------------------------------------------------------------------


def test_last_metrics_populated_after_run() -> None:
    """skill.last_metrics is set after a successful run (InstrumentedRunnable contract)."""
    store, embedder = _make_store_and_embedder()
    skill = RetrievalSkill(embedder, store)
    skill.run(RetrievalSkillInput(query=RetrievalQuery(text="apple pie", k=2)))
    assert skill.last_metrics is not None
    assert skill.last_metrics.success is True
    assert skill.last_metrics.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Layer filter test
# ---------------------------------------------------------------------------


def test_layer_filter_excludes_other_layers() -> None:
    """Querying with layers=[GLOBAL] should exclude draft-layer chunks."""
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    # One global chunk
    global_text = "global knowledge doc"
    global_chunk = Chunk(
        id="g0#0",
        doc_id="g0",
        index=0,
        text=global_text,
        start=0,
        end=len(global_text),
        metadata={"layer": "global", "doc_id": "g0"},
    )
    global_emb = embedder.embed([global_text])[0]

    # One draft chunk
    draft_text = "draft knowledge doc"
    draft_chunk = Chunk(
        id="dr0#0",
        doc_id="dr0",
        index=0,
        text=draft_text,
        start=0,
        end=len(draft_text),
        metadata={"layer": "draft", "doc_id": "dr0"},
    )
    draft_emb = embedder.embed([draft_text])[0]

    store.add([
        EmbeddedChunk(chunk=global_chunk, embedding=global_emb),
        EmbeddedChunk(chunk=draft_chunk, embedding=draft_emb),
    ])

    skill = RetrievalSkill(embedder, store)
    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(
                text="knowledge doc",
                k=5,
                layers=[KnowledgeLayer.GLOBAL],
            )
        )
    )

    returned_ids = {h.chunk.id for h in out.result.hits}
    assert "g0#0" in returned_ids, "global chunk should be included"
    assert "dr0#0" not in returned_ids, "draft chunk must be excluded by layer filter"


def test_empty_layers_means_no_filter() -> None:
    """layers=[] (default) passes both global and draft chunks through."""
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    for layer_val, idx in [("global", 0), ("draft", 1)]:
        text = f"{layer_val} text"
        chunk = Chunk(
            id=f"c{idx}#0",
            doc_id=f"c{idx}",
            index=0,
            text=text,
            start=0,
            end=len(text),
            metadata={"layer": layer_val, "doc_id": f"c{idx}"},
        )
        emb = embedder.embed([text])[0]
        store.add([EmbeddedChunk(chunk=chunk, embedding=emb)])

    skill = RetrievalSkill(embedder, store)
    # Default layers=[] → no filter
    out = skill.run(RetrievalSkillInput(query=RetrievalQuery(text="text", k=5)))
    assert len(out.result.hits) == 2
