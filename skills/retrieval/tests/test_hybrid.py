"""Unit tests for graph-expanded hybrid retrieval in RetrievalSkill.

Covers the ``expand_graph`` flag: when True and a ``GraphStore`` is injected,
base hits are augmented with discounted chunks from depends_on neighbours.
All tests are deterministic: MockEmbeddingProvider + InMemoryVectorStore, no
real LLM, no network.  CLAUDE.md rule 5.
"""

from __future__ import annotations

from lottie.knowledge.embeddings import MockEmbeddingProvider
from lottie.knowledge.graph import GraphStore
from lottie.knowledge.manifest import KnowledgeManifest
from lottie.knowledge.schema import (
    Chunk,
    DocStatus,
    Document,
    EmbeddedChunk,
    KnowledgeLayer,
    RetrievalQuery,
)
from lottie.knowledge.store import InMemoryVectorStore
from skills.retrieval.schema import RetrievalSkillInput
from skills.retrieval.skill import GRAPH_EXPANSION_DISCOUNT, RetrievalSkill

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_store_and_graph() -> (
    tuple[InMemoryVectorStore, MockEmbeddingProvider, GraphStore]
):
    """Return a pre-seeded (store, embedder, graph) triple.

    Two documents:
      - "a": text "alpha apple", no dependencies.
      - "b": text "beta banana", depends_on=["a"].

    Edges: a → b  (a is a dependency of b, so b depends on a).
    GraphStore.neighbors("b") == ["a"].
    """
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    # Manifest
    doc_a = Document(
        id="a",
        source="knowledge/global/a.md",
        layer=KnowledgeLayer.GLOBAL,
        content="alpha apple",
        depends_on=[],
        status=DocStatus.CURATED,
    )
    doc_b = Document(
        id="b",
        source="knowledge/global/b.md",
        layer=KnowledgeLayer.GLOBAL,
        content="beta banana",
        depends_on=["a"],
        status=DocStatus.CURATED,
    )
    manifest = KnowledgeManifest(documents=[doc_a, doc_b])
    graph = GraphStore(manifest)

    # Chunks — doc_id MUST match the manifest document id
    text_a = "alpha apple"
    chunk_a = Chunk(
        id="a#0",
        doc_id="a",
        index=0,
        text=text_a,
        start=0,
        end=len(text_a),
        metadata={"layer": "global", "doc_id": "a"},
    )
    emb_a = embedder.embed([text_a])[0]

    text_b = "beta banana"
    chunk_b = Chunk(
        id="b#0",
        doc_id="b",
        index=0,
        text=text_b,
        start=0,
        end=len(text_b),
        metadata={"layer": "global", "doc_id": "b"},
    )
    emb_b = embedder.embed([text_b])[0]

    store.add([
        EmbeddedChunk(chunk=chunk_a, embedding=emb_a),
        EmbeddedChunk(chunk=chunk_b, embedding=emb_b),
    ])

    return store, embedder, graph


# ---------------------------------------------------------------------------
# Test: expand_graph=True adds the dependency chunk with discounted score
# ---------------------------------------------------------------------------


def test_expand_graph_includes_dependency_chunk() -> None:
    """Querying for 'b' with expand_graph=True includes 'a' (b depends_on a)."""
    store, embedder, graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store, graph)

    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=1, expand_graph=True)
        )
    )

    hit_ids = [h.chunk.id for h in out.result.hits]
    assert "b#0" in hit_ids, "top base hit 'b' should be present"
    assert "a#0" in hit_ids, "expanded neighbour 'a' should be appended"


def test_expand_graph_b_ranks_above_a() -> None:
    """'b' (full-score base hit) ranks above 'a' (discounted expansion hit)."""
    store, embedder, graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store, graph)

    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=1, expand_graph=True)
        )
    )

    hit_ids = [h.chunk.id for h in out.result.hits]
    assert hit_ids.index("b#0") < hit_ids.index("a#0"), (
        "'b' (higher score) must rank before discounted 'a'"
    )


def test_expand_graph_discount_applied_to_a_score() -> None:
    """Expanded 'a' chunk's score == its vector score * GRAPH_EXPANSION_DISCOUNT."""
    store, embedder, graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store, graph)

    # First, get the raw vector score for chunk "a#0" without expansion
    out_no_expand = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=2, expand_graph=False)
        )
    )
    raw_a_score = next(h.score for h in out_no_expand.result.hits if h.chunk.id == "a#0")

    # Now with expansion (k=1 so only "b" is a base hit; "a" is expanded)
    out_expand = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=1, expand_graph=True)
        )
    )
    expanded_a_hit = next(h for h in out_expand.result.hits if h.chunk.id == "a#0")

    assert abs(expanded_a_hit.score - raw_a_score * GRAPH_EXPANSION_DISCOUNT) < 1e-9, (
        f"expected {raw_a_score * GRAPH_EXPANSION_DISCOUNT}, got {expanded_a_hit.score}"
    )


# ---------------------------------------------------------------------------
# Test: expand_graph=False — no expansion
# ---------------------------------------------------------------------------


def test_expand_graph_false_returns_only_base_hits() -> None:
    """expand_graph=False (k=1) returns only the top base hit; no expansion."""
    store, embedder, graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store, graph)

    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=1, expand_graph=False)
        )
    )

    assert len(out.result.hits) == 1
    assert out.result.hits[0].chunk.id == "b#0"


# ---------------------------------------------------------------------------
# Test: no graph injected — behaves like expand_graph=False
# ---------------------------------------------------------------------------


def test_no_graph_injected_expand_true_behaves_like_no_expansion() -> None:
    """RetrievalSkill without a GraphStore: expand_graph=True has no effect."""
    store, embedder, _graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store)  # no graph

    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=1, expand_graph=True)
        )
    )

    assert len(out.result.hits) == 1
    assert out.result.hits[0].chunk.id == "b#0"


def test_no_graph_injected_no_crash() -> None:
    """Calling with expand_graph=True and no GraphStore does not raise."""
    store, embedder, _graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store)

    # Must not raise
    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="alpha apple", k=5, expand_graph=True)
        )
    )
    assert out is not None


# ---------------------------------------------------------------------------
# Test: neighbour already in base hits — no duplicate
# ---------------------------------------------------------------------------


def test_no_duplicate_when_neighbour_already_base_hit() -> None:
    """When k is large enough to include both a and b as base hits, expansion
    adds no new chunks; chunk ids are unique in the result."""
    store, embedder, graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store, graph)

    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=2, expand_graph=True)
        )
    )

    hit_ids = [h.chunk.id for h in out.result.hits]
    # No duplicates
    assert len(hit_ids) == len(set(hit_ids)), "duplicate chunk ids found in result"
    # Both present (both are base hits — no new expansion chunks)
    assert "a#0" in hit_ids
    assert "b#0" in hit_ids
    # Still only 2 — expansion added nothing new
    assert len(hit_ids) == 2


# ---------------------------------------------------------------------------
# Test: metrics instrumentation still works with expansion
# ---------------------------------------------------------------------------


def test_last_metrics_populated_after_expand_run() -> None:
    """skill.last_metrics is set after an expand_graph=True run."""
    store, embedder, graph = _build_store_and_graph()
    skill = RetrievalSkill(embedder, store, graph)

    skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(text="beta banana", k=1, expand_graph=True)
        )
    )

    assert skill.last_metrics is not None
    assert skill.last_metrics.success is True


# ---------------------------------------------------------------------------
# Test: graph expansion crosses layer/tag filters
# ---------------------------------------------------------------------------


def test_expansion_crosses_layer_filter() -> None:
    """Graph expansion returns cross-layer dependency chunks.

    Setup:
      - doc "a": layer=PLATFORM, text "alpha apple", no deps.
      - doc "b": layer=GLOBAL,   text "beta banana", depends_on=["a"].
      - doc "c": layer=PLATFORM, text "alpha cherry", no deps (non-dependency).

    Query with layers=[GLOBAL] and expand_graph=True:
      - Base hits respect the GLOBAL filter: only "b#0" (GLOBAL) is a base hit.
      - Expansion ignores the layer filter: "a#0" (PLATFORM, dependency of b)
        IS included as an expansion hit.
      - Non-dependency PLATFORM doc "c#0" is NOT included (not a neighbour).
    """
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    doc_a = Document(
        id="a",
        source="knowledge/platform/a.md",
        layer=KnowledgeLayer.PLATFORM,
        content="alpha apple",
        depends_on=[],
        status=DocStatus.CURATED,
    )
    doc_b = Document(
        id="b",
        source="knowledge/global/b.md",
        layer=KnowledgeLayer.GLOBAL,
        content="beta banana",
        depends_on=["a"],
        status=DocStatus.CURATED,
    )
    doc_c = Document(
        id="c",
        source="knowledge/platform/c.md",
        layer=KnowledgeLayer.PLATFORM,
        content="alpha cherry",
        depends_on=[],
        status=DocStatus.CURATED,
    )
    manifest = KnowledgeManifest(documents=[doc_a, doc_b, doc_c])
    graph = GraphStore(manifest)

    # Seed store with all three chunks
    text_a = "alpha apple"
    chunk_a = Chunk(
        id="a#0",
        doc_id="a",
        index=0,
        text=text_a,
        start=0,
        end=len(text_a),
        metadata={"layer": "platform", "doc_id": "a"},
    )
    text_b = "beta banana"
    chunk_b = Chunk(
        id="b#0",
        doc_id="b",
        index=0,
        text=text_b,
        start=0,
        end=len(text_b),
        metadata={"layer": "global", "doc_id": "b"},
    )
    text_c = "alpha cherry"
    chunk_c = Chunk(
        id="c#0",
        doc_id="c",
        index=0,
        text=text_c,
        start=0,
        end=len(text_c),
        metadata={"layer": "platform", "doc_id": "c"},
    )
    store.add([
        EmbeddedChunk(chunk=chunk_a, embedding=embedder.embed([text_a])[0]),
        EmbeddedChunk(chunk=chunk_b, embedding=embedder.embed([text_b])[0]),
        EmbeddedChunk(chunk=chunk_c, embedding=embedder.embed([text_c])[0]),
    ])

    skill = RetrievalSkill(embedder, store, graph)
    out = skill.run(
        RetrievalSkillInput(
            query=RetrievalQuery(
                text="beta banana",
                k=1,
                layers=[KnowledgeLayer.GLOBAL],
                expand_graph=True,
            )
        )
    )

    hit_ids = [h.chunk.id for h in out.result.hits]

    # Base hit: GLOBAL doc "b" is present
    assert "b#0" in hit_ids, "base hit 'b' (GLOBAL) must be present"

    # Cross-layer expansion: PLATFORM dependency "a" IS present despite layer filter
    assert "a#0" in hit_ids, (
        "expansion must include PLATFORM dependency 'a' even though layer filter is GLOBAL"
    )

    # Non-dependency PLATFORM doc "c" must NOT appear
    assert "c#0" not in hit_ids, (
        "non-dependency PLATFORM doc 'c' must not be included"
    )
