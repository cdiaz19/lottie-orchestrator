"""Tests for lottie.knowledge.graph — GraphStore and build_knowledge_graph.

All tests use in-memory KnowledgeManifest/Document instances — no filesystem I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lottie.knowledge.graph import GraphStore, build_knowledge_graph
from lottie.knowledge.manifest import KnowledgeManifest
from lottie.knowledge.schema import DocStatus, Document, KnowledgeLayer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    doc_id: str,
    depends_on: list[str] | None = None,
    last_verified: str | None = None,
) -> Document:
    """Build a minimal Document for graph tests."""
    fm: dict[str, str] = {}
    if last_verified is not None:
        fm["last_verified"] = last_verified
    return Document(
        id=doc_id,
        source=f"knowledge/global/{doc_id}.md",
        layer=KnowledgeLayer.GLOBAL,
        content="body",
        frontmatter=fm,
        depends_on=depends_on or [],
        status=DocStatus.CURATED,
    )


def _manifest(*docs: Document) -> KnowledgeManifest:
    return KnowledgeManifest(list(docs))


# ---------------------------------------------------------------------------
# Fixtures (module-level for re-use)
# ---------------------------------------------------------------------------

# Chain: a <- b <- c  (edges: a->b, b->c)
@pytest.fixture()
def chain_store() -> GraphStore:
    m = _manifest(_doc("a"), _doc("b", depends_on=["a"]), _doc("c", depends_on=["b"]))
    return GraphStore(m)


@pytest.fixture()
def cycle_store() -> GraphStore:
    m = _manifest(
        _doc("a"),
        _doc("b", depends_on=["a"]),
        _doc("c", depends_on=["b"]),
        # x <-> y cycle
        _doc("x", depends_on=["y"]),
        _doc("y", depends_on=["x"]),
    )
    return GraphStore(m)


# ---------------------------------------------------------------------------
# build_knowledge_graph — basic structure
# ---------------------------------------------------------------------------


class TestBuildKnowledgeGraph:
    def test_nodes_present(self) -> None:
        m = _manifest(_doc("a"), _doc("b", depends_on=["a"]))
        g = build_knowledge_graph(m)
        assert "a" in g.nodes
        assert "b" in g.nodes

    def test_edge_orientation_dep_to_dependent(self) -> None:
        """Edge goes FROM dependency TO dependent: a -> b when b depends_on a."""
        m = _manifest(_doc("a"), _doc("b", depends_on=["a"]))
        g = build_knowledge_graph(m)
        assert g.has_edge("a", "b"), "edge a->b must exist (b depends_on a)"
        assert not g.has_edge("b", "a"), "reverse edge must NOT exist"

    def test_node_attrs_stored(self) -> None:
        m = _manifest(_doc("a", last_verified="2025-06-01"))
        g = build_knowledge_graph(m)
        attrs = g.nodes["a"]
        assert attrs["layer"] == KnowledgeLayer.GLOBAL.value
        assert attrs["status"] == DocStatus.CURATED.value
        assert attrs["last_verified"] == "2025-06-01"

    def test_dangling_dep_node_created(self) -> None:
        """A dep referenced in depends_on but not in documents still gets a node."""
        m = _manifest(_doc("b", depends_on=["ghost"]))
        g = build_knowledge_graph(m)
        assert "ghost" in g.nodes
        assert g.has_edge("ghost", "b")


# ---------------------------------------------------------------------------
# GraphStore.graph property
# ---------------------------------------------------------------------------


class TestGraphProperty:
    def test_graph_is_digraph(self, chain_store: GraphStore) -> None:
        import networkx as nx

        assert isinstance(chain_store.graph, nx.DiGraph)


# ---------------------------------------------------------------------------
# GraphStore.neighbors — direct deps (predecessors in dep->dependent graph)
# ---------------------------------------------------------------------------


class TestNeighbors:
    def test_c_depends_on_b(self, chain_store: GraphStore) -> None:
        assert chain_store.neighbors("c") == ["b"]

    def test_root_has_no_deps(self, chain_store: GraphStore) -> None:
        assert chain_store.neighbors("a") == []

    def test_absent_node_returns_empty(self, chain_store: GraphStore) -> None:
        assert chain_store.neighbors("nope") == []

    def test_result_is_sorted(self) -> None:
        m = _manifest(
            _doc("z"),
            _doc("a"),
            _doc("target", depends_on=["z", "a"]),
        )
        store = GraphStore(m)
        assert store.neighbors("target") == ["a", "z"]


# ---------------------------------------------------------------------------
# GraphStore.impact — transitive dependents (descendants in dep->dependent graph)
# ---------------------------------------------------------------------------


class TestImpact:
    def test_impact_of_root_is_all_dependents(self, chain_store: GraphStore) -> None:
        assert chain_store.impact("a") == ["b", "c"]

    def test_impact_of_leaf_is_empty(self, chain_store: GraphStore) -> None:
        assert chain_store.impact("c") == []

    def test_impact_of_middle_node(self, chain_store: GraphStore) -> None:
        assert chain_store.impact("b") == ["c"]

    def test_absent_node_returns_empty(self, chain_store: GraphStore) -> None:
        assert chain_store.impact("nope") == []

    def test_result_is_sorted(self) -> None:
        # root -> z and root -> a (both depend on root)
        m = _manifest(_doc("root"), _doc("z", depends_on=["root"]), _doc("a", depends_on=["root"]))
        store = GraphStore(m)
        assert store.impact("root") == ["a", "z"]


# ---------------------------------------------------------------------------
# GraphStore.cycles
# ---------------------------------------------------------------------------


class TestCycles:
    def test_no_cycles_in_chain(self, chain_store: GraphStore) -> None:
        assert chain_store.cycles() == []

    def test_xy_cycle_detected(self, cycle_store: GraphStore) -> None:
        cycles = cycle_store.cycles()
        # At minimum one cycle involving x and y
        assert any(set(c) == {"x", "y"} for c in cycles), (
            f"expected x/y cycle in {cycles}"
        )

    def test_cycles_deterministic(self, cycle_store: GraphStore) -> None:
        """Calling cycles() twice returns identical result."""
        assert cycle_store.cycles() == cycle_store.cycles()

    def test_self_loop_is_a_cycle(self) -> None:
        m = _manifest(_doc("self_ref", depends_on=["self_ref"]))
        store = GraphStore(m)
        cycles = store.cycles()
        assert any("self_ref" in c for c in cycles)


# ---------------------------------------------------------------------------
# GraphStore.orphans
# ---------------------------------------------------------------------------


class TestOrphans:
    def test_lonely_is_orphan(self) -> None:
        m = _manifest(
            _doc("a"),
            _doc("b", depends_on=["a"]),
            _doc("c", depends_on=["b"]),
            _doc("lonely"),
        )
        store = GraphStore(m)
        assert "lonely" in store.orphans()

    def test_chain_nodes_not_orphans(self, chain_store: GraphStore) -> None:
        orphans = chain_store.orphans()
        assert "a" not in orphans
        assert "b" not in orphans
        assert "c" not in orphans

    def test_orphans_sorted(self) -> None:
        m = _manifest(_doc("z_lone"), _doc("a_lone"))
        store = GraphStore(m)
        result = store.orphans()
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# GraphStore.stale
# ---------------------------------------------------------------------------


class TestStale:
    _NOW = datetime(2026, 6, 1, tzinfo=UTC)

    def test_old_doc_is_stale(self) -> None:
        m = _manifest(_doc("old", last_verified="2025-01-01"))
        store = GraphStore(m)
        result = store.stale(days=90, now=self._NOW)
        assert "old" in result

    def test_recent_doc_not_stale(self) -> None:
        # 10 days before NOW → not stale for days=90
        m = _manifest(_doc("fresh", last_verified="2026-05-22"))
        store = GraphStore(m)
        result = store.stale(days=90, now=self._NOW)
        assert "fresh" not in result

    def test_missing_last_verified_not_stale(self) -> None:
        m = _manifest(_doc("no_date"))  # no last_verified in frontmatter
        store = GraphStore(m)
        result = store.stale(days=90, now=self._NOW)
        assert "no_date" not in result

    def test_yyyymm_format_treated_as_first_of_month(self) -> None:
        # "2025-01" → 2025-01-01, which is 151 days before 2026-06-01
        m = _manifest(_doc("monthly", last_verified="2025-01"))
        store = GraphStore(m)
        assert "monthly" in store.stale(days=90, now=self._NOW)

    def test_unparseable_last_verified_not_stale(self) -> None:
        m = _manifest(_doc("bad_date", last_verified="not-a-date"))
        store = GraphStore(m)
        result = store.stale(days=90, now=self._NOW)
        assert "bad_date" not in result

    def test_stale_result_sorted(self) -> None:
        m = _manifest(
            _doc("z_old", last_verified="2024-01-01"),
            _doc("a_old", last_verified="2024-01-01"),
        )
        store = GraphStore(m)
        result = store.stale(days=90, now=self._NOW)
        assert result == sorted(result)

    def test_exactly_on_boundary_not_stale(self) -> None:
        """A doc verified exactly `days` days before now is NOT stale (strict <)."""
        boundary = self._NOW - timedelta(days=90)
        m = _manifest(_doc("boundary", last_verified=boundary.strftime("%Y-%m-%d")))
        store = GraphStore(m)
        # verified_date == now - days → NOT stale (strict <)
        assert "boundary" not in store.stale(days=90, now=self._NOW)
