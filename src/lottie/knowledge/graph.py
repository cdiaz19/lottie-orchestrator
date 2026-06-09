"""NetworkX-backed dependency graph for the Lottie knowledge layer.

Edge orientation (spec §15)
---------------------------
Edges point **FROM dependency TO dependent**: if document ``B`` has
``depends_on: [A]``, the graph contains the edge ``A -> B``.

This lets us answer two key questions with standard nx primitives:

* *What does doc X depend on?*  → predecessors of X (nodes that have an edge
  **into** X, i.e. nodes X depends on).
* *What breaks if X is deprecated?*  → ``nx.descendants(g, X)`` — all nodes
  reachable from X following the forward edges, i.e. everything that (directly
  or transitively) depends on X.

Example::

    manifest = KnowledgeManifest.from_root(Path("."))
    store = GraphStore(manifest)
    print(store.impact("lottie/auth-conventions"))
"""

from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx  # type stubs provided by types-networkx

from lottie.knowledge.manifest import KnowledgeManifest


def build_knowledge_graph(manifest: KnowledgeManifest) -> nx.DiGraph[str]:
    """Build a directed dependency graph from *manifest*.

    Each document becomes a node (keyed by ``doc.id``) carrying ``layer``,
    ``status``, and ``last_verified`` attributes.  For every entry in
    ``doc.depends_on`` an edge is added ``dep -> doc.id``  (dep *before* the
    dependent in dependency-resolution order).

    Parameters
    ----------
    manifest:
        Loaded :class:`~lottie.knowledge.manifest.KnowledgeManifest`.

    Returns
    -------
    nx.DiGraph[str]
        Directed graph; nodes are document ids (strings).
    """
    g: nx.DiGraph[str] = nx.DiGraph()
    for doc in manifest.documents:
        g.add_node(
            doc.id,
            layer=doc.layer.value,
            status=doc.status.value,
            last_verified=doc.frontmatter.get("last_verified", ""),
        )
        for dep in doc.depends_on:
            # dep -> doc.id  (dependency points to the dependent)
            g.add_edge(dep, doc.id, rel="depends_on")
    return g


class GraphStore:
    """Query interface over the knowledge dependency graph.

    Builds a :class:`networkx.DiGraph` from a
    :class:`~lottie.knowledge.manifest.KnowledgeManifest` on construction and
    exposes high-level query methods used by ``lottie memory`` commands.

    Edge orientation: dependency → dependent (see module docstring).
    """

    def __init__(self, manifest: KnowledgeManifest) -> None:
        self._graph: nx.DiGraph[str] = build_knowledge_graph(manifest)

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph[str]:
        """The underlying :class:`networkx.DiGraph` (read-only by convention)."""
        return self._graph

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def neighbors(self, doc_id: str) -> list[str]:
        """Direct dependencies of *doc_id* — the documents it depends on.

        Because edges run *dep → dependent*, the docs that ``doc_id`` depends on
        are its **predecessors** in the graph.

        Returns a sorted list so callers get deterministic output.
        Returns ``[]`` (not an error) when *doc_id* is not in the graph.
        """
        if doc_id not in self._graph:
            return []
        return sorted(self._graph.predecessors(doc_id))

    def impact(self, doc_id: str) -> list[str]:
        """Documents that (transitively) depend on *doc_id*.

        Answers "what breaks if *doc_id* is deprecated?" by returning all
        nodes reachable from *doc_id* following forward edges
        (``nx.descendants``).

        Returns a sorted list.  Returns ``[]`` when *doc_id* is absent.
        """
        if doc_id not in self._graph:
            return []
        return sorted(nx.descendants(self._graph, doc_id))

    def cycles(self) -> list[list[str]]:
        """All simple cycles in the graph.

        Uses :func:`networkx.simple_cycles`.  Each cycle is normalised to start
        at its lexicographically smallest element so that the output is stable
        across runs.  The outer list is sorted for determinism.

        Returns ``[]`` when there are no cycles (the graph is a DAG).
        """
        raw: list[list[str]] = list(nx.simple_cycles(self._graph))
        normalised: list[list[str]] = []
        for cycle in raw:
            # Rotate cycle so it starts at the min element
            min_idx = cycle.index(min(cycle))
            normalised.append(cycle[min_idx:] + cycle[:min_idx])
        normalised.sort()
        return normalised

    def orphans(self) -> list[str]:
        """Nodes with no edges at all (degree 0).

        A document is an orphan when it neither depends on anything nor has
        anything depending on it.

        Returns a sorted list.
        """
        return sorted(n for n in self._graph.nodes if self._graph.degree(n) == 0)

    def stale(self, days: int, *, now: datetime | None = None) -> list[str]:
        """Document ids whose ``last_verified`` is older than *days* before *now*.

        Parameters
        ----------
        days:
            Age threshold in calendar days.
        now:
            Reference point (default: ``datetime.utcnow()``).  Provide a fixed
            value in tests for determinism.

        Accepted ``last_verified`` formats
        -----------------------------------
        * ``YYYY-MM-DD``  — full ISO date
        * ``YYYY-MM``     — treated as the 1st of that month

        Nodes with a missing or unparseable ``last_verified`` attribute are
        **not** considered stale (they are silently skipped).

        Returns a sorted list.
        """
        reference: datetime = now if now is not None else datetime.utcnow()
        cutoff = reference - timedelta(days=days)
        stale_ids: list[str] = []
        for node, attrs in self._graph.nodes(data=True):
            raw: str = attrs.get("last_verified", "")
            if not raw:
                continue
            verified: datetime | None = _parse_date(raw)
            if verified is None:
                continue
            # strict less-than: verified_date < cutoff means it IS stale
            if verified < cutoff:
                stale_ids.append(node)
        return sorted(stale_ids)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date(raw: str) -> datetime | None:
    """Parse a ``last_verified`` string into a :class:`datetime`.

    Accepts ``YYYY-MM-DD`` or ``YYYY-MM``.  Returns ``None`` on any parse
    failure (callers must treat None as "skip, not stale").
    """
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None
