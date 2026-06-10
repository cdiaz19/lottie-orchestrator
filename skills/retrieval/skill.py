"""RetrievalSkill — embed-query → vector-store retrieval with optional graph expansion.

Wraps an injected ``EmbeddingProvider``, ``VectorStore``, and (optionally) a
``GraphStore`` behind the ``BaseSkill`` interface.  Agents must never import the
store directly; they reach it only through this skill (Golden Rule, CLAUDE.md
rule 12 / store base-class docstring).

No LLM is involved.  Given a fixed store and embedder the results are fully
deterministic, making this straightforwardly unit-testable (CLAUDE.md rule 5).

Graph expansion (v1 re-query approach)
---------------------------------------
When ``RetrievalQuery.expand_graph is True`` AND a ``GraphStore`` was injected,
after computing the normal vector hits, the skill also appends the best-matching
chunk from each document that the top hits *depend on* (their ``depends_on``
neighbours as returned by ``GraphStore.neighbors``), with a discounted score
(``score * GRAPH_EXPANSION_DISCOUNT``).

Key properties:

* **Additive**: expansion chunks are appended; the result may exceed ``q.k``.
  This is intentional — the caller decides how many results to consume.
* **Discounted**: ``GRAPH_EXPANSION_DISCOUNT = 0.5`` ensures base hits always
  rank above expansion hits with the same raw similarity, signalling their
  lower editorial confidence.
* **Non-duplicating**: a chunk id already present in base hits is never added
  again as an expansion hit.  One expansion chunk per neighbour document.
* **Deterministic**: neighbours are processed in sorted order; the final list is
  sorted by ``(score DESC, chunk.id ASC)``.
* **v1 re-query rationale**: to find the best chunk per neighbour doc without
  adding a ``get_by_doc_id`` API to ``VectorStore``, we issue a WIDE query
  (``max(q.k, store.count())``) against the same embedding and scan the result
  for chunks whose ``doc_id`` matches a neighbour id.  This avoids coupling the
  store abstraction to graph concepts and keeps both layers independently
  replaceable.  The trade-off — a second O(n) scan — is acceptable for the
  corpus sizes targeted by ``InMemoryVectorStore`` (< ~200 chunks).
* **Layer/tag filter independence**: the wide re-query used for expansion
  intentionally ignores the query's ``layers``/``tags`` filters, because a
  document's dependencies are relevant context wherever they live.  A dependency
  in a different layer is still a dependency.
"""

from __future__ import annotations

from pathlib import Path

from lottie.core import BaseSkill
from lottie.knowledge.embeddings import EmbeddingProvider
from lottie.knowledge.graph import GraphStore
from lottie.knowledge.schema import Embedding, RetrievalHit, RetrievalResult
from lottie.knowledge.store import VectorStore

from .schema import RetrievalSkillInput, RetrievalSkillOutput

#: Multiplicative discount applied to the vector score of graph-expanded hits.
#: A value of 0.5 ensures that a dependency chunk retrieved via graph expansion
#: always ranks below a base hit with the same raw similarity score.
GRAPH_EXPANSION_DISCOUNT: float = 0.5


class RetrievalSkill(BaseSkill[RetrievalSkillInput, RetrievalSkillOutput]):
    """Embed a query and retrieve top-k scored hits from an injected VectorStore.

    Optionally augments results with discounted chunks from ``depends_on``
    neighbours when ``RetrievalQuery.expand_graph is True`` and a ``GraphStore``
    was injected.  See module docstring for the full expansion contract.

    Parameters
    ----------
    embedder:
        Provider that converts query text into a dense ``Embedding`` vector.
    store:
        Vector store to query.  Populated by the ingestion pipeline; never
        modified by this skill.
    graph:
        Optional dependency graph.  When provided and the query sets
        ``expand_graph=True``, base-hit neighbours are fetched and appended
        with a discounted score.  ``None`` disables expansion silently.
    name:
        Optional display name forwarded to ``InstrumentedRunnable``.
    enable_benchmarks:
        If ``True`` / ``False``, overrides the ``LOTTIE_DISABLE_BENCHMARKS``
        env-var check.  ``None`` (default) defers to the env var.
    benchmarks_root:
        Directory under which benchmark JSONL files are appended.  Defaults
        to ``Path.cwd()``.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        graph: GraphStore | None = None,
        *,
        name: str | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._embedder = embedder
        self._store = store
        self._graph = graph

    def _execute(self, data: RetrievalSkillInput) -> RetrievalSkillOutput:
        """Embed the query text, retrieve ranked hits, and optionally expand via graph.

        When ``expand_graph`` is False (or no graph was injected) the behaviour
        is identical to the pre-expansion implementation: exactly the top-``k``
        base hits are returned, sorted by score DESC then chunk.id ASC.

        When ``expand_graph`` is True and a GraphStore is present:
        1. Compute base hits (top-k by vector similarity, respecting caller
           layer/tag filters).
        2. Delegate to ``_expand_graph_hits`` to collect discounted dependency
           chunks (layer/tag filters are intentionally not applied — see module
           docstring).
        3. Merge and sort final list by score DESC then chunk.id ASC.

        The result may exceed ``q.k`` when expansion is active — this is
        intentional (expansion is additive, not a replacement).
        """
        q = data.query
        embeddings = self._embedder.embed([q.text])
        if not embeddings:
            raise ValueError("embedding provider returned no embedding for the query text")
        embedding = embeddings[0]

        # Step 1 — base hits (caller filters respected here)
        base_hits = self._store.query(
            embedding,
            q.k,
            layers=q.layers or None,
            tags=q.tags or None,
        )

        # Step 2 — graph expansion (only when requested AND graph is available)
        if not q.expand_graph or self._graph is None:
            return RetrievalSkillOutput(result=RetrievalResult(hits=base_hits))

        expansion_hits = self._expand_graph_hits(base_hits, embedding)
        if not expansion_hits:
            return RetrievalSkillOutput(result=RetrievalResult(hits=base_hits))

        # Merge and sort: score DESC, chunk.id ASC for determinism on ties
        final_hits = base_hits + expansion_hits
        final_hits.sort(key=lambda h: (-h.score, h.chunk.id))

        return RetrievalSkillOutput(result=RetrievalResult(hits=final_hits))

    def _expand_graph_hits(
        self,
        base_hits: list[RetrievalHit],
        embedding: Embedding,
    ) -> list[RetrievalHit]:
        """Return discounted expansion hits for all unrepresented neighbour docs.

        Collects ``depends_on`` neighbour doc ids from every base-hit doc, then
        issues a WIDE query (``k = max(base k, store.count())``) WITHOUT the
        caller's layer/tag filters.  This is intentional: a document's
        dependencies are relevant context wherever they live, regardless of
        which layer or tags the caller originally filtered for.  See module
        docstring for the full rationale.

        For each remaining neighbour doc id (sorted for determinism) the first
        wide-hit chunk for that doc is selected, deduplicated against base-hit
        chunk ids, and appended with ``score * GRAPH_EXPANSION_DISCOUNT``.

        The wide hits are indexed into a dict (O(W)) so each neighbour lookup
        is O(1), giving O(W + N) overall rather than O(N × W).

        Parameters
        ----------
        base_hits:
            The top-k base results already computed by ``_execute``.
        embedding:
            Query embedding reused for the wide re-query.

        Returns
        -------
        list[RetrievalHit]
            Zero or more discounted expansion hits, in sorted order
            ``(-score, chunk.id)``.  Empty list when all neighbours are already
            covered by base hits.
        """
        assert self._graph is not None  # caller guarantees this

        # Collect doc ids already covered by base hits
        seen_doc_ids: set[str] = {hit.chunk.doc_id for hit in base_hits}

        # Gather all depends_on neighbour doc ids across base-hit docs
        neighbor_ids: set[str] = set()
        for hit in base_hits:
            neighbor_ids |= set(self._graph.neighbors(hit.chunk.doc_id))
        # Remove any neighbour already represented in the base hits
        neighbor_ids -= seen_doc_ids

        if not neighbor_ids:
            return []

        # WIDE query: intentionally NO layer/tag filters so cross-layer
        # dependencies are reachable.  See module docstring.
        wide_k = max(len(base_hits), self._store.count())
        wide_hits = self._store.query(embedding, wide_k, layers=None, tags=None)

        # Build O(1)-lookup dict: first wide hit per doc_id (preserves wide order
        # so the best-scored chunk per doc is selected).
        first_hit_by_doc: dict[str, RetrievalHit] = {}
        for wide_hit in wide_hits:
            if wide_hit.chunk.doc_id not in first_hit_by_doc:
                first_hit_by_doc[wide_hit.chunk.doc_id] = wide_hit

        # Build the set of chunk ids already in base results (for dedup)
        seen_chunk_ids: set[str] = {hit.chunk.id for hit in base_hits}

        # Collect one expansion hit per neighbour doc, processed in sorted
        # order for determinism
        expansion_hits: list[RetrievalHit] = []
        for neighbor_doc_id in sorted(neighbor_ids):
            candidate: RetrievalHit | None = first_hit_by_doc.get(neighbor_doc_id)
            if candidate is None or candidate.chunk.id in seen_chunk_ids:
                continue
            expansion_hits.append(
                RetrievalHit(
                    chunk=candidate.chunk,
                    score=candidate.score * GRAPH_EXPANSION_DISCOUNT,
                )
            )
            seen_chunk_ids.add(candidate.chunk.id)

        return expansion_hits
