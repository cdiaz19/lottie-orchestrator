"""RetrievalSkill — embed-query → vector-store retrieval.

Wraps an injected ``EmbeddingProvider`` and ``VectorStore`` behind the
``BaseSkill`` interface.  Agents must never import the store directly; they
reach it only through this skill (Golden Rule, CLAUDE.md rule 12 / store
base-class docstring).

No LLM is involved.  Given a fixed store and embedder the results are fully
deterministic, making this straightforwardly unit-testable (CLAUDE.md rule 5).
"""

from __future__ import annotations

from pathlib import Path

from lottie.core import BaseSkill
from lottie.knowledge.embeddings import EmbeddingProvider
from lottie.knowledge.schema import RetrievalResult
from lottie.knowledge.store import VectorStore

from .schema import RetrievalSkillInput, RetrievalSkillOutput


class RetrievalSkill(BaseSkill[RetrievalSkillInput, RetrievalSkillOutput]):
    """Embed a query and retrieve top-k scored hits from an injected VectorStore.

    Parameters
    ----------
    embedder:
        Provider that converts query text into a dense ``Embedding`` vector.
    store:
        Vector store to query.  Populated by the ingestion pipeline; never
        modified by this skill.
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

    def _execute(self, data: RetrievalSkillInput) -> RetrievalSkillOutput:
        """Embed the query text and retrieve ranked hits from the store."""
        q = data.query
        embedding = self._embedder.embed([q.text])[0]
        hits = self._store.query(
            embedding,
            q.k,
            layers=q.layers or None,
            tags=q.tags or None,
        )
        return RetrievalSkillOutput(result=RetrievalResult(hits=hits))
