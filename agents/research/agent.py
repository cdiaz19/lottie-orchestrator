"""ResearchAgent — reads the knowledge layer via RetrievalSkill + SummarizerSkill.

Architecture
------------
The agent never touches a vector store or graph directly.  All retrieval goes
through ``RetrievalSkill`` (Golden Rule, CLAUDE.md rule 12).  All LLM calls go
through ``self.complete`` (CLAUDE.md rule 1), which auto-accumulates token and
cost usage into the active run context.

Flow
----
1. Build a ``RetrievalQuery`` from the caller's ``ResearchInput``.
2. Delegate to ``RetrievalSkill`` — receives ``RetrievalHit`` objects.
3. Build a grounded numbered context string from the hits.
4. Call ``self.complete`` with the system prompt + (query + context) user message.
5. Pass the LLM response to ``SummarizerSkill`` to extract digest + bullets.
6. Return ``ResearchOutput(digest, points, citations)``.
"""

from __future__ import annotations

from pathlib import Path

from skills.retrieval.schema import RetrievalSkillInput
from skills.retrieval.skill import RetrievalSkill
from skills.summarizer.schema import SummarizerInput
from skills.summarizer.skill import SummarizerSkill

from lottie.core import BaseAgent
from lottie.knowledge import (
    GraphStore,
    KnowledgeManifest,
    index_manifest,
    resolve_embedding_settings,
)
from lottie.knowledge.embeddings import build_embedding_provider
from lottie.knowledge.schema import RetrievalQuery
from lottie.knowledge.store import build_vector_store
from lottie.llm import LLMProvider, Message
from lottie.memory.base import MemoryClient
from lottie.project.config import AgentConfig

from .prompts import SYSTEM_PROMPT
from .schema import Citation, ResearchInput, ResearchOutput


class ResearchAgent(BaseAgent[ResearchInput, ResearchOutput]):
    """Knowledge-grounded research agent.

    Retrieves relevant chunks via ``RetrievalSkill``, synthesises a grounded
    response with the injected LLM, then summarises into digest + bullet points
    via ``SummarizerSkill``.

    Parameters
    ----------
    llm:
        LLM provider for reasoning.  Also used by the default summarizer unless
        a dedicated ``summarizer`` is injected.
    retrieval:
        Required injected retrieval skill.  The agent never touches the store
        directly — all reads go through this skill (Golden Rule).
    summarizer:
        Optional summariser skill.  When ``None`` a ``SummarizerSkill`` is
        constructed from the agent's own ``llm``.
    name, memory, enable_benchmarks, benchmarks_root:
        Forwarded to :class:`~lottie.core.BaseAgent`.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        retrieval: RetrievalSkill,
        summarizer: SummarizerSkill | None = None,
        name: str | None = None,
        memory: MemoryClient | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            llm,
            name=name,
            memory=memory,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._retrieval = retrieval
        self._summarizer = summarizer or SummarizerSkill(llm, enable_benchmarks=enable_benchmarks)

    # ------------------------------------------------------------------
    # DI factory — used by AgentService / CLI when knowledge layer present
    # ------------------------------------------------------------------

    @classmethod
    def from_project(
        cls,
        *,
        llm: LLMProvider,
        root: Path,
        config: AgentConfig,
        enable_benchmarks: bool | None = None,
    ) -> ResearchAgent:
        """Construct a ResearchAgent wired to the project's knowledge layer.

        Reads embedding settings from the environment via
        :func:`~lottie.knowledge.resolve_embedding_settings` so that
        ``lottie run research`` works with NO API key in Phase 1; production
        deployments override ``LOTTIE_EMBEDDING_MODEL`` and
        ``LOTTIE_VECTOR_STORE``.

        Steps
        -----
        1. Build embedding provider + vector store from env vars.
        2. Load the project's ``KnowledgeManifest`` from *root*.
        3. Warm the store via :func:`~lottie.knowledge.index.index_manifest`
           (existing, already-vetted knowledge docs — security gate not re-run).
        4. Build a ``GraphStore`` over the manifest for graph expansion.
        5. Construct and return a ``ResearchAgent`` with injected skills.

        Parameters
        ----------
        enable_benchmarks:
            Forwarded to every ``InstrumentedRunnable`` constructed here
            (``RetrievalSkill``, ``SummarizerSkill``, and the agent itself) so
            that nested benchmark writes are uniformly suppressed when the
            benchmark runner passes ``False``.
        """
        # config is part of the from_project seam contract; reserved for
        # per-agent embedder/store overrides (Phase 2).  Unused today.
        _ = config

        # TODO: index_manifest re-chunks+re-embeds the whole corpus on EVERY
        # construction (i.e. every run_agent call).  Fine for Phase 1 (in-memory
        # + mock embedder) but production must cache the warmed store or use a
        # persistent Chroma backend to avoid O(corpus) embedding cost per request.
        embedding_model, vector_store_kind = resolve_embedding_settings()

        embedder = build_embedding_provider(embedding_model)
        store = build_vector_store(vector_store_kind, root)
        manifest = KnowledgeManifest.from_root(root)
        index_manifest(manifest, embedder, store)
        graph = GraphStore(manifest)
        retrieval = RetrievalSkill(embedder, store, graph, enable_benchmarks=enable_benchmarks)
        summarizer = SummarizerSkill(llm, enable_benchmarks=enable_benchmarks)
        return cls(
            llm,
            retrieval=retrieval,
            summarizer=summarizer,
            enable_benchmarks=enable_benchmarks,
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _execute(self, data: ResearchInput) -> ResearchOutput:
        """Retrieve → ground → complete → summarise → return."""
        # 1. Build retrieval query
        rq = RetrievalQuery(
            text=data.query,
            k=data.k,
            layers=data.layers,
            expand_graph=data.expand_graph,
        )

        # 2. Retrieve hits — only through the skill, never the store directly
        hits = self._retrieval.run(RetrievalSkillInput(query=rq)).result.hits

        # 3. Build grounded numbered context.
        # Retrieved chunk text was already scanned by PromptInjectionScanSkill
        # at ingest time (CLAUDE.md rule 10) — no re-scan needed here.
        if hits:
            context_lines: list[str] = []
            for i, hit in enumerate(hits, start=1):
                context_lines.append(
                    f"[{i}] (doc_id={hit.chunk.doc_id})\n{hit.chunk.text}"
                )
            context = "\n\n".join(context_lines)
        else:
            context = "No relevant knowledge found."

        user_content = (
            f"Query: {data.query}\n\nContext:\n{context}"
        )

        # 4. LLM reasoning — tokens auto-accumulated via self.complete
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ]
        )

        # 5. Summarise LLM response into digest + bullets
        summ = self._summarizer.run(
            SummarizerInput(text=response.content, max_points=data.max_points)
        )

        # 6. Build citations
        citations = [
            Citation(
                doc_id=h.chunk.doc_id,
                chunk_id=h.chunk.id,
                score=h.score,
                source=h.chunk.metadata.get("source", h.chunk.doc_id),
            )
            for h in hits
        ]

        return ResearchOutput(
            digest=summ.summary,
            points=summ.points,
            citations=citations,
        )
