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
from lottie.llm import LLMProvider, Message
from lottie.memory.base import MemoryClient

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
        self._summarizer = summarizer or SummarizerSkill(llm)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _execute(self, data: ResearchInput) -> ResearchOutput:
        """Retrieve → ground → complete → summarise → return."""
        from lottie.knowledge.schema import RetrievalQuery

        # 1. Build retrieval query
        rq = RetrievalQuery(
            text=data.query,
            k=data.k,
            layers=data.layers,
            expand_graph=data.expand_graph,
        )

        # 2. Retrieve hits — only through the skill, never the store directly
        hits = self._retrieval.run(RetrievalSkillInput(query=rq)).result.hits

        # 3. Build grounded numbered context
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
            SummarizerInput(text=response.content, max_points=data.k)
        )

        # 6. Build citations
        citations = [
            Citation(
                doc_id=h.chunk.doc_id,
                chunk_id=h.chunk.id,
                score=h.score,
                source=h.chunk.metadata.get("doc_id", h.chunk.doc_id),
            )
            for h in hits
        ]

        return ResearchOutput(
            digest=summ.summary,
            points=summ.points,
            citations=citations,
        )
