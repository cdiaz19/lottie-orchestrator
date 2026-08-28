"""Integration tests for ResearchAgent (MockLLMProvider — no real LLM).

All tests follow TDD Red-Green-Refactor.  No real LLM or network (CLAUDE.md rule 5).
"""

from __future__ import annotations

from skills.retrieval.skill import RetrievalSkill
from skills.summarizer.skill import SummarizerSkill

from agents.research.agent import ResearchAgent
from agents.research.schema import ResearchInput, ResearchOutput
from lottie.knowledge.embeddings import MockEmbeddingProvider
from lottie.knowledge.graph import GraphStore
from lottie.knowledge.manifest import KnowledgeManifest
from lottie.knowledge.schema import (
    Chunk,
    DocStatus,
    Document,
    EmbeddedChunk,
    KnowledgeLayer,
)
from lottie.knowledge.store import InMemoryVectorStore
from lottie.llm import MockLLMProvider

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_FIXTURE_CHUNK = Chunk(
    id="kb/multiagent#0",
    doc_id="kb/multiagent",
    index=0,
    text="Multi-agent AI systems coordinate specialized agents via typed messages.",
    start=0,
    end=60,
    metadata={
        "layer": "global",
        "doc_id": "kb/multiagent",
        "source": "knowledge/global/multiagent.md",
    },
)

_SUMMARIZER_RESPONSE = (
    "Multi-agent systems coordinate agents.\n"
    "- typed messages\n"
    "- specialized roles"
)

_AGENT_LLM_RESPONSE = (
    "Multi-agent AI systems use typed messages to coordinate specialized agents."
)


def _make_seeded_deps() -> (
    tuple[RetrievalSkill, SummarizerSkill, MockLLMProvider, MockLLMProvider]
):
    """Build a seeded RetrievalSkill + SummarizerSkill + two mock LLMs.

    Returns (retrieval, summarizer, agent_llm_mock, summarizer_llm_mock).
    """
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()

    emb = embedder.embed([_FIXTURE_CHUNK.text])[0]
    store.add([EmbeddedChunk(chunk=_FIXTURE_CHUNK, embedding=emb)])

    # Minimal manifest with the fixture doc
    manifest = KnowledgeManifest(
        documents=[
            Document(
                id="kb/multiagent",
                source="knowledge/global/multiagent.md",
                layer=KnowledgeLayer.GLOBAL,
                content=_FIXTURE_CHUNK.text,
                status=DocStatus.CURATED,
            )
        ]
    )
    graph = GraphStore(manifest)
    retrieval = RetrievalSkill(embedder, store, graph)

    summarizer_llm = MockLLMProvider([_SUMMARIZER_RESPONSE])
    summarizer = SummarizerSkill(summarizer_llm)

    agent_llm = MockLLMProvider([_AGENT_LLM_RESPONSE])

    return retrieval, summarizer, agent_llm, summarizer_llm


# ---------------------------------------------------------------------------
# Test 1 — basic output shape and citations
# ---------------------------------------------------------------------------


def test_research_output_has_digest_and_citations() -> None:
    """Agent returns non-empty digest, at least one citation from fixture chunk."""
    retrieval, summarizer, agent_llm, _ = _make_seeded_deps()
    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)

    out = agent.run(ResearchInput(query="multi-agent AI", k=3))

    assert isinstance(out, ResearchOutput)
    assert out.digest  # non-empty
    assert len(out.citations) >= 1
    # The citation references the fixture chunk
    doc_ids = [c.doc_id for c in out.citations]
    chunk_ids = [c.chunk_id for c in out.citations]
    assert "kb/multiagent" in doc_ids
    assert "kb/multiagent#0" in chunk_ids
    # Citation.source must be the document file path, not doc_id
    sources = [c.source for c in out.citations]
    assert "knowledge/global/multiagent.md" in sources


# ---------------------------------------------------------------------------
# Test 2 — summarizer bullets appear in points
# ---------------------------------------------------------------------------


def test_research_points_reflect_summarizer_bullets() -> None:
    """Agent.points mirrors the summarizer bullet output."""
    retrieval, summarizer, agent_llm, _ = _make_seeded_deps()
    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)

    out = agent.run(ResearchInput(query="multi-agent AI", k=3))

    assert "typed messages" in out.points
    assert "specialized roles" in out.points


# ---------------------------------------------------------------------------
# Test 3 — retrieved chunk text reaches the LLM prompt
# ---------------------------------------------------------------------------


def test_research_prompt_contains_retrieved_chunk_text() -> None:
    """The grounded context from hits reaches the agent LLM.

    E4 S4 moved knowledge out of the user message and into a separate DROPPABLE
    context source, so it now arrives as its own message rather than concatenated
    into the query. The guarantee that matters — the retrieved text reaches the
    prompt — is unchanged, so the assertion is over the whole prompt.
    """
    retrieval, summarizer, agent_llm, _ = _make_seeded_deps()
    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)

    agent.run(ResearchInput(query="multi-agent AI", k=3))

    assert len(agent_llm.calls) == 1
    prompt = "\n".join(m.content for m in agent_llm.calls[0])
    assert _FIXTURE_CHUNK.text in prompt


def test_research_query_survives_when_knowledge_is_dropped() -> None:
    """Knowledge is droppable; the query is not.

    The point of E4 S4: over budget, the compiler gives up retrieved knowledge before
    it touches the task. Concatenated context could only ever be compacted by position,
    which is how a query used to get summarised away along with its own grounding.
    """
    retrieval, summarizer, _, _ = _make_seeded_deps()
    # Two responses: summarising the droppable knowledge consumes one, the real
    # completion the other.
    agent_llm = MockLLMProvider(["a summary of the context", _AGENT_LLM_RESPONSE])
    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)
    # A ceiling far below the retrieved context forces the drop policy to act.
    agent.set_compaction(enabled=True, max_context_tokens=1, keep_recent=6)

    agent.run(ResearchInput(query="multi-agent AI", k=3))

    final_prompt = "\n".join(m.content for m in agent_llm.calls[-1])
    assert "multi-agent AI" in final_prompt
    # …and the verbatim knowledge is gone, replaced by its summary.
    assert _FIXTURE_CHUNK.text not in final_prompt


# ---------------------------------------------------------------------------
# Test 4 — metrics are set after run
# ---------------------------------------------------------------------------


def test_research_last_metrics_set_after_run() -> None:
    """last_metrics is not None after a successful run."""
    retrieval, summarizer, agent_llm, _ = _make_seeded_deps()
    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)

    agent.run(ResearchInput(query="multi-agent AI", k=3))

    assert agent.last_metrics is not None


# ---------------------------------------------------------------------------
# Test 5 — empty store: no citations, digest still present, no crash
# ---------------------------------------------------------------------------


def test_research_empty_store_no_citations() -> None:
    """With an empty store, citations==[], digest is non-empty, no exception."""
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()  # empty — nothing seeded
    manifest = KnowledgeManifest(documents=[])
    graph = GraphStore(manifest)
    retrieval = RetrievalSkill(embedder, store, graph)

    # The agent's LLM and summarizer each need exactly one response
    agent_llm = MockLLMProvider(["No relevant knowledge found in the knowledge base."])
    summarizer_llm = MockLLMProvider(
        ["No knowledge found.\n- check knowledge layer\n- add relevant docs"]
    )
    summarizer = SummarizerSkill(summarizer_llm)

    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)
    out = agent.run(ResearchInput(query="unknown topic", k=3))

    assert out.citations == []
    assert out.digest  # non-empty


# ---------------------------------------------------------------------------
# Test 6 — summarizer defaults to agent's own LLM when not injected
# ---------------------------------------------------------------------------


def test_research_default_summarizer_uses_agent_llm() -> None:
    """When summarizer=None the agent builds one from its own LLM (two calls total)."""
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()
    emb = embedder.embed([_FIXTURE_CHUNK.text])[0]
    store.add([EmbeddedChunk(chunk=_FIXTURE_CHUNK, embedding=emb)])
    manifest = KnowledgeManifest(documents=[])
    graph = GraphStore(manifest)
    retrieval = RetrievalSkill(embedder, store, graph)

    # Two responses: first for agent.complete, second for the internal summarizer
    agent_llm = MockLLMProvider(
        [
            _AGENT_LLM_RESPONSE,
            "Fallback summary.\n- point one",
        ]
    )
    agent = ResearchAgent(agent_llm, retrieval=retrieval)  # no summarizer injected

    out = agent.run(ResearchInput(query="multi-agent AI", k=3))

    # The agent LLM is called by both self.complete AND the internal summarizer
    assert len(agent_llm.calls) == 2
    assert out.digest  # non-empty


# ---------------------------------------------------------------------------
# Test 7 — max_points controls bullet cap independently of k
# ---------------------------------------------------------------------------


def test_max_points_controls_bullet_cap_independently_of_k() -> None:
    """max_points=1 limits bullets to 1 even when k=5 would previously cap at 5."""
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()
    emb = embedder.embed([_FIXTURE_CHUNK.text])[0]
    store.add([EmbeddedChunk(chunk=_FIXTURE_CHUNK, embedding=emb)])
    manifest = KnowledgeManifest(documents=[])
    graph = GraphStore(manifest)
    retrieval = RetrievalSkill(embedder, store, graph)

    # Summarizer returns one bullet point
    summarizer_llm = MockLLMProvider(
        ["Summary with limited bullets.\n- only bullet"]
    )
    summarizer = SummarizerSkill(summarizer_llm)
    agent_llm = MockLLMProvider([_AGENT_LLM_RESPONSE])
    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)

    # k=5 but max_points=1 — the SummarizerInput should receive max_points=1
    out = agent.run(ResearchInput(query="multi-agent AI", k=5, max_points=1))

    assert isinstance(out, ResearchOutput)
    # Verify max_points=1 was passed (summarizer received 1-point instruction)
    # The mock summarizer returned one bullet; it should appear in points
    assert "only bullet" in out.points
