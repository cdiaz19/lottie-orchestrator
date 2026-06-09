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
    metadata={"layer": "global", "doc_id": "kb/multiagent"},
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
    from lottie.knowledge.schema import DocStatus, Document
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
    """The grounded context from hits is included in the user message to the agent LLM."""
    retrieval, summarizer, agent_llm, _ = _make_seeded_deps()
    agent = ResearchAgent(agent_llm, retrieval=retrieval, summarizer=summarizer)

    agent.run(ResearchInput(query="multi-agent AI", k=3))

    # agent_llm.calls[0] is the messages list sent to self.complete
    assert len(agent_llm.calls) == 1
    user_message = next(m for m in agent_llm.calls[0] if m.role == "user")
    # The fixture chunk text must appear verbatim in the user message content
    assert _FIXTURE_CHUNK.text in user_message.content


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
