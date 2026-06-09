"""Benchmark runner test for ResearchAgent using mocks only (no network, no real LLM).

TDD: written before the runner seam-wiring fix.

Coverage
--------
- The benchmark runner's `benchmark()` function constructs ResearchAgent via the
  `instantiate_agent` DI seam (not the bare ``agent_cls(llm=...)`` call), so
  knowledge-backed agents that require extra constructor args are supported.
- The resulting BenchmarkReport has the expected number of CaseResult entries and
  populated metrics (latency, etc.) for each case.

Response budget per case
------------------------
Each eval case triggers TWO LLM calls through the ResearchAgent:
  1. agent.complete()  — the reasoning call inside ResearchAgent._execute
  2. SummarizerSkill._llm.complete() — the summarisation call
For N cases that is 2 * N responses.  With 4 cases: 8 responses.

MockLLMProvider is seeded with exactly 8 responses (4 agent + 4 summariser,
interleaved by call order).  The provider draws from the list in FIFO order so
the pairing is: case1 → resp0 (agent) + resp1 (summariser),
               case2 → resp2 + resp3, etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.benchmark.runner import benchmark
from lottie.benchmark.schema import BenchmarkReport
from lottie.core import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.project.config import AgentConfig
from lottie.project.discovery import instantiate_agent

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_KNOWLEDGE_DOC = """\
---
id: kb/multiagent
layer: global
scope: project
tags: [ai, agents]
status: curated
last_verified: "2025-01-01"
depends_on: []
---

Multi-agent AI systems coordinate specialised agents via typed messages.
Orchestration frameworks like Lottie route tasks to the best-suited agent.
Each agent has a defined schema, skills, and a scoped knowledge layer.
"""

_AGENT_RESPONSE = (
    "Multi-agent AI systems use typed messages and a shared knowledge layer."
)
_SUMMARIZER_RESPONSE = (
    "Systems coordinate agents.\n- typed messages\n- specialised roles"
)

# 4 cases × 2 LLM calls each = 8 responses total
_MOCK_RESPONSES = [
    _AGENT_RESPONSE,
    _SUMMARIZER_RESPONSE,
    _AGENT_RESPONSE,
    _SUMMARIZER_RESPONSE,
    _AGENT_RESPONSE,
    _SUMMARIZER_RESPONSE,
    _AGENT_RESPONSE,
    _SUMMARIZER_RESPONSE,
]


def _make_fixture_root(tmp_path: Path) -> Path:
    """Minimal hermetic project: knowledge/global + agents/research stubs."""
    # Knowledge document
    knowledge_global = tmp_path / "knowledge" / "global"
    knowledge_global.mkdir(parents=True)
    (knowledge_global / "multiagent.md").write_text(_KNOWLEDGE_DOC, encoding="utf-8")

    # agents/research must be importable from tmp_path; symlink is fragile in CI
    # so we copy the real research agent files into tmp_path/agents/research.
    import shutil

    real_agents = Path(__file__).parent.parent.parent.parent.parent / "agents"
    shutil.copytree(str(real_agents / "research"), str(tmp_path / "agents" / "research"))
    # Make agents a package
    (tmp_path / "agents" / "__init__.py").touch()

    # skills must also be importable — symlink to actual skills/
    real_skills = real_agents.parent / "skills"
    shutil.copytree(str(real_skills), str(tmp_path / "skills"))

    return tmp_path


# ---------------------------------------------------------------------------
# TDD: failing test (before seam wiring) → should pass after wiring
# ---------------------------------------------------------------------------


def test_benchmark_research_with_mock_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """benchmark() runs research evals via the DI seam → BenchmarkReport with 4 CaseResults.

    No network: mock/embed embedder + InMemoryVectorStore + MockLLMProvider.
    The runner must call instantiate_agent (seam) not agent_cls(llm=...) directly,
    otherwise ResearchAgent raises TypeError (missing 'retrieval' kwarg).
    """
    # Force mock embedding + in-memory store for the from_project seam
    monkeypatch.setenv("LOTTIE_EMBEDDING_MODEL", "mock/embed")
    monkeypatch.setenv("LOTTIE_VECTOR_STORE", "memory")

    root = _make_fixture_root(tmp_path)

    # Patch build_provider in the runner to return our MockLLMProvider.
    # The runner calls build_provider(p) for each provider string; we intercept it.
    monkeypatch.setattr(
        "lottie.benchmark.runner.build_provider",
        lambda _p: MockLLMProvider(_MOCK_RESPONSES),
    )

    # Also patch load_agent_config so runner finds a valid config without needing
    # a real lottie.yaml (the runner loads config via load_agent_config).
    from lottie.project.config import AgentConfig

    monkeypatch.setattr(
        "lottie.benchmark.runner.load_agent_config",
        lambda _path: AgentConfig(provider="mock/mock-model"),
    )

    report: BenchmarkReport = benchmark(root, "research", ["mock/mock-model"])

    # Structural assertions — valid BenchmarkReport
    assert isinstance(report, BenchmarkReport)
    assert report.agent == "research"
    assert len(report.providers) == 1

    provider_report = report.providers[0]
    # evals.yaml has 4 cases (after our expansion in this task)
    assert provider_report.case_count == 4
    assert len(provider_report.cases) == 4

    # All cases must have succeeded (no TypeError from missing retrieval arg)
    for case_result in provider_report.cases:
        assert case_result.success, (
            f"Case '{case_result.name}' failed: {case_result.error}"
        )
        assert case_result.passed, (
            f"Case '{case_result.name}' eval expectations not met"
        )

    # Aggregate metrics are populated
    assert provider_report.success_rate == pytest.approx(1.0)
    # latency_p50 >= 0 (mock timings are 0 or near-0 but field is set)
    assert provider_report.latency_p50_ms >= 0.0
    assert provider_report.latency_p95_ms >= 0.0


def test_benchmark_research_report_has_named_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CaseResult names match the names declared in evals.yaml."""
    monkeypatch.setenv("LOTTIE_EMBEDDING_MODEL", "mock/embed")
    monkeypatch.setenv("LOTTIE_VECTOR_STORE", "memory")

    root = _make_fixture_root(tmp_path)

    monkeypatch.setattr(
        "lottie.benchmark.runner.build_provider",
        lambda _p: MockLLMProvider(_MOCK_RESPONSES),
    )

    from lottie.project.config import AgentConfig

    monkeypatch.setattr(
        "lottie.benchmark.runner.load_agent_config",
        lambda _path: AgentConfig(provider="mock/mock-model"),
    )

    report = benchmark(root, "research", ["mock/mock-model"])
    names = [c.name for c in report.providers[0].cases]

    # The four cases we define in evals.yaml — assert each is present
    expected_names = {
        "multi-agent query returns non-empty digest",
        "coordination query returns digest",
        "minimal query no filters",
        "k=1 returns single hit digest",
    }
    assert set(names) == expected_names


# ---------------------------------------------------------------------------
# Regression guard: enable_benchmarks=False must propagate through the seam
# ---------------------------------------------------------------------------


class _DummyIn(BaseModel):
    query: str


class _DummyOut(BaseModel):
    result: str


class _DummyAgent(BaseAgent[_DummyIn, _DummyOut]):
    """Minimal plain agent (no from_project) used to test the plain-cls branch."""

    def _execute(self, data: _DummyIn) -> _DummyOut:
        return _DummyOut(result=data.query)


def test_research_agent_fallback_summarizer_forwards_enable_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ResearchAgent with no explicit summarizer arg uses a fallback SummarizerSkill
    that inherits enable_benchmarks=False — regression guard for the bug where the
    fallback was constructed without forwarding the flag.
    """
    monkeypatch.setenv("LOTTIE_EMBEDDING_MODEL", "mock/embed")
    monkeypatch.setenv("LOTTIE_VECTOR_STORE", "memory")

    from lottie.knowledge.embeddings import build_embedding_provider
    from lottie.knowledge.store import build_vector_store

    embedder = build_embedding_provider("mock/embed")
    store = build_vector_store("memory", tmp_path)

    from skills.retrieval.skill import RetrievalSkill

    retrieval = RetrievalSkill(embedder, store)

    from agents.research.agent import ResearchAgent

    llm = MockLLMProvider(["unused"])
    # No summarizer passed → fallback SummarizerSkill must pick up enable_benchmarks=False
    agent = ResearchAgent(llm, retrieval=retrieval, enable_benchmarks=False)

    assert agent._summarizer._enable_benchmarks is False


def test_instantiate_agent_plain_enable_benchmarks_false(tmp_path: Path) -> None:
    """Plain agent built via instantiate_agent(..., enable_benchmarks=False) has flag False.

    Locks the regression: the benchmark runner must suppress nested writes for
    simple agents that have no from_project classmethod.
    """
    llm = MockLLMProvider(["unused"])
    config = AgentConfig(provider="mock/mock-model")
    agent = instantiate_agent(
        _DummyAgent,  # type: ignore[arg-type]
        llm=llm,
        root=tmp_path,
        config=config,
        enable_benchmarks=False,
    )
    assert agent._enable_benchmarks is False


def test_instantiate_agent_from_project_enable_benchmarks_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ResearchAgent built via instantiate_agent(..., enable_benchmarks=False) has flag False.

    Locks the regression for from_project agents: the flag must propagate through
    the seam into the agent AND its nested skills (retrieval, summarizer).
    """
    monkeypatch.setenv("LOTTIE_EMBEDDING_MODEL", "mock/embed")
    monkeypatch.setenv("LOTTIE_VECTOR_STORE", "memory")

    root = _make_fixture_root(tmp_path)

    # Load the ResearchAgent class from the tmp_path fixture root
    from lottie.project.discovery import load_agent_class

    agent_cls = load_agent_class(root, "research")
    llm = MockLLMProvider(["unused"])
    config = AgentConfig(provider="mock/mock-model")

    agent = instantiate_agent(
        agent_cls,
        llm=llm,
        root=root,
        config=config,
        enable_benchmarks=False,
    )

    # The agent itself must have the flag suppressed
    assert agent._enable_benchmarks is False

    # The nested skills must also have the flag suppressed so no spurious
    # benchmark writes occur when the runner calls agent.run() per eval case.
    from agents.research.agent import ResearchAgent

    assert isinstance(agent, ResearchAgent)
    assert agent._retrieval._enable_benchmarks is False
    assert agent._summarizer._enable_benchmarks is False
