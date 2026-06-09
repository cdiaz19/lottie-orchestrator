"""Tests for ResearchAgent.from_project DI seam (Task 18 — Phase 1).

TDD Red-Green: tests written before implementation.
No real LLM, no real embedder, no network (CLAUDE.md rule 5).

Three coverage areas:
A. index.py helper wired through from_project → knowledge doc indexed, hit returned
B. from_project builds a working ResearchAgent (digest + citations)
C. DI-seam regression: plain agents without from_project still use (llm=llm)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lottie.llm import MockLLMProvider
from lottie.project.config import AgentConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AGENT_LLM_RESPONSE = (
    "Multi-agent AI systems use typed messages to coordinate specialized agents."
)
_SUMMARIZER_RESPONSE = (
    "Multi-agent systems coordinate agents.\n"
    "- typed messages\n"
    "- specialized roles"
)

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

Multi-agent AI systems coordinate specialized agents via typed messages.
Orchestration frameworks like Lottie route tasks to the best-suited agent.
"""


def _make_fixture_root(tmp_path: Path) -> Path:
    """Create a minimal fixture project with one knowledge doc under global/."""
    knowledge_global = tmp_path / "knowledge" / "global"
    knowledge_global.mkdir(parents=True)
    (knowledge_global / "multiagent.md").write_text(_KNOWLEDGE_DOC, encoding="utf-8")
    return tmp_path


def _default_agent_config() -> AgentConfig:
    return AgentConfig(provider="anthropic/claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# A. from_project builds a working ResearchAgent
# ---------------------------------------------------------------------------


def test_from_project_returns_research_agent_instance(tmp_path: Path) -> None:
    """from_project returns a ResearchAgent with the injected LLM."""
    import os

    os.environ["LOTTIE_EMBEDDING_MODEL"] = "mock/embed"
    os.environ["LOTTIE_VECTOR_STORE"] = "memory"

    root = _make_fixture_root(tmp_path)
    llm = MockLLMProvider([_AGENT_LLM_RESPONSE, _SUMMARIZER_RESPONSE])
    cfg = _default_agent_config()

    # Import after setting env vars to avoid stale module state
    from agents.research.agent import ResearchAgent

    agent = ResearchAgent.from_project(llm=llm, root=root, config=cfg)

    assert isinstance(agent, ResearchAgent)


def test_from_project_run_produces_digest(tmp_path: Path) -> None:
    """from_project → .run(query) returns a non-empty digest."""
    import os

    os.environ["LOTTIE_EMBEDDING_MODEL"] = "mock/embed"
    os.environ["LOTTIE_VECTOR_STORE"] = "memory"

    root = _make_fixture_root(tmp_path)
    llm = MockLLMProvider([_AGENT_LLM_RESPONSE, _SUMMARIZER_RESPONSE])
    cfg = _default_agent_config()

    from agents.research.agent import ResearchAgent
    from agents.research.schema import ResearchInput

    agent = ResearchAgent.from_project(llm=llm, root=root, config=cfg)
    out = agent.run(ResearchInput(query="multi-agent AI"))

    assert out.digest


def test_from_project_run_produces_citations_from_indexed_knowledge(
    tmp_path: Path,
) -> None:
    """from_project indexes knowledge docs; retrieval returns a citation."""
    import os

    os.environ["LOTTIE_EMBEDDING_MODEL"] = "mock/embed"
    os.environ["LOTTIE_VECTOR_STORE"] = "memory"

    root = _make_fixture_root(tmp_path)
    llm = MockLLMProvider([_AGENT_LLM_RESPONSE, _SUMMARIZER_RESPONSE])
    cfg = _default_agent_config()

    from agents.research.agent import ResearchAgent
    from agents.research.schema import ResearchInput

    agent = ResearchAgent.from_project(llm=llm, root=root, config=cfg)
    out = agent.run(ResearchInput(query="multi-agent AI", k=3))

    # The knowledge doc was indexed, so at least one citation should appear
    assert len(out.citations) >= 1
    doc_ids = [c.doc_id for c in out.citations]
    assert "kb/multiagent" in doc_ids


# ---------------------------------------------------------------------------
# B. DI-seam regression: plain agents (no from_project) still use (llm=llm)
# ---------------------------------------------------------------------------


def test_service_constructs_plain_agent_without_from_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AgentService falls back to (llm=llm) when from_project is absent (plain agents)."""
    from typer.testing import CliRunner

    from lottie.cli import app
    from lottie.llm import MockLLMProvider
    from lottie.serve.service import AgentService

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0

    # Patch build_provider to return a MockLLMProvider
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider(["hello world"]),
    )

    svc = AgentService(demo)
    result = svc.run_agent("echo", {"query": "hi"})
    assert result.output == {"result": "hello world"}


def test_service_uses_from_project_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AgentService calls from_project when the class defines it.

    We inject a spy via monkeypatching load_agent_class to return a class
    that has from_project — verifying the conditional branch is taken.
    """
    import os

    os.environ["LOTTIE_EMBEDDING_MODEL"] = "mock/embed"
    os.environ["LOTTIE_VECTOR_STORE"] = "memory"

    from typer.testing import CliRunner

    from lottie.cli import app
    from lottie.llm import LLMProvider, MockLLMProvider
    from lottie.serve.service import AgentService

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0

    from_project_called: list[bool] = []

    # We need a proper BaseAgent subclass for the service to instantiate
    from pydantic import BaseModel

    from lottie.core import BaseAgent

    class EchoInput(BaseModel):
        query: str

    class EchoOutput(BaseModel):
        result: str

    class SpyAgent(BaseAgent[EchoInput, EchoOutput]):
        def _execute(self, data: EchoInput) -> EchoOutput:
            return EchoOutput(result="from_project_result")

        @classmethod
        def from_project(
            cls, *, llm: LLMProvider, root: Path, config: AgentConfig
        ) -> SpyAgent:
            from_project_called.append(True)
            return cls(llm=llm)

    # Patch load_agent_class to return our SpyAgent
    monkeypatch.setattr(
        "lottie.serve.service.load_agent_class",
        lambda root, name: SpyAgent,
    )
    # Patch load_input_model to return EchoInput
    monkeypatch.setattr(
        "lottie.serve.service.load_input_model",
        lambda root, name: EchoInput,
    )
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider(["ignored"]),
    )

    svc = AgentService(demo)
    result = svc.run_agent("echo", {"query": "hi"})

    assert from_project_called, "from_project was not called despite being present"
    assert result.output["result"] == "from_project_result"
