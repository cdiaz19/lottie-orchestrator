"""AssistantMesh integration — no real LLM/embedder/network (CLAUDE.md rule 5)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from lottie.llm import MockLLMProvider
from lottie.mesh.schema import MeshState, StepResult
from lottie.project.config import AgentConfig

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
"""


def _fixture_root(tmp_path: Path) -> Path:
    kb = tmp_path / "knowledge" / "global"
    kb.mkdir(parents=True)
    (kb / "multiagent.md").write_text(_KNOWLEDGE_DOC, encoding="utf-8")
    return tmp_path


def test_from_project_routes_research_then_critic_then_finish(tmp_path: Path) -> None:
    os.environ["LOTTIE_EMBEDDING_MODEL"] = "mock/embed"
    os.environ["LOTTIE_VECTOR_STORE"] = "memory"

    # Ordered mock script (single shared provider across all LLM calls):
    #  1 route -> "research"
    #  2 research.complete (agent reasoning)
    #  3 research summarizer
    #  4 route -> "critic"
    #  5 critic.complete
    #  6 route -> "FINISH"
    llm = MockLLMProvider(
        [
            "research",
            "Multi-agent systems coordinate agents via typed messages.",
            "Multi-agent systems coordinate agents.\n- typed messages\n- roles",
            "critic",
            "Accurate; add an example of routing.",
            "FINISH",
        ]
    )

    from agents.assistant.agent import AssistantMesh
    from agents.assistant.schema import AssistantInput

    mesh = AssistantMesh.from_project(
        llm=llm,
        root=_fixture_root(tmp_path),
        config=AgentConfig(provider="mock/x"),
        enable_benchmarks=False,
    )
    out = mesh.run(AssistantInput(task="What are multi-agent AI systems?"))

    assert [s.worker for s in out.history] == ["research", "critic"]
    assert out.final  # critic's review is the final
    assert mesh.last_metrics is not None and mesh.last_metrics.success


def test_routing_loop_contract_with_stub_nodes() -> None:
    from agents.assistant.agent import AssistantMesh
    from agents.assistant.schema import AssistantInput

    def _stub(name: str) -> Callable[[MeshState], MeshState]:
        def _run(state: MeshState) -> MeshState:
            return state.with_step(StepResult(worker=name, result=f"{name}!"))

        return _run

    mesh = AssistantMesh(
        MockLLMProvider(["research", "FINISH"]),
        nodes={"research": _stub("research"), "critic": _stub("critic")},
        descriptions={"research": "r", "critic": "c"},
        enable_benchmarks=False,
    )
    out = mesh.run(AssistantInput(task="x"))
    assert [s.worker for s in out.history] == ["research"]
    assert out.final == "research!"
