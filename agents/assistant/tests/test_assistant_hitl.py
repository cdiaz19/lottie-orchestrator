from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

pytestmark = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="needs [mesh] extra")

from lottie.llm import MockLLMProvider  # noqa: E402
from lottie.mesh.schema import ApprovalDecision  # noqa: E402
from lottie.project.config import AgentConfig  # noqa: E402

_DOC = """\
---
id: kb/ma
layer: global
scope: project
tags: [ai]
status: curated
last_verified: "2025-01-01"
depends_on: []
---

Multi-agent systems coordinate specialized agents via typed messages.
"""


def _fixture_root(tmp_path: Path) -> Path:
    kb = tmp_path / "knowledge" / "global"
    kb.mkdir(parents=True)
    (kb / "ma.md").write_text(_DOC, encoding="utf-8")
    return tmp_path


def test_assistant_pauses_before_publisher_then_resumes(tmp_path: Path) -> None:
    os.environ["LOTTIE_EMBEDDING_MODEL"] = "mock/embed"
    os.environ["LOTTIE_VECTOR_STORE"] = "memory"

    # route: research -> publisher (interrupt) ; after resume -> FINISH
    # LLM calls before pause: route(research)=1, research.complete=2,
    #   summarizer=3, route(publisher)=4
    # after approve: publisher.complete=5, route(FINISH)=6
    llm = MockLLMProvider(
        [
            "research",
            "Multi-agent systems coordinate agents.",
            "Summary.\n- a\n- b",
            "publisher",
            "Published: multi-agent overview.",
            "FINISH",
        ]
    )

    from agents.assistant.agent import AssistantMesh
    from agents.assistant.schema import AssistantInput

    mesh = AssistantMesh.from_project(
        llm=llm,
        root=_fixture_root(tmp_path),
        config=AgentConfig(provider="mock/x", workers=["research", "critic", "publisher"],
                           interrupt_before=["publisher"]),
        enable_benchmarks=False,
    )
    out = mesh.run(AssistantInput(task="write an overview"))
    assert out.status == "interrupted"
    assert out.pending is not None and out.pending.worker == "publisher"
    assert out.thread_id is not None

    resumed = mesh.resume(out.thread_id, ApprovalDecision(action="approve"))
    assert resumed.status == "complete"
    assert any(s.worker == "publisher" for s in resumed.history)
