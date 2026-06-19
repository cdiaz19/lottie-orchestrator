from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from lottie.llm import MockLLMProvider  # noqa: E402
from lottie.mesh import MeshAgent, MeshState, StepResult  # noqa: E402
from lottie.mesh.errors import ThreadNotFoundError  # noqa: E402
from lottie.mesh.langgraph_engine import LangGraphEngine  # noqa: E402
from lottie.mesh.schema import ApprovalDecision, MeshInput  # noqa: E402


def _worker(state: MeshState) -> MeshState:
    return state.with_step(StepResult(worker="w", result="did work"))


def _mesh(root: Path) -> MeshAgent:
    eng = LangGraphEngine(checkpoint="sqlite", root=root, interrupt_before=["w"])
    return MeshAgent(
        MockLLMProvider(["w", "FINISH", "FINISH"]),
        name="m",
        nodes={"w": _worker},
        descriptions={"w": "worker"},
        engine=eng,
    )


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_resume_unknown_thread_raises_typed(tmp_path: Path, action: str) -> None:
    mesh = _mesh(tmp_path)
    mesh.run(MeshInput(task="go"))  # creates a real checkpoint; we resume a DIFFERENT thread
    with pytest.raises(ThreadNotFoundError):
        mesh.resume("no-such-thread", ApprovalDecision(action=action))
