"""Regression: a worker running in a parallel LangGraph branch (its own thread)
must be audited with root=False, like a sequential nested worker — not root=True."""

from __future__ import annotations

import pytest

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

pytestmark = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="needs [mesh] extra")

from pathlib import Path  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from lottie.core import BaseAgent  # noqa: E402
from lottie.governance.audit import SqliteAuditLogger  # noqa: E402
from lottie.llm import MockLLMProvider  # noqa: E402
from lottie.mesh import MeshAgent, MeshNode, MeshState, StepResult  # noqa: E402
from lottie.mesh.schema import MeshInput  # noqa: E402


class _WIn(BaseModel):
    q: str


class _WOut(BaseModel):
    r: str


class _Worker(BaseAgent[_WIn, _WOut]):
    def _execute(self, data: _WIn) -> _WOut:
        return _WOut(r=f"{self.name}:done")


def _worker_node(name: str, worker: _Worker) -> MeshNode:
    def _run(state: MeshState) -> MeshState:
        out = worker.run(_WIn(q=state.task))
        return state.with_step(StepResult(worker=name, result=out.r))

    return _run


def test_parallel_worker_records_are_non_root(tmp_path: Path) -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    logger = SqliteAuditLogger(tmp_path)
    worker_a = _Worker(MockLLMProvider(["x"]), name="worker-a", audit=logger)
    worker_b = _Worker(MockLLMProvider(["x"]), name="worker-b", audit=logger)

    # supervisor: fan out a+b in parallel, then FINISH.
    mesh = MeshAgent(
        MockLLMProvider(["a, b", "FINISH", "FINISH"]),
        name="mesh",
        nodes={
            "a": _worker_node("a", worker_a),
            "b": _worker_node("b", worker_b),
        },
        descriptions={"a": "worker a", "b": "worker b"},
        engine=LangGraphEngine(),
    )
    mesh._audit = logger  # MeshAgent doesn't forward `audit`; inject for the test

    mesh.run(MeshInput(task="t"))

    rows = {r.agent: r for r in logger.query(limit=20)}
    assert "mesh" in rows and rows["mesh"].root is True  # top-level stays root
    assert rows["worker-a"].root is False  # nested parallel worker
    assert rows["worker-b"].root is False  # nested parallel worker
