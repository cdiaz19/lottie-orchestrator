"""Rule 11 under real parallel fan-out: each worker's capability gate is active in its
own LangGraph branch/thread (contextvars are copied in, like the audit-depth + otel
gates). A worker calling a skill IT declared succeeds; one calling an undeclared skill
is blocked even from a parallel branch."""

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

from lottie.core import BaseAgent, BaseSkill  # noqa: E402
from lottie.governance.audit import SqliteAuditLogger  # noqa: E402
from lottie.governance.capability import (  # noqa: E402
    CapabilityDenied,
    build_capability_gate,
)
from lottie.llm import MockLLMProvider  # noqa: E402
from lottie.mesh import MeshAgent, MeshNode, MeshState, StepResult  # noqa: E402
from lottie.mesh.schema import MeshInput  # noqa: E402


class _SIn(BaseModel):
    x: int


class _SOut(BaseModel):
    y: int


class AlphaSkill(BaseSkill[_SIn, _SOut]):
    def _execute(self, data: _SIn) -> _SOut:
        return _SOut(y=data.x)


class _WIn(BaseModel):
    q: str


class _WOut(BaseModel):
    r: str


class _Worker(BaseAgent[_WIn, _WOut]):
    def __init__(self, name: str, caps: list[str], logger: SqliteAuditLogger) -> None:
        super().__init__(MockLLMProvider(["x"]), name=name, audit=logger)
        self.set_capability_gate(build_capability_gate(capabilities=caps))
        self._skill = AlphaSkill(enable_benchmarks=False)

    def _execute(self, data: _WIn) -> _WOut:
        out = self._skill.run(_SIn(x=1))  # capability name "alpha"
        return _WOut(r=f"{self.name}:{out.y}")


def _node(name: str, worker: _Worker) -> MeshNode:
    def _run(state: MeshState) -> MeshState:
        out = worker.run(_WIn(q=state.task))
        return state.with_step(StepResult(worker=name, result=out.r))

    return _run


def _mesh(logger: SqliteAuditLogger, worker_a: _Worker, worker_b: _Worker) -> MeshAgent:
    mesh = MeshAgent(
        MockLLMProvider(["a, b", "FINISH", "FINISH"]),
        name="mesh",
        nodes={"a": _node("a", worker_a), "b": _node("b", worker_b)},
        descriptions={"a": "worker a", "b": "worker b"},
        engine=__import__(
            "lottie.mesh.langgraph_engine", fromlist=["LangGraphEngine"]
        ).LangGraphEngine(),
    )
    mesh._audit = logger
    return mesh


def test_parallel_workers_with_declared_skill_succeed(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    a = _Worker("worker-a", ["alpha"], logger)
    b = _Worker("worker-b", ["alpha"], logger)
    _mesh(logger, a, b).run(MeshInput(task="t"))
    rows = {r.agent: r for r in logger.query(limit=20)}
    assert rows["worker-a"].status == "ok"
    assert rows["worker-b"].status == "ok"


def test_parallel_worker_with_undeclared_skill_is_blocked(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    a = _Worker("worker-a", ["alpha"], logger)
    # worker-b declares "beta" only, but calls AlphaSkill -> must be denied in its branch
    b = _Worker("worker-b", ["beta"], logger)
    with pytest.raises(CapabilityDenied):
        _mesh(logger, a, b).run(MeshInput(task="t"))
    # worker-b's own run was audited as an error (its gate was active in the branch)
    rows = [r for r in logger.query(limit=20) if r.agent == "worker-b"]
    assert rows and rows[0].status == "error"
