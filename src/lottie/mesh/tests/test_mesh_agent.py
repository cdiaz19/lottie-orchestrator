from __future__ import annotations

from collections.abc import Callable

import pytest

from lottie.llm import MockLLMProvider
from lottie.mesh.base import MeshAgent
from lottie.mesh.errors import CapabilityViolation, MeshStepLimitExceeded
from lottie.mesh.schema import MeshInput, MeshState, StepResult


def _stub_node(name: str) -> Callable[[MeshState], MeshState]:
    def _run(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker=name, result=f"{name}-ran"))

    return _run


def _make_mesh(llm: MockLLMProvider) -> MeshAgent:
    return MeshAgent(
        llm,
        nodes={"alpha": _stub_node("alpha"), "beta": _stub_node("beta")},
        descriptions={"alpha": "first worker", "beta": "second worker"},
        enable_benchmarks=False,
    )


def test_mesh_routes_until_finish() -> None:
    # supervisor script: alpha → beta → FINISH
    mesh = _make_mesh(MockLLMProvider(["alpha", "beta", "FINISH"]))
    out = mesh.run(MeshInput(task="do it"))
    assert [s.worker for s in out.history] == ["alpha", "beta"]
    assert out.final == "beta-ran"


def test_mesh_capability_violation_propagates() -> None:
    mesh = _make_mesh(MockLLMProvider(["hacker"]))
    with pytest.raises(CapabilityViolation):  # surfaces through run
        mesh.run(MeshInput(task="x"))


def test_mesh_step_limit() -> None:
    mesh = _make_mesh(MockLLMProvider(["alpha"] * 10))
    with pytest.raises(MeshStepLimitExceeded):
        mesh.run(MeshInput(task="x", max_steps=3))


def test_mesh_metrics_recorded() -> None:
    mesh = _make_mesh(MockLLMProvider(["alpha", "FINISH"]))
    mesh.run(MeshInput(task="x"))
    assert mesh.last_metrics is not None
    assert mesh.last_metrics.success is True


def test_mesh_rolls_up_worker_metrics() -> None:
    from datetime import UTC, datetime

    from lottie.core.metrics import RunMetrics

    fake = RunMetrics(
        name="worker",
        kind="agent",
        provider=None,
        timestamp=datetime.now(UTC),
        latency_ms=0.0,
        success=True,
        version=None,
        input_tokens=7,
        output_tokens=11,
        cost_usd=0.5,
    )

    mesh = MeshAgent(
        MockLLMProvider(["w", "FINISH"]),
        nodes={},
        descriptions={"w": "a worker"},
        enable_benchmarks=False,
    )

    def node(state: MeshState) -> MeshState:
        mesh._accumulate(fake)
        return state.with_step(StepResult(worker="w", result="done"))

    mesh._nodes = {"w": node}

    mesh.run(MeshInput(task="t"))

    assert mesh.last_metrics is not None
    # worker tokens/cost folded into the mesh's own run metrics
    assert mesh.last_metrics.input_tokens == 7
    assert mesh.last_metrics.output_tokens == 11
    assert mesh.last_metrics.cost_usd == 0.5
