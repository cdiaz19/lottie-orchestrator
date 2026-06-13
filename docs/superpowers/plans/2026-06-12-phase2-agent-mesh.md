# Phase 2 — Agent Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the core multi-agent mesh — a supervisor `BaseAgent` that routes a task to declared worker agents by LLM intent, over conditional edges, with fully typed Pydantic state, testable end-to-end on `MockLLMProvider`.

**Architecture:** A mesh **is a `BaseAgent`**, so it reuses the existing `instantiate_agent`/`from_project` DI seam, `AgentService`, `lottie run`, serve/MCP, and benchmark unchanged. New framework code lives under `src/lottie/mesh/`: a `MeshEngine` ABC with a hand-rolled `LocalEngine` default (LangGraph deferred to Phase 3), a `SupervisorRouter` that validates the LLM's choice against the declared worker set (= capability enforcement), and a `MeshAgent` base whose `_execute` runs the engine and rolls worker token/cost into its own run context. A reference `agents/assistant/` mesh routes between the existing `research` agent and a new minimal `critic` agent.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, pytest. No new third-party dependency (LangGraph deferred). `mypy --strict src` + `ruff check .` gate every file.

---

## Spec reference

Implements `docs/superpowers/specs/2026-06-12-phase2-agent-mesh-design.md`. Decisions D1–D7 there are locked. Re-read it before starting.

## File structure (created/modified)

**New framework package — `src/lottie/mesh/`:**
- `__init__.py` — public exports (schemas, engine, router, `MeshAgent`, errors).
- `schema.py` — `MeshState`, `StepResult`, `RouteDecision`, `MeshInput`, `MeshOutput`, `FINISH`.
- `errors.py` — `MeshError`, `CapabilityViolation`, `MeshStepLimitExceeded`.
- `router.py` — `SupervisorRouter`.
- `engine.py` — `MeshEngine` ABC + `MeshNode`, `RouteFn` type aliases.
- `local.py` — `LocalEngine`.
- `base.py` — `MeshAgent(BaseAgent[MeshInput, MeshOutput])`.
- `tests/test_schema.py`, `tests/test_router.py`, `tests/test_local_engine.py`, `tests/test_mesh_agent.py`, `tests/__init__.py`.

**New reference units:**
- `agents/critic/` — minimal worker (`agent.py`, `schema.py`, `prompts.py`, `config.yaml`, `AGENT.md`, `__init__.py`, `tests/test_critic.py`).
- `agents/assistant/` — the mesh (`agent.py`, `schema.py`, `config.yaml`, `AGENT.md`, `evals.yaml`, `__init__.py`, `tests/test_assistant.py`).

**Modified framework:**
- `src/lottie/project/config.py` — add `workers: list[str] = []` to `AgentConfig`.
- `tests/contracts/test_mesh_schema.py` — new contract round-trip tests.

**Docs (final task):** `CLAUDE.md`, `LOTTIE_PHASE0_SPEC.md` release row, `README.md` roadmap row.

---

## Conventions (read once)

- TDD throughout: failing test → run red → implement → run green → commit. One conventional commit per task.
- Run tools **from the project dir** (`/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator`), with the venv active: `source .venv/bin/activate`.
- After every task: `mypy --strict src` and `ruff check .` must be clean before committing.
- `MeshNode = Callable[[MeshState], MeshState]`. `RouteFn = Callable[[MeshState], RouteDecision]`.
- The `FINISH` sentinel is the literal string `"FINISH"` (a worker may never be named `FINISH`).

---

## Sub-phase A — Mesh schemas + errors

### Task 1: Mesh schemas (pure Pydantic)

**Files:**
- Create: `src/lottie/mesh/__init__.py` (empty for now), `src/lottie/mesh/schema.py`
- Create: `src/lottie/mesh/tests/__init__.py`, `src/lottie/mesh/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

`src/lottie/mesh/tests/test_schema.py`:
```python
from __future__ import annotations

from lottie.mesh.schema import (
    FINISH,
    MeshInput,
    MeshOutput,
    MeshState,
    RouteDecision,
    StepResult,
)


def test_mesh_state_defaults_and_add_step() -> None:
    state = MeshState(task="summarize X")
    assert state.history == [] and state.final is None
    nxt = state.with_step(StepResult(worker="research", result="r"))
    assert [s.worker for s in nxt.history] == ["research"]
    # original is not mutated (with_step returns a new state)
    assert state.history == []


def test_route_decision_and_finish_sentinel() -> None:
    assert FINISH == "FINISH"
    assert RouteDecision(next="research").next == "research"
    assert RouteDecision(next=FINISH).next == "FINISH"


def test_mesh_io_models() -> None:
    assert MeshInput(task="t").max_steps == 8
    out = MeshOutput(final="done", history=[StepResult(worker="critic", result="ok")])
    assert out.final == "done" and out.history[0].worker == "critic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/lottie/mesh/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.mesh.schema'`.

- [ ] **Step 3: Write minimal implementation**

`src/lottie/mesh/schema.py`:
```python
"""Typed state and I/O models for the agent mesh (all Pydantic — CLAUDE.md rule 2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

FINISH = "FINISH"
"""Sentinel returned by the supervisor to end the routing loop."""


class StepResult(BaseModel):
    """One worker invocation's outcome, recorded in mesh history."""

    worker: str
    result: str
    metadata: dict[str, str] = {}


class MeshState(BaseModel):
    """Evolving, typed state threaded through every mesh node."""

    task: str
    history: list[StepResult] = []
    final: str | None = None

    def with_step(self, step: StepResult) -> MeshState:
        """Return a new state with `step` appended (does not mutate self)."""
        return self.model_copy(update={"history": [*self.history, step]})


class RouteDecision(BaseModel):
    """Supervisor's choice of the next worker, or FINISH."""

    next: str


class MeshInput(BaseModel):
    """Input to a mesh agent."""

    task: str
    max_steps: int = Field(default=8, ge=1)


class MeshOutput(BaseModel):
    """Output of a mesh agent."""

    final: str
    history: list[StepResult] = []
```

Leave `src/lottie/mesh/__init__.py` empty for now (filled in Task 7).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/lottie/mesh/tests/test_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/mesh/__init__.py src/lottie/mesh/schema.py src/lottie/mesh/tests/
git commit -m "feat(mesh): typed MeshState + StepResult + RouteDecision + I/O models"
```

### Task 2: Mesh errors

**Files:**
- Create: `src/lottie/mesh/errors.py`
- Test: `src/lottie/mesh/tests/test_schema.py` (extend — add error import test)

- [ ] **Step 1: Write the failing test** — append to `test_schema.py`:
```python
def test_mesh_errors_hierarchy() -> None:
    from lottie.mesh.errors import (
        CapabilityViolation,
        MeshError,
        MeshStepLimitExceeded,
    )

    assert issubclass(CapabilityViolation, MeshError)
    assert issubclass(MeshStepLimitExceeded, MeshError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/lottie/mesh/tests/test_schema.py::test_mesh_errors_hierarchy -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.mesh.errors'`.

- [ ] **Step 3: Write minimal implementation**

`src/lottie/mesh/errors.py`:
```python
"""Typed mesh errors. Transport-agnostic — no typer."""

from __future__ import annotations


class MeshError(Exception):
    """Base for all mesh-orchestration errors."""


class CapabilityViolation(MeshError):
    """The supervisor routed to a worker not declared in config.yaml `workers`."""


class MeshStepLimitExceeded(MeshError):
    """The routing loop hit `max_steps` without reaching FINISH."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/lottie/mesh/tests/test_schema.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/mesh/errors.py src/lottie/mesh/tests/test_schema.py
git commit -m "feat(mesh): typed mesh error hierarchy"
```

---

## Sub-phase B — Router + engine

### Task 3: SupervisorRouter (LLM intent, validated to declared set)

**Files:**
- Create: `src/lottie/mesh/router.py`
- Test: `src/lottie/mesh/tests/test_router.py`

The router takes a `complete` callable (`Callable[[list[Message]], LLMResponse]`) so it stays decoupled from `LLMProvider` and so token usage accumulates in the caller's run context (the mesh passes `self.complete`).

- [ ] **Step 1: Write the failing test**

`src/lottie/mesh/tests/test_router.py`:
```python
from __future__ import annotations

import pytest

from lottie.llm import LLMResponse, Message, TokenUsage
from lottie.mesh.errors import CapabilityViolation
from lottie.mesh.router import SupervisorRouter
from lottie.mesh.schema import MeshState


def _complete_returning(text: str):
    def _complete(messages: list[Message]) -> LLMResponse:
        return LLMResponse(content=text, usage=TokenUsage(), model="mock/mock-model")

    return _complete


_WORKERS = {"research": "Finds and summarizes knowledge.", "critic": "Reviews a draft."}


def test_router_returns_validated_worker() -> None:
    router = SupervisorRouter(_complete_returning("research"))
    decision = router.route(MeshState(task="t"), _WORKERS)
    assert decision.next == "research"


def test_router_accepts_finish() -> None:
    router = SupervisorRouter(_complete_returning("FINISH"))
    assert router.route(MeshState(task="t"), _WORKERS).next == "FINISH"


def test_router_is_case_insensitive_and_trims() -> None:
    router = SupervisorRouter(_complete_returning("  Critic \n"))
    assert router.route(MeshState(task="t"), _WORKERS).next == "critic"


def test_router_rejects_undeclared_worker() -> None:
    router = SupervisorRouter(_complete_returning("hacker"))
    with pytest.raises(CapabilityViolation):
        router.route(MeshState(task="t"), _WORKERS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/lottie/mesh/tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.mesh.router'`.

- [ ] **Step 3: Write minimal implementation**

`src/lottie/mesh/router.py`:
```python
"""Supervisor routing: ask the LLM which worker runs next, validate the answer."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from lottie.llm import LLMResponse, Message
from lottie.mesh.errors import CapabilityViolation
from lottie.mesh.schema import FINISH, MeshState, RouteDecision

CompleteFn = Callable[[list[Message]], LLMResponse]

_SYSTEM = (
    "You are a supervisor routing a task to the single best-suited worker. "
    "Reply with ONLY the worker name, or FINISH when the task is complete. "
    "Never reply with anything else."
)


class SupervisorRouter:
    """Routes by LLM intent, constrained to a declared worker set."""

    def __init__(self, complete: CompleteFn) -> None:
        self._complete = complete

    def route(self, state: MeshState, workers: Mapping[str, str]) -> RouteDecision:
        roster = "\n".join(f"- {name}: {desc}" for name, desc in workers.items())
        done = "\n".join(f"[{s.worker}] {s.result}" for s in state.history) or "(none yet)"
        user = (
            f"Task: {state.task}\n\n"
            f"Workers:\n{roster}\n\n"
            f"Work done so far:\n{done}\n\n"
            "Next worker (or FINISH):"
        )
        raw = self._complete(
            [Message(role="system", content=_SYSTEM), Message(role="user", content=user)]
        ).content.strip()
        return RouteDecision(next=self._resolve(raw, workers))

    @staticmethod
    def _resolve(raw: str, workers: Mapping[str, str]) -> str:
        if raw.upper() == FINISH:
            return FINISH
        lowered = {name.lower(): name for name in workers}
        if raw.lower() in lowered:
            return lowered[raw.lower()]
        raise CapabilityViolation(
            f"supervisor chose undeclared worker {raw!r}; allowed: {sorted(workers)}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/lottie/mesh/tests/test_router.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/mesh/router.py src/lottie/mesh/tests/test_router.py
git commit -m "feat(mesh): SupervisorRouter with capability-validated routing"
```

### Task 4: MeshEngine ABC + LocalEngine

**Files:**
- Create: `src/lottie/mesh/engine.py`, `src/lottie/mesh/local.py`
- Test: `src/lottie/mesh/tests/test_local_engine.py`

- [ ] **Step 1: Write the failing test**

`src/lottie/mesh/tests/test_local_engine.py`:
```python
from __future__ import annotations

import pytest

from lottie.mesh.errors import MeshStepLimitExceeded
from lottie.mesh.local import LocalEngine
from lottie.mesh.schema import FINISH, MeshState, RouteDecision, StepResult


def _node(name: str):
    def _run(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker=name, result=f"{name}:{state.task}"))

    return _run


def test_engine_runs_route_until_finish() -> None:
    nodes = {"a": _node("a"), "b": _node("b")}
    script = iter(["a", "b", FINISH])

    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=next(script))

    final = LocalEngine().run(MeshState(task="t"), nodes=nodes, route=route, max_steps=8)
    assert [s.worker for s in final.history] == ["a", "b"]
    # final is set to the last step's result
    assert final.final == "b:t"


def test_engine_finish_with_no_steps_sets_empty_final() -> None:
    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=FINISH)

    final = LocalEngine().run(MeshState(task="t"), nodes={}, route=route, max_steps=8)
    assert final.final == "" and final.history == []


def test_engine_raises_on_step_limit() -> None:
    nodes = {"a": _node("a")}

    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next="a")  # never FINISH

    with pytest.raises(MeshStepLimitExceeded):
        LocalEngine().run(MeshState(task="t"), nodes=nodes, route=route, max_steps=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/lottie/mesh/tests/test_local_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.mesh.local'`.

- [ ] **Step 3: Write minimal implementation**

`src/lottie/mesh/engine.py`:
```python
"""MeshEngine ABC. LocalEngine is the v1 default; a LangGraphEngine adapter
lands in Phase 3 behind this same interface (keep `run` engine-agnostic)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from lottie.mesh.schema import MeshState, RouteDecision

MeshNode = Callable[[MeshState], MeshState]
RouteFn = Callable[[MeshState], RouteDecision]


class MeshEngine(ABC):
    """Drives a supervisor→worker loop over typed state."""

    @abstractmethod
    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
    ) -> MeshState:
        """Loop: route → dispatch chosen node → repeat until FINISH or max_steps."""
```

`src/lottie/mesh/local.py`:
```python
"""Hand-rolled, dependency-free mesh engine (Phase 2 default)."""

from __future__ import annotations

from collections.abc import Mapping

from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.errors import MeshStepLimitExceeded
from lottie.mesh.schema import FINISH, MeshState


class LocalEngine(MeshEngine):
    """Deterministic in-process supervisor loop."""

    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
    ) -> MeshState:
        state = initial
        for _ in range(max_steps):
            decision = route(state)
            if decision.next == FINISH:
                last = state.history[-1].result if state.history else ""
                return state.model_copy(update={"final": last})
            state = nodes[decision.next](state)
        raise MeshStepLimitExceeded(
            f"routing loop exceeded max_steps={max_steps} without FINISH"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/lottie/mesh/tests/test_local_engine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/mesh/engine.py src/lottie/mesh/local.py src/lottie/mesh/tests/test_local_engine.py
git commit -m "feat(mesh): MeshEngine ABC + hand-rolled LocalEngine"
```

### Task 5: `AgentConfig.workers` field

**Files:**
- Modify: `src/lottie/project/config.py:29-35` (`AgentConfig`)
- Test: `src/lottie/mesh/tests/test_schema.py` (extend)

- [ ] **Step 1: Write the failing test** — append to `test_schema.py`:
```python
def test_agent_config_has_workers_field() -> None:
    from lottie.project.config import AgentConfig

    cfg = AgentConfig(provider="mock/x", workers=["research", "critic"])
    assert cfg.workers == ["research", "critic"]
    # default empty for non-mesh agents
    assert AgentConfig(provider="mock/x").workers == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/lottie/mesh/tests/test_schema.py::test_agent_config_has_workers_field -v`
Expected: FAIL — `AttributeError: 'AgentConfig' object has no attribute 'workers'`.

- [ ] **Step 3: Write minimal implementation** — in `src/lottie/project/config.py`, add the field to `AgentConfig`:
```python
class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    model_params: dict[str, object] = {}
    capabilities: list[str] = []
    policies: list[str] = []
    workers: list[str] = []  # mesh routing allow-set (capability enforcement)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/lottie/mesh/tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/project/config.py src/lottie/mesh/tests/test_schema.py
git commit -m "feat(config): add workers allow-set to AgentConfig"
```

---

## Sub-phase C — MeshAgent base

### Task 6: MeshAgent (BaseAgent subclass, metrics rollup)

`MeshAgent` is the reusable base every mesh extends. It holds the worker node adapters, their descriptions, a router, and an engine; `_execute` builds `MeshState` from `MeshInput`, runs the engine, and returns `MeshOutput`. Worker token/cost is rolled into the active run context **per node-call** via `_accumulate` (a worker routed twice is counted twice; mesh wall-clock latency already includes workers because they run synchronously inside `_execute`, so no `core/` change is needed).

**Files:**
- Create: `src/lottie/mesh/base.py`
- Test: `src/lottie/mesh/tests/test_mesh_agent.py`

- [ ] **Step 1: Write the failing test**

`src/lottie/mesh/tests/test_mesh_agent.py`:
```python
from __future__ import annotations

import pytest

from lottie.llm import MockLLMProvider
from lottie.mesh.base import MeshAgent
from lottie.mesh.errors import MeshStepLimitExceeded
from lottie.mesh.schema import MeshInput, MeshState, StepResult


def _stub_node(name: str):
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
    with pytest.raises(Exception):  # CapabilityViolation surfaces through run
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/lottie/mesh/tests/test_mesh_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.mesh.base'`.

- [ ] **Step 3: Write minimal implementation**

`src/lottie/mesh/base.py`:
```python
"""MeshAgent — a BaseAgent that orchestrates worker agents via a supervisor loop."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from lottie.core import BaseAgent
from lottie.core.metrics import RunMetrics
from lottie.llm import LLMProvider, TokenUsage
from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.local import LocalEngine
from lottie.mesh.router import SupervisorRouter
from lottie.mesh.schema import MeshInput, MeshOutput, MeshState, RouteDecision
from lottie.memory.base import MemoryClient


class MeshAgent(BaseAgent[MeshInput, MeshOutput]):
    """Routes a task across declared worker nodes until the supervisor says FINISH."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        nodes: Mapping[str, MeshNode],
        descriptions: Mapping[str, str],
        engine: MeshEngine | None = None,
        max_steps: int = 8,
        name: str | None = None,
        memory: MemoryClient | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            llm,
            name=name,
            memory=memory,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._nodes = dict(nodes)
        self._descriptions = dict(descriptions)
        self._engine = engine or LocalEngine()
        self._max_steps = max_steps
        self._router = SupervisorRouter(self.complete)

    def _accumulate(self, metrics: RunMetrics | None) -> None:
        """Fold one worker run's tokens/cost into the active mesh run context."""
        if metrics is None or self._active_ctx is None:
            return
        self._active_ctx.add_usage(
            TokenUsage(
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
            ),
            metrics.cost_usd,
        )

    def _execute(self, data: MeshInput) -> MeshOutput:
        route: RouteFn = lambda state: self._router.route(state, self._descriptions)  # noqa: E731
        final = self._engine.run(
            MeshState(task=data.task),
            nodes=self._nodes,
            route=route,
            max_steps=data.max_steps or self._max_steps,
        )
        return MeshOutput(final=final.final or "", history=final.history)
```

Note: `_accumulate` is wired into the worker node adapters in Task 8 (the adapters call `self._accumulate(worker.last_metrics)` right after running each worker). It is defined here so the rollup contract lives on the base.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/lottie/mesh/tests/test_mesh_agent.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/mesh/base.py src/lottie/mesh/tests/test_mesh_agent.py
git commit -m "feat(mesh): MeshAgent base with engine + router + metrics rollup"
```

### Task 7: Mesh package exports

**Files:**
- Modify: `src/lottie/mesh/__init__.py`
- Test: `src/lottie/mesh/tests/test_schema.py` (extend)

- [ ] **Step 1: Write the failing test** — append to `test_schema.py`:
```python
def test_package_exports() -> None:
    import lottie.mesh as m

    for sym in ("MeshAgent", "MeshInput", "MeshOutput", "MeshState", "LocalEngine"):
        assert hasattr(m, sym), sym
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/lottie/mesh/tests/test_schema.py::test_package_exports -v`
Expected: FAIL — `AssertionError: MeshAgent`.

- [ ] **Step 3: Write minimal implementation**

`src/lottie/mesh/__init__.py`:
```python
"""Agent mesh — supervisor→worker orchestration over typed Pydantic state."""

from lottie.mesh.base import MeshAgent
from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.errors import CapabilityViolation, MeshError, MeshStepLimitExceeded
from lottie.mesh.local import LocalEngine
from lottie.mesh.router import SupervisorRouter
from lottie.mesh.schema import (
    FINISH,
    MeshInput,
    MeshOutput,
    MeshState,
    RouteDecision,
    StepResult,
)

__all__ = [
    "FINISH",
    "CapabilityViolation",
    "LocalEngine",
    "MeshAgent",
    "MeshEngine",
    "MeshError",
    "MeshInput",
    "MeshNode",
    "MeshOutput",
    "MeshState",
    "MeshStepLimitExceeded",
    "RouteDecision",
    "RouteFn",
    "SupervisorRouter",
    "StepResult",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/lottie/mesh/tests/ -v`
Expected: PASS (all mesh tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/mesh/__init__.py src/lottie/mesh/tests/test_schema.py
git commit -m "feat(mesh): public package exports"
```

---

## Sub-phase D — Reference workers + mesh

### Task 8: `critic` worker agent

A minimal knowledge-free worker: one `self.complete` call that reviews/refines a piece of text. Gives the mesh a second real routing target.

> Scaffold first per CLAUDE.md rule 4: `lottie create agent critic`. Then replace the generated `schema.py`, `prompts.py`, `agent.py`, `config.yaml`, and write `AGENT.md` before logic.

**Files:**
- Create (scaffold then edit): `agents/critic/agent.py`, `agents/critic/schema.py`, `agents/critic/prompts.py`, `agents/critic/config.yaml`, `agents/critic/AGENT.md`, `agents/critic/__init__.py`
- Test: `agents/critic/tests/test_critic.py`

- [ ] **Step 1: Scaffold + write the failing test**

```bash
lottie create agent critic
```

`agents/critic/tests/test_critic.py`:
```python
from __future__ import annotations

from lottie.llm import MockLLMProvider

from agents.critic.agent import CriticAgent
from agents.critic.schema import CriticInput


def test_critic_reviews_text() -> None:
    agent = CriticAgent(MockLLMProvider(["Concise and accurate; tighten the intro."]))
    out = agent.run(CriticInput(text="A long draft about multi-agent systems."))
    assert out.review
    assert agent.last_metrics is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest agents/critic/tests/test_critic.py -v`
Expected: FAIL — import error / `CriticInput` not defined (scaffold uses a different schema).

- [ ] **Step 3: Write minimal implementation**

`agents/critic/schema.py`:
```python
"""Typed I/O for CriticAgent."""

from __future__ import annotations

from pydantic import BaseModel


class CriticInput(BaseModel):
    text: str


class CriticOutput(BaseModel):
    review: str
```

`agents/critic/prompts.py`:
```python
SYSTEM_PROMPT = (
    "You are a rigorous reviewer. Given a draft, return a short critique: what is "
    "accurate, what is missing, and one concrete improvement. Be terse."
)
```

`agents/critic/agent.py`:
```python
"""CriticAgent — single-LLM-call reviewer used as a mesh worker."""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message

from .prompts import SYSTEM_PROMPT
from .schema import CriticInput, CriticOutput


class CriticAgent(BaseAgent[CriticInput, CriticOutput]):
    """Reviews a draft and returns a terse critique."""

    def _execute(self, data: CriticInput) -> CriticOutput:
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=data.text),
            ]
        )
        return CriticOutput(review=response.content)
```

`agents/critic/config.yaml`:
```yaml
provider: anthropic/claude-sonnet-4-6
model_params:
  temperature: 0.2
  max_tokens: 1024
capabilities: []
policies:
  - base
```

`agents/critic/AGENT.md` — write a short doc: purpose (terse reviewer worker), input (`CriticInput{text}`), output (`CriticOutput{review}`), that it makes exactly one LLM call via `self.complete`, and that it is used as a worker node in the `assistant` mesh.

Ensure `agents/critic/__init__.py` exists (scaffold creates it; leave as-is).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest agents/critic/tests/test_critic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/critic/
git commit -m "feat(agents): minimal CriticAgent worker"
```

### Task 9: `assistant` reference mesh

The mesh unit. Its `schema.py` re-exports the mesh I/O models under the discovery-friendly `AssistantInput`/`AssistantOutput` names (mirrors the `<Name>Input`/`<Name>Output` convention `load_input_model` relies on). `agent.py` defines `AssistantMesh(MeshAgent)` plus a `from_project` that wires the real `research` + `critic` workers as typed node adapters and rolls their metrics up.

**Files:**
- Create: `agents/assistant/schema.py`, `agents/assistant/agent.py`, `agents/assistant/config.yaml`, `agents/assistant/AGENT.md`, `agents/assistant/__init__.py`
- Test: deferred to Task 10.

- [ ] **Step 1: Write `agents/assistant/schema.py`**
```python
"""Typed I/O for the assistant mesh (discovery-named aliases over mesh models)."""

from __future__ import annotations

from lottie.mesh.schema import MeshInput, MeshOutput


class AssistantInput(MeshInput):
    """Task input for the assistant mesh."""


class AssistantOutput(MeshOutput):
    """Final answer + step history from the assistant mesh."""
```

- [ ] **Step 2: Write `agents/assistant/agent.py`**
```python
"""AssistantMesh — supervisor routing between the research and critic workers."""

from __future__ import annotations

from pathlib import Path

from agents.critic.agent import CriticAgent
from agents.critic.schema import CriticInput
from agents.research.agent import ResearchAgent
from agents.research.schema import ResearchInput

from lottie.core import BaseAgent
from lottie.llm import LLMProvider
from lottie.mesh import MeshAgent, MeshNode, MeshState, StepResult
from lottie.project.config import AgentConfig

_DESCRIPTIONS = {
    "research": "Retrieves and synthesizes knowledge to answer the task.",
    "critic": "Reviews the latest draft and suggests one concrete improvement.",
}


class AssistantMesh(MeshAgent):
    """Reference mesh: research → critic, supervised by the injected LLM."""

    @classmethod
    def from_project(
        cls,
        *,
        llm: LLMProvider,
        root: Path,
        config: AgentConfig,
        enable_benchmarks: bool | None = None,
    ) -> AssistantMesh:
        research = ResearchAgent.from_project(
            llm=llm, root=root, config=config, enable_benchmarks=enable_benchmarks
        )
        critic = CriticAgent(llm, enable_benchmarks=enable_benchmarks)
        mesh = cls(
            llm,
            nodes={},  # replaced below once `mesh` exists (adapters close over it)
            descriptions=_DESCRIPTIONS,
            enable_benchmarks=enable_benchmarks,
        )
        mesh._nodes = {
            "research": mesh._research_node(research),
            "critic": mesh._critic_node(critic),
        }
        return mesh

    def _research_node(self, research: ResearchAgent) -> MeshNode:
        def _run(state: MeshState) -> MeshState:
            out = research.run(ResearchInput(query=state.task))
            self._accumulate(research.last_metrics)
            return state.with_step(StepResult(worker="research", result=out.digest))

        return _run

    def _critic_node(self, critic: CriticAgent) -> MeshNode:
        def _run(state: MeshState) -> MeshState:
            draft = state.history[-1].result if state.history else state.task
            out = critic.run(CriticInput(text=draft))
            self._accumulate(critic.last_metrics)
            return state.with_step(StepResult(worker="critic", result=out.review))

        return _run
```

Note: `nodes={}` then assigning `mesh._nodes` is the documented seam for adapters that must close over the mesh instance (so `_accumulate` is reachable). Keep the constructor's `descriptions` as the routing roster.

- [ ] **Step 3: Write `agents/assistant/config.yaml`**
```yaml
provider: anthropic/claude-sonnet-4-6
model_params:
  temperature: 0.2
  max_tokens: 2048
capabilities: []
policies:
  - base
workers:
  - research
  - critic
```

- [ ] **Step 4: Write `agents/assistant/AGENT.md` and `agents/assistant/__init__.py`**

`AGENT.md` — document: it is a mesh (`MeshAgent` subclass), input `AssistantInput{task, max_steps}`, output `AssistantOutput{final, history}`, the declared `workers: [research, critic]` allow-set (= capability enforcement at routing), that the supervisor uses the injected LLM, and that worker token/cost rolls into the mesh metrics. Note routing terminates at `FINISH` or `max_steps`.

`agents/assistant/__init__.py`:
```python
```
(empty file)

- [ ] **Step 5: Verify it imports + commit**

Run: `python -c "from agents.assistant.agent import AssistantMesh; print('ok')"`
Expected: prints `ok` (no import errors).
Run: `mypy --strict src && ruff check .`
Expected: clean.

```bash
git add agents/assistant/schema.py agents/assistant/agent.py agents/assistant/config.yaml agents/assistant/AGENT.md agents/assistant/__init__.py
git commit -m "feat(agents): assistant reference mesh (research + critic)"
```

---

## Sub-phase E — Integration tests, contracts, benchmark, docs

### Task 10: Assistant integration tests (hermetic)

Two coverage areas: (A) `from_project` wires real workers and runs end-to-end on mock providers; (B) the routing loop's typed contract via constructor-injected stub nodes (fast, no knowledge stack).

**Files:**
- Test: `agents/assistant/tests/__init__.py`, `agents/assistant/tests/test_assistant.py`

- [ ] **Step 1: Write the failing test**

`agents/assistant/tests/test_assistant.py`:
```python
"""AssistantMesh integration — no real LLM/embedder/network (CLAUDE.md rule 5)."""

from __future__ import annotations

import os
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
        llm=llm, root=_fixture_root(tmp_path), config=AgentConfig(provider="mock/x"),
        enable_benchmarks=False,
    )
    out = mesh.run(AssistantInput(task="What are multi-agent AI systems?"))

    assert [s.worker for s in out.history] == ["research", "critic"]
    assert out.final  # critic's review is the final
    assert mesh.last_metrics is not None and mesh.last_metrics.success


def test_routing_loop_contract_with_stub_nodes() -> None:
    from agents.assistant.agent import AssistantMesh
    from agents.assistant.schema import AssistantInput

    def _stub(name: str):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest agents/assistant/tests/test_assistant.py -v`
Expected: FAIL initially only if anything is mis-wired; if Task 9 is correct it may already pass. If it fails, fix `agent.py` until green (do not change the test).

- [ ] **Step 3: Make it pass**

If `from_project`’s node-replacement seam or `_accumulate` wiring is off, correct `agents/assistant/agent.py`. Re-run until green.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest agents/assistant/tests/test_assistant.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add agents/assistant/tests/
git commit -m "test(agents): assistant mesh integration (from_project + routing contract)"
```

### Task 11: Contract tests for mesh schemas

**Files:**
- Create: `tests/contracts/test_mesh_schema.py`

- [ ] **Step 1: Write the test**

`tests/contracts/test_mesh_schema.py`:
```python
"""Round-trip contract tests for mesh schemas (no raw dict/str crosses a boundary)."""

from __future__ import annotations

from lottie.mesh.schema import (
    MeshInput,
    MeshOutput,
    MeshState,
    RouteDecision,
    StepResult,
)


def test_mesh_state_roundtrip() -> None:
    state = MeshState(
        task="t", history=[StepResult(worker="research", result="r")], final="f"
    )
    assert MeshState.model_validate_json(state.model_dump_json()) == state


def test_route_decision_roundtrip() -> None:
    d = RouteDecision(next="critic")
    assert RouteDecision.model_validate_json(d.model_dump_json()) == d


def test_mesh_io_roundtrip() -> None:
    mi = MeshInput(task="t", max_steps=4)
    assert MeshInput.model_validate_json(mi.model_dump_json()) == mi
    mo = MeshOutput(final="f", history=[StepResult(worker="critic", result="ok")])
    assert MeshOutput.model_validate_json(mo.model_dump_json()) == mo
```

- [ ] **Step 2: Run to verify**

Run: `pytest tests/contracts/test_mesh_schema.py -v`
Expected: PASS (3 tests). (If a schema gap surfaces, fix `schema.py` and re-run.)

- [ ] **Step 3: Commit**

```bash
git add tests/contracts/test_mesh_schema.py
git commit -m "test(contracts): mesh schema round-trips"
```

### Task 12: Benchmark eval suite for the mesh

`evals.yaml` is data; the hermetic check invokes the benchmark runner with a monkeypatched `build_provider` returning a precisely-seeded `MockLLMProvider`. One eval case keeps the mock script bounded and deterministic (the runner constructs the agent once and runs all cases through the shared provider).

**Files:**
- Create: `agents/assistant/evals.yaml`
- Test: `agents/assistant/tests/test_benchmark.py`

- [ ] **Step 1: Write `agents/assistant/evals.yaml`**
```yaml
# AssistantMesh eval suite — agents/assistant/evals.yaml
#
# Structural (always-true with a mock provider): confirms the mesh runs the
# supervisor→worker→FINISH loop end-to-end and produces a non-empty `final`.
# Against a real provider + populated knowledge, expect a substantive answer.
#
# LLM call budget for this single case (one shared MockLLMProvider):
#   route(research), research.complete, summarizer, route(critic),
#   critic.complete, route(FINISH) = 6 responses.
cases:
  - name: research-then-critic produces a final answer
    input:
      task: "What are multi-agent AI systems?"
      max_steps: 8
    expect:
      contains:
        final: ""
```

- [ ] **Step 2: Write the failing test**

`agents/assistant/tests/test_benchmark.py`:
```python
"""Hermetic benchmark run for the assistant mesh (mock provider, no network)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lottie.benchmark.runner import benchmark
from lottie.llm import MockLLMProvider

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


def _seeded_provider(_name: str) -> MockLLMProvider:
    return MockLLMProvider(
        [
            "research",
            "Multi-agent systems coordinate agents.",
            "Summary.\n- a\n- b",
            "critic",
            "Looks good; add an example.",
            "FINISH",
        ]
    )


def test_benchmark_assistant_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["LOTTIE_EMBEDDING_MODEL"] = "mock/embed"
    os.environ["LOTTIE_VECTOR_STORE"] = "memory"

    # Build a fixture project: lottie.yaml + knowledge + the assistant unit symlinked in.
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from lottie.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    kb = demo / "knowledge" / "global"
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "ma.md").write_text(_DOC, encoding="utf-8")

    # Copy the assistant + worker units into the fixture project.
    import shutil

    repo = Path(__file__).resolve().parents[3]
    for unit in ("assistant", "critic", "research"):
        shutil.copytree(repo / "agents" / unit, demo / "agents" / unit, dirs_exist_ok=True)
    for skill in ("retrieval", "summarizer"):
        shutil.copytree(repo / "skills" / skill, demo / "skills" / skill, dirs_exist_ok=True)

    monkeypatch.setattr("lottie.benchmark.runner.build_provider", _seeded_provider)

    report = benchmark(demo, "assistant", ["mock/x"])
    assert report.agent == "assistant"
    assert report.providers and report.providers[0].cases
```

- [ ] **Step 3: Run to verify it fails, then passes**

Run: `pytest agents/assistant/tests/test_benchmark.py -v`
Expected first run: FAIL if `evals.yaml` or wiring is missing. Fix wiring (not the assertions) until PASS.
If the mock-response count is off (`MockLLMProvider responses exhausted` or leftover responses), adjust the seeded list to exactly match the 6-call budget documented in `evals.yaml`.

- [ ] **Step 4: Confirm via CLI (manual, optional)**

Run: `LOTTIE_EMBEDDING_MODEL=mock/embed LOTTIE_VECTOR_STORE=memory pytest agents/assistant/tests/test_benchmark.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/assistant/evals.yaml agents/assistant/tests/test_benchmark.py
git commit -m "test(bench): assistant mesh eval suite + hermetic benchmark run"
```

### Task 13: Full gate + docs/spec sync

**Files:**
- Modify: `CLAUDE.md` (project-structure block: add `mesh/`), `README.md` (roadmap row `v0.3.0` → ✅ / 🚧 as appropriate), `LOTTIE_PHASE0_SPEC.md` (release table `v0.3.0` note)

- [ ] **Step 1: Run the full gate**

Run: `pytest -q`
Expected: all green (existing 618 + new mesh/critic/assistant tests).
Run: `pytest --cov=lottie --cov-report=term-missing -q`
Expected: TOTAL coverage ≥ 80%.
Run: `mypy --strict src`
Expected: `Success: no issues found`.
Run: `ruff check .`
Expected: `All checks passed!`

- [ ] **Step 2: Update `CLAUDE.md`** — in the project-structure block, add under `src/lottie/`:
```
  mesh/         — MeshAgent, MeshEngine (LocalEngine), SupervisorRouter, mesh schemas
```
And add a one-line note that a mesh is a `BaseAgent` (reuses run/serve/benchmark).

- [ ] **Step 3: Update `README.md` roadmap row** — change the `v0.3.0 | 2 — Agent Mesh` status from `◻` to `✅` (core mesh shipped), and update the **Status** line to mention the agent mesh. Note in the row that parallel/HITL/time-travel are Phase 3.

- [ ] **Step 4: Update `LOTTIE_PHASE0_SPEC.md`** — in the release table (~line 420), mark `v0.3.0` Agent Mesh as delivered (core slice), noting deferred patterns.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md LOTTIE_PHASE0_SPEC.md
git commit -m "docs: close Phase 2 core agent mesh, sync roadmap + spec"
```

---

## Definition of done

- `pytest -q` green; coverage ≥ 80%; `mypy --strict src` clean; `ruff check .` clean.
- `lottie run assistant --input '{"task":"..."}'` returns an `AssistantOutput` (with mock providers, no API key).
- `lottie serve` exposes an `assistant()` MCP tool (verified by existing serve discovery — no mesh-specific change needed).
- Supervisor routing rejects undeclared workers (`CapabilityViolation`); loop bounded by `max_steps` (`MeshStepLimitExceeded`).
- Worker token/cost rolled into the mesh `RunResult`.
- No new third-party dependency added (LangGraph deferred to Phase 3).
- Tag `v0.3.0` cut on merge to `main` (per the always-ask-before-push rule, confirm before pushing).

## Deferred to Phase 3 (do NOT build here)

`LangGraphEngine` adapter + checkpointer (FU-1), parallel fork/join (FU-2), human-in-the-loop interrupt/resume (FU-3), time-travel replay (FU-4), per-node security gate (FU-5), capability enforcement on every skill call (FU-6).
