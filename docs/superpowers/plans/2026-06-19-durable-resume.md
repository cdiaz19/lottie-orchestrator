# Durable Resume over REST Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/agents/{name}/resume` backed by a durable sqlite checkpointer, so a mesh interrupt can be resumed across server restarts / workers — closing FU-9.

**Architecture:** `LangGraphEngine` resolves its checkpoint backend from `LOTTIE_MESH_CHECKPOINT` (arg > env > "memory"); `lottie serve --port` sets `sqlite`, persisting checkpoints to a shared root-derived db. Resume rehydrates by `thread_id` from that store (the agent cache stops being load-bearing). The REST route reuses `AgentService.resume_agent` (no second gate); a `get_state` pre-check raises a typed `ThreadNotFoundError` for unknown threads, mapped to a clean HTTP error contract.

**Tech Stack:** Python 3.12, Pydantic v2, langgraph + langgraph-checkpoint-sqlite (`[mesh]`), Starlette (`[api]`), pytest, `uv run` (mypy --strict, ruff).

**Design:** `docs/superpowers/specs/2026-06-19-durable-resume-design.md`

---

## File structure

- **Modify** `src/lottie/mesh/langgraph_engine.py` — `checkpoint` resolves arg>env>memory (`_resolve_checkpoint`); `resume` pre-checks `get_state` → raises `ThreadNotFoundError`.
- **Modify** `src/lottie/mesh/checkpoint.py` — remove the `# pragma: no cover` on the sqlite branch.
- **Modify** `src/lottie/mesh/errors.py` — add `ThreadNotFoundError(MeshError)`.
- **Modify** `src/lottie/serve/errors.py` — add `NotResumable(ServeError)`, `ThreadNotFound(ServeError)`.
- **Modify** `src/lottie/serve/service.py` — `resume_agent` maps the new errors, lazily converts the decision to `ApprovalDecision`, uses `_check_output`.
- **Modify** `src/lottie/serve/rest_schema.py` — add `ResumeDecision`, `ResumeRequest` (mesh-import-free).
- **Modify** `src/lottie/serve/rest_app.py` — add the resume route + handler.
- **Modify** `src/lottie/cli/serve.py` — `--port` sets `LOTTIE_MESH_CHECKPOINT=sqlite` (setdefault).
- **Modify** `CLAUDE.md` — note the resume endpoint + the env.
- **Tests:** `src/lottie/mesh/tests/test_checkpoint_resolve.py` (new), additions to `src/lottie/mesh/tests/` for the engine, `src/lottie/serve/tests/test_service.py`, `test_rest_app.py`, `src/lottie/cli/tests/test_serve.py`.

Known facts (verified):
- `LangGraphEngine.__init__(*, checkpoint: str = "memory", root: Path | None = None, interrupt_before=None)`; lazily builds `self._saver = build_checkpointer(self._checkpoint, self._root)` in `_build`. `resume(thread_id, *, nodes, route, decision: ApprovalDecision)` builds the graph, then branches on `decision.action`.
- `build_checkpointer(kind="memory", root=None)` already returns a `SqliteSaver` under `<root or cwd>/.lottie/mesh/checkpoints.db` for `kind == "sqlite"` (currently `# pragma: no cover`).
- `mesh/errors.py`: `MeshError(Exception)` base.
- `LocalEngine.resume` unconditionally raises `MeshError` (HITL needs LangGraph) — do NOT change it.
- `serve/errors.py`: `ServeError(Exception)`, `SecurityViolation`/`InputSecurityViolation`/`OutputSecurityViolation`.
- `AgentService.resume_agent(self, name, thread_id, decision)` currently: `_require_agent` → `_get_agent(name, None)` → `getattr(agent,"resume",None)`; `None` → raises `AgentExecutionError("does not support resume")`; else `resume(thread_id, decision)`, then `self._gate.check_output(...)`, then `self._result(...)`. `_check_output(agent, output)` helper already exists (runs the output gate, re-raises `OutputSecurityViolation` with metrics).
- `ApprovalDecision(BaseModel)` (`lottie.mesh.schema`): `action: Literal["approve","reject"]`, `edited_input: dict[str,str] = {}`.
- `RunResult` carries `status`/`thread_id`/`pending`; `_result` duck-types them off the mesh output.
- Test template for an inline mesh agent: `test_service.py::test_service_surfaces_interrupted_and_resumes` writes `agents/gate/{agent,schema,config}.py` building a `LangGraphEngine(interrupt_before=['deploy'])` mesh, drives it with `MockLLMProvider(["deploy","FINISH","FINISH"])` → `status="interrupted"` then resume → `complete`.
- Audit: autouse fixture sets `LOTTIE_DISABLE_AUDIT=1`; `monkeypatch.delenv` re-enables; db at `<cwd>/.lottie/audit.db`; agent audit name = class name.

---

## Task 1: Env-resolved checkpoint backend + exercise sqlite

**Files:**
- Modify: `src/lottie/mesh/langgraph_engine.py`
- Modify: `src/lottie/mesh/checkpoint.py`
- Test: `src/lottie/mesh/tests/test_checkpoint_resolve.py` (create)

- [ ] **Step 1: Write the failing tests** — create `src/lottie/mesh/tests/test_checkpoint_resolve.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")


def test_resolve_precedence_arg_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.mesh.langgraph_engine import _resolve_checkpoint

    monkeypatch.setenv("LOTTIE_MESH_CHECKPOINT", "sqlite")
    assert _resolve_checkpoint("memory") == "memory"  # explicit arg beats env


def test_resolve_env_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.mesh.langgraph_engine import _resolve_checkpoint

    monkeypatch.setenv("LOTTIE_MESH_CHECKPOINT", "sqlite")
    assert _resolve_checkpoint(None) == "sqlite"
    monkeypatch.delenv("LOTTIE_MESH_CHECKPOINT", raising=False)
    assert _resolve_checkpoint(None) == "memory"


def test_build_sqlite_checkpointer(tmp_path: Path) -> None:
    from lottie.mesh.checkpoint import build_checkpointer

    saver = build_checkpointer("sqlite", tmp_path)
    assert saver is not None
    assert (tmp_path / ".lottie" / "mesh").is_dir()
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/mesh/tests/test_checkpoint_resolve.py -q` (ImportError: `_resolve_checkpoint`).

- [ ] **Step 3: Add `_resolve_checkpoint` + env-resolve the engine** — in `src/lottie/mesh/langgraph_engine.py`, add `import os` (top) and a module-level helper:

```python
def _resolve_checkpoint(arg: str | None) -> str:
    """Checkpoint backend precedence: explicit arg > LOTTIE_MESH_CHECKPOINT env > 'memory'."""
    if arg is not None:
        return arg
    return os.getenv("LOTTIE_MESH_CHECKPOINT", "memory")
```

Change `__init__` to accept `checkpoint: str | None = None` and resolve it:

```python
    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        root: Path | None = None,
        interrupt_before: list[str] | None = None,
    ) -> None:
        if not _HAS_LANGGRAPH:
            raise MeshError("LangGraphEngine requires: pip install lottie-orchestrator[mesh]")
        self._checkpoint = _resolve_checkpoint(checkpoint)
        self._root = root
        self._interrupt_before = list(interrupt_before or [])
        self._saver: Any = None
```

- [ ] **Step 4: Remove the sqlite no-cover** — in `src/lottie/mesh/checkpoint.py`, delete the `  # pragma: no cover - not exercised in the hermetic suite` comment on the `if kind == "sqlite":` line (the branch is now tested).

- [ ] **Step 5: Run, verify PASS** — `uv run pytest src/lottie/mesh/tests/test_checkpoint_resolve.py -q` (3 tests).

- [ ] **Step 6: No regressions** — `uv run pytest src/lottie/mesh/tests/ -q` (existing mesh tests build `LangGraphEngine()` / `LangGraphEngine(interrupt_before=[...])` with no checkpoint → resolves to "memory" with the env unset; all still pass). Then `uv run mypy --strict src/lottie/mesh` + `uv run ruff check src/lottie/mesh` clean.

- [ ] **Step 7: Commit**

```bash
git add src/lottie/mesh/langgraph_engine.py src/lottie/mesh/checkpoint.py src/lottie/mesh/tests/test_checkpoint_resolve.py
git commit -m "feat(mesh): resolve checkpoint backend from LOTTIE_MESH_CHECKPOINT (arg>env>memory)"
```

---

## Task 2: `ThreadNotFoundError` — engine pre-check for unknown threads

**Files:**
- Modify: `src/lottie/mesh/errors.py`
- Modify: `src/lottie/mesh/langgraph_engine.py`
- Test: `src/lottie/mesh/tests/test_resume_unknown_thread.py` (create)

- [ ] **Step 1: Write the failing test** — create `src/lottie/mesh/tests/test_resume_unknown_thread.py`:

```python
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
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/mesh/tests/test_resume_unknown_thread.py -q` (ImportError `ThreadNotFoundError`, or a raw langgraph/pydantic error instead of the typed one).

- [ ] **Step 3: Add the mesh error** — append to `src/lottie/mesh/errors.py`:

```python
class ThreadNotFoundError(MeshError):
    """No checkpoint exists for the given thread_id (never existed, or pruned)."""
```

- [ ] **Step 4: Pre-check in `LangGraphEngine.resume`** — in `src/lottie/mesh/langgraph_engine.py`, import the error (add to the existing `from lottie.mesh.errors import MeshError` line → `from lottie.mesh.errors import MeshError, ThreadNotFoundError`) and add the pre-check at the top of `resume`, before the `decision.action` branch:

```python
    def resume(self, thread_id, *, nodes, route, decision):
        graph = self._build(nodes, route)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        # A bogus/expired thread has no persisted state; get_state returns an empty snapshot.
        # Detect it deterministically here so neither langgraph's EmptyInputError (approve) nor
        # our own MeshState validation error (reject) leaks — wrap as the typed ThreadNotFoundError.
        snap = graph.get_state(config)
        if not snap.values.get("task"):
            raise ThreadNotFoundError(f"no checkpoint for thread {thread_id!r}")
        if decision.action == "reject":
            # (existing reject body unchanged — it can reuse `snap`)
            worker = snap.next[0] if snap.next else "unknown"
            base = len(MeshState.model_validate(snap.values).history)
            graph.update_state(
                config,
                {"history": [StepResult(worker=worker, result="rejected by human", step=base)]},
                as_node=worker,
            )
        graph.invoke(None, config)
        return self._snapshot(graph, config, thread_id)
```

(Keep the existing `reject`/`approve` logic exactly; the only change is the `snap = get_state` pre-check + the `ThreadNotFoundError` raise. Reuse `snap` in the reject branch instead of re-fetching, if it previously re-fetched.)

- [ ] **Step 5: Run, verify PASS** — `uv run pytest src/lottie/mesh/tests/test_resume_unknown_thread.py -q` (2 params pass) and `uv run pytest src/lottie/mesh/tests/ -q` (the existing happy-path resume tests still pass — a REAL thread has a `task` in its snapshot, so the pre-check doesn't trip).

- [ ] **Step 6: Gates** — `uv run mypy --strict src/lottie/mesh`, `uv run ruff check src/lottie/mesh` clean.

- [ ] **Step 7: Commit**

```bash
git add src/lottie/mesh/errors.py src/lottie/mesh/langgraph_engine.py src/lottie/mesh/tests/test_resume_unknown_thread.py
git commit -m "feat(mesh): raise typed ThreadNotFoundError on resume of an unknown thread"
```

---

## Task 3: Serve error leaves + `resume_agent` rework

**Files:**
- Modify: `src/lottie/serve/errors.py`
- Modify: `src/lottie/serve/service.py`
- Test: `src/lottie/serve/tests/test_service.py`

- [ ] **Step 1: Write/Update the failing tests** — in `src/lottie/serve/tests/test_service.py`:

(a) The existing `test_resume_agent_on_non_mesh_raises_execution_error` asserts a non-mesh resume raises `AgentExecutionError`. Change it to expect the new `NotResumable`:

```python
def test_resume_agent_on_non_mesh_raises_not_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain agent has no HITL resume — resume_agent must reject it as NotResumable."""
    from lottie.serve.errors import NotResumable

    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    svc = AgentService(demo)

    class _D:
        action = "approve"
        edited_input: dict[str, str] = {}

    with pytest.raises(NotResumable):
        svc.resume_agent("echo", "t1", _D())
```

(Delete the old `test_resume_agent_on_non_mesh_raises_execution_error`; its `ApprovalDecision` import there is no longer needed.)

(b) Append a unit test that an unknown thread on a real mesh maps to `ThreadNotFound`:

```python
def test_resume_agent_unknown_thread_maps_to_thread_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        pytest.skip("needs [mesh] extra")
    from lottie.serve.errors import ThreadNotFound

    demo = _gate_mesh_project(tmp_path, monkeypatch, checkpoint="sqlite")  # see helper below
    svc = AgentService(demo)
    svc.run_agent("gate", {"task": "ship"})  # creates a checkpoint under a real thread

    class _D:
        action = "approve"
        edited_input: dict[str, str] = {}

    with pytest.raises(ThreadNotFound):
        svc.resume_agent("gate", "no-such-thread", _D())
```

Add a `_gate_mesh_project` helper near the top of the file (factor it from the inline mesh setup in `test_service_surfaces_interrupted_and_resumes`), parameterized by checkpoint kind so the engine persists to sqlite:

```python
def _gate_mesh_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, checkpoint: str = "memory"
) -> Path:
    """Scaffold a project with a `gate` mesh agent whose engine uses the given checkpoint."""
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    gate = demo / "agents" / "gate"
    gate.mkdir(parents=True)
    (gate / "__init__.py").write_text("", encoding="utf-8")
    (gate / "schema.py").write_text(
        "from __future__ import annotations\n"
        "from lottie.mesh.schema import MeshInput, MeshOutput\n"
        "class GateInput(MeshInput):\n    pass\n"
        "class GateOutput(MeshOutput):\n    pass\n",
        encoding="utf-8",
    )
    (gate / "agent.py").write_text(
        "from __future__ import annotations\n"
        "from pathlib import Path\n"
        "from lottie.llm import LLMProvider\n"
        "from lottie.mesh import MeshAgent, MeshState, StepResult\n"
        "from lottie.mesh.langgraph_engine import LangGraphEngine\n"
        "from lottie.project.config import AgentConfig\n"
        "def _deploy(state: MeshState) -> MeshState:\n"
        "    return state.with_step(StepResult(worker='deploy', result='shipped'))\n"
        "class GateMesh(MeshAgent):\n"
        "    @classmethod\n"
        "    def from_project(cls, *, llm: LLMProvider, root: Path, config: AgentConfig,"
        " enable_benchmarks=None):\n"
        f"        engine = LangGraphEngine(checkpoint={checkpoint!r}, root=root,"
        " interrupt_before=['deploy'])\n"
        "        return cls(llm, nodes={'deploy': _deploy},"
        " descriptions={'deploy': 'ships'}, engine=engine,"
        " enable_benchmarks=enable_benchmarks)\n",
        encoding="utf-8",
    )
    (gate / "config.yaml").write_text(
        "provider: mock/x\nworkers: [deploy]\ninterrupt_before: [deploy]\npolicies: [base]\n",
        encoding="utf-8",
    )
    (gate / "AGENT.md").write_text("# gate mesh\n", encoding="utf-8")
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider(["deploy", "FINISH", "FINISH"]),
    )
    return demo
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/serve/tests/test_service.py -q -k "resume"` (ImportError `NotResumable`/`ThreadNotFound`, and the non-mesh test fails because resume still raises `AgentExecutionError`).

- [ ] **Step 3: Add the serve error leaves** — append to `src/lottie/serve/errors.py`:

```python
class NotResumable(ServeError):
    """The agent exists but cannot be resumed (not a mesh / no HITL)."""


class ThreadNotFound(ServeError):
    """No checkpoint exists for the given thread_id (never existed or pruned)."""
```

- [ ] **Step 4: Rework `resume_agent`** — in `src/lottie/serve/service.py`, add a structural Protocol for the decision (read-only properties dodge mypy invariance), update the imports, and rewrite `resume_agent`:

```python
from typing import Protocol


class _ResumeDecision(Protocol):
    @property
    def action(self) -> str: ...
    @property
    def edited_input(self) -> dict[str, str]: ...
```

Add to the service imports: `from lottie.serve.errors import NotResumable, ThreadNotFound` (alongside the existing `OutputSecurityViolation` import), and `from lottie.mesh.errors import ThreadNotFoundError` — NO, that would import mesh at module top and break the base install. Instead import `ThreadNotFoundError` lazily inside `resume_agent`. Rewrite:

```python
    def resume_agent(
        self, name: str, thread_id: str, decision: _ResumeDecision
    ) -> RunResult:
        """Resume an interrupted mesh agent from its checkpoint.

        Durable when the engine uses the sqlite checkpointer (the `lottie serve --port`
        default, via LOTTIE_MESH_CHECKPOINT=sqlite): a fresh process rebuilds the agent by
        name and rehydrates the thread from the shared db. With the in-memory checkpointer it
        is process-local (only the process that ran the interrupt can resume).
        """
        from lottie.mesh.errors import ThreadNotFoundError  # lazy: keep serve base-install mesh-free
        from lottie.mesh.schema import ApprovalDecision

        self._require_agent(name)
        agent = self._get_agent(name, None)
        resume = getattr(agent, "resume", None)
        if resume is None:
            raise NotResumable(f"agent '{name}' is not resumable (not a mesh)")
        approval = ApprovalDecision(
            action=decision.action, edited_input=dict(decision.edited_input)
        )
        try:
            output = resume(thread_id, approval)
        except ThreadNotFoundError as exc:
            raise ThreadNotFound(f"thread '{thread_id}' not found") from exc
        except Exception as exc:  # noqa: BLE001 — any other failure → one typed error
            raise AgentExecutionError(f"agent '{name}' failed: {exc}") from exc
        self._check_output(agent, output)
        return self._result(name, output, agent.last_metrics)
```

(mypy note: `ApprovalDecision(action=decision.action, ...)` — `action` is typed `str` by the Protocol but `ApprovalDecision.action` is `Literal["approve","reject"]`; pydantic validates at runtime, and mypy accepts the `str` since pydantic's generated `__init__` accepts the literal's supertype. If mypy complains, narrow with `cast`-free pydantic `model_validate({"action": decision.action, "edited_input": dict(decision.edited_input)})`.)

- [ ] **Step 5: Run, verify PASS** — `uv run pytest src/lottie/serve/tests/test_service.py -q` (the reworked non-mesh test, the new unknown-thread test, and the existing `test_service_surfaces_interrupted_and_resumes` — which passes an `ApprovalDecision` that structurally satisfies `_ResumeDecision` — all pass).

- [ ] **Step 6: Gates** — `uv run mypy --strict src/lottie/serve`, `uv run ruff check src/lottie/serve` clean; `uv run python -c "import lottie.serve"` works WITHOUT langgraph forced (the mesh imports are lazy): `uv run python -c "import lottie.serve, sys; print('langgraph' in sys.modules)"` → `False`.

- [ ] **Step 7: Commit**

```bash
git add src/lottie/serve/errors.py src/lottie/serve/service.py src/lottie/serve/tests/test_service.py
git commit -m "feat(serve): resume_agent maps NotResumable/ThreadNotFound; lazy mesh decision convert"
```

---

## Task 4: Durable cross-process resume (test)

**Files:**
- Test: `src/lottie/serve/tests/test_service.py`

- [ ] **Step 1: Write the test** — append to `src/lottie/serve/tests/test_service.py`:

```python
def test_resume_across_fresh_service_with_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable resume: a FRESH AgentService (new process simulated) resumes a checkpoint
    written by another, via the shared sqlite db — proving rehydrate-by-thread_id."""
    try:
        import langgraph  # noqa: F401
    except ImportError:
        pytest.skip("needs [mesh] extra")
    from lottie.mesh.schema import ApprovalDecision

    demo = _gate_mesh_project(tmp_path, monkeypatch, checkpoint="sqlite")

    svc1 = AgentService(demo)
    started = svc1.run_agent("gate", {"task": "ship"})
    assert started.status == "interrupted"
    assert started.thread_id

    # A brand-new AgentService (empty agent cache) — only the shared sqlite db links them.
    # The DURABILITY guarantee is that svc2 FINDS the checkpoint written by svc1 (no
    # ThreadNotFound) and produces a real RunResult — i.e. rehydrate-by-thread_id works with
    # zero shared in-memory state. The exact terminal status depends on the mock supervisor
    # script (a fresh MockLLMProvider replays from the top, so a multi-gate re-interrupt is
    # legitimate); assert the result is well-formed, not a specific terminal status.
    svc2 = AgentService(demo)
    resumed = svc2.resume_agent("gate", started.thread_id, ApprovalDecision(action="approve"))
    assert resumed.agent == "gate"
    assert resumed.status in {"complete", "interrupted"}
```

- [ ] **Step 2: Run the test** — `uv run pytest src/lottie/serve/tests/test_service.py::test_resume_across_fresh_service_with_sqlite -q`. Expected PASS — the fresh service resumes without `ThreadNotFound` (the checkpoint is on disk).

- [ ] **Step 3: Confirm the durability signal** — if `resume_agent` raises `ThreadNotFound`, the sqlite wiring is broken (the fresh service didn't see the checkpoint) — that's a real failure to fix, not a reconcile. Verify `<demo>/.lottie/mesh/checkpoints.db` exists after the run. Do NOT weaken the test to swallow a `ThreadNotFound`.

- [ ] **Step 4: Full mesh+serve green** — `uv run pytest src/lottie/serve/tests/test_service.py src/lottie/mesh/tests/ -q`.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/serve/tests/test_service.py
git commit -m "test(serve): durable cross-process resume via shared sqlite checkpointer"
```

---

## Task 5: `ResumeRequest` schema + `POST /v1/agents/{name}/resume` route

**Files:**
- Modify: `src/lottie/serve/rest_schema.py`
- Modify: `src/lottie/serve/rest_app.py`
- Test: `src/lottie/serve/tests/test_rest_app.py`

- [ ] **Step 1: Write the failing tests** — append to `src/lottie/serve/tests/test_rest_app.py` (it already has `_project`, `_mock_provider`, `TestClient`, `runner`). Add a mesh-project helper mirroring Task 3's, then the route tests:

```python
def _gate_mesh_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project with a `gate` mesh agent (sqlite checkpointer) for resume tests."""
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    gate = demo / "agents" / "gate"
    gate.mkdir(parents=True)
    (gate / "__init__.py").write_text("", encoding="utf-8")
    (gate / "schema.py").write_text(
        "from __future__ import annotations\n"
        "from lottie.mesh.schema import MeshInput, MeshOutput\n"
        "class GateInput(MeshInput):\n    pass\n"
        "class GateOutput(MeshOutput):\n    pass\n",
        encoding="utf-8",
    )
    (gate / "agent.py").write_text(
        "from __future__ import annotations\n"
        "from pathlib import Path\n"
        "from lottie.llm import LLMProvider\n"
        "from lottie.mesh import MeshAgent, MeshState, StepResult\n"
        "from lottie.mesh.langgraph_engine import LangGraphEngine\n"
        "from lottie.project.config import AgentConfig\n"
        "def _deploy(state: MeshState) -> MeshState:\n"
        "    return state.with_step(StepResult(worker='deploy', result='shipped'))\n"
        "class GateMesh(MeshAgent):\n"
        "    @classmethod\n"
        "    def from_project(cls, *, llm: LLMProvider, root: Path, config: AgentConfig,"
        " enable_benchmarks=None):\n"
        "        engine = LangGraphEngine(checkpoint='sqlite', root=root,"
        " interrupt_before=['deploy'])\n"
        "        return cls(llm, nodes={'deploy': _deploy},"
        " descriptions={'deploy': 'ships'}, engine=engine,"
        " enable_benchmarks=enable_benchmarks)\n",
        encoding="utf-8",
    )
    (gate / "config.yaml").write_text(
        "provider: mock/x\nworkers: [deploy]\ninterrupt_before: [deploy]\npolicies: [base]\n",
        encoding="utf-8",
    )
    (gate / "AGENT.md").write_text("# gate mesh\n", encoding="utf-8")
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: __import__("lottie.llm", fromlist=["MockLLMProvider"]).MockLLMProvider(
            ["deploy", "FINISH", "FINISH"]
        ),
    )
    return demo


def _needs_mesh() -> None:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        pytest.skip("needs [mesh] extra")


def test_resume_route_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_mesh()
    from lottie.serve.rest_app import build_rest_app

    demo = _gate_mesh_project(tmp_path, monkeypatch)
    client = TestClient(build_rest_app(demo))
    started = client.post("/v1/agents/gate/run", json={"task": "ship"})
    assert started.status_code == 200
    tid = started.json()["thread_id"]
    assert started.json()["status"] == "interrupted" and tid

    resumed = client.post(
        "/v1/agents/gate/resume",
        json={"thread_id": tid, "decision": {"action": "approve"}},
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] in {"complete", "interrupted"}


def test_resume_unknown_agent_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.serve.rest_app import build_rest_app

    demo = _project(tmp_path, monkeypatch)
    client = TestClient(build_rest_app(demo))
    resp = client.post(
        "/v1/agents/nope/resume",
        json={"thread_id": "t", "decision": {"action": "approve"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found"


def test_resume_non_mesh_400_not_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lottie.serve.rest_app import build_rest_app

    demo = _project(tmp_path, monkeypatch)  # echo is not a mesh
    _mock_provider(monkeypatch)
    client = TestClient(build_rest_app(demo))
    resp = client.post(
        "/v1/agents/echo/resume",
        json={"thread_id": "t", "decision": {"action": "approve"}},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "not_resumable"


def test_resume_unknown_thread_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_mesh()
    from lottie.serve.rest_app import build_rest_app

    demo = _gate_mesh_project(tmp_path, monkeypatch)
    client = TestClient(build_rest_app(demo))
    client.post("/v1/agents/gate/run", json={"task": "ship"})  # create a real checkpoint
    resp = client.post(
        "/v1/agents/gate/resume",
        json={"thread_id": "no-such-thread", "decision": {"action": "approve"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "thread_not_found"


def test_resume_bad_body_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.serve.rest_app import build_rest_app

    demo = _project(tmp_path, monkeypatch)
    client = TestClient(build_rest_app(demo))
    resp = client.post("/v1/agents/echo/resume", json={"decision": {"action": "approve"}})  # no thread_id
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request"
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/serve/tests/test_rest_app.py -q -k resume` (no `/resume` route → 404/405; or ImportError on `ResumeRequest`).

- [ ] **Step 3: Add the request models** — append to `src/lottie/serve/rest_schema.py` (NO mesh import — pure pydantic):

```python
from typing import Literal

from pydantic import BaseModel


class ResumeDecision(BaseModel):
    """A human HITL decision, mesh-import-free (converted to ApprovalDecision in the service)."""

    action: Literal["approve", "reject"]
    edited_input: dict[str, str] = {}


class ResumeRequest(BaseModel):
    thread_id: str
    decision: ResumeDecision
```

(`from __future__ import annotations` is already at the top; add the `Literal`/`BaseModel` imports if not present — `BaseModel` is already imported via `from lottie.serve.schema import ...`? No: `rest_schema` imports only `AgentInfo, RunResult`. Add `from pydantic import BaseModel` and `from typing import Literal`.)

- [ ] **Step 4: Add the resume route + handler** — in `src/lottie/serve/rest_app.py`, extend the imports:

```python
from lottie.serve.errors import (
    InputSecurityViolation,
    NotResumable,
    OutputSecurityViolation,
    ThreadNotFound,
)
from lottie.serve.rest_schema import (
    ResumeRequest,
    agent_detail_dict,
    agent_list_dict,
    run_result_dict,
    withheld_dict,
)
```

Add the handler inside `rest_routes` (after `run_agent_route`):

```python
    async def resume_agent_route(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        try:
            body = await request.json()
            req = ResumeRequest.model_validate(body)
        except (ValueError, ValidationError):
            return json_error(400, "invalid request body", type_="invalid_request")
        try:
            result = await anyio.to_thread.run_sync(
                lambda: svc.resume_agent(name, req.thread_id, req.decision)
            )
        except InputSecurityViolation:
            return json_error(400, "request blocked by content policy", type_="content_filter")
        except OutputSecurityViolation as exc:
            return JSONResponse(
                withheld_dict(name, input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
            )
        except NotResumable:
            return json_error(400, f"agent '{name}' is not resumable", type_="not_resumable")
        except ThreadNotFound:
            return json_error(404, "thread not found", type_="thread_not_found")
        except AgentNotFoundError:
            return json_error(404, f"agent '{name}' not found", type_="not_found")
        except (AgentLoadError, AgentExecutionError):
            return json_error(500, "internal error", type_="internal_error")
        return JSONResponse(run_result_dict(result))
```

Add `ValidationError` to the imports if not already there (`from pydantic import ValidationError`). Register the route:

```python
    return [
        Route("/v1/agents", list_agents, methods=["GET"]),
        Route("/v1/agents/{name}", agent_detail, methods=["GET"]),
        Route("/v1/agents/{name}/run", run_agent_route, methods=["POST"]),
        Route("/v1/agents/{name}/resume", resume_agent_route, methods=["POST"]),
    ]
```

Note: `ResumeRequest`/`rest_schema` and `rest_app` must NOT import `lottie.mesh.*` — the `ResumeDecision` → `ApprovalDecision` conversion happens lazily inside `service.resume_agent` (Task 3). Confirm with a grep in Step 6.

- [ ] **Step 5: Run, verify PASS** — `uv run pytest src/lottie/serve/tests/test_rest_app.py -q` (all prior REST tests + the 6 resume tests).

- [ ] **Step 6: Gates** — `uv run mypy --strict src/lottie/serve`, `uv run ruff check src/lottie/serve` clean; confirm no mesh import in the REST modules: `grep -r "lottie.mesh" src/lottie/serve/rest_app.py src/lottie/serve/rest_schema.py` → no matches.

- [ ] **Step 7: Commit**

```bash
git add src/lottie/serve/rest_schema.py src/lottie/serve/rest_app.py src/lottie/serve/tests/test_rest_app.py
git commit -m "feat(serve): REST POST /v1/agents/{name}/resume (typed errors; mesh-free schema)"
```

---

## Task 6: CLI sets the durable checkpoint env + CLAUDE.md

**Files:**
- Modify: `src/lottie/cli/serve.py`
- Modify: `CLAUDE.md`
- Test: `src/lottie/cli/tests/test_serve.py`

- [ ] **Step 1: Write the failing test** — append to `src/lottie/cli/tests/test_serve.py`:

```python
def test_serve_port_sets_mesh_checkpoint_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lottie serve --port` opts served meshes into durable sqlite checkpoints."""
    pytest.importorskip("starlette")
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "demo"])
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    monkeypatch.delenv("LOTTIE_MESH_CHECKPOINT", raising=False)

    seen: dict[str, str | None] = {}

    def fake_run(application: object, **kwargs: object) -> None:
        seen["checkpoint"] = __import__("os").environ.get("LOTTIE_MESH_CHECKPOINT")

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert runner.invoke(app, ["serve", "--port", "8125"]).exit_code == 0
    assert seen["checkpoint"] == "sqlite"
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/cli/tests/test_serve.py::test_serve_port_sets_mesh_checkpoint_env -q` (env not set → None).

- [ ] **Step 3: Set the env in the `--port` branch** — in `src/lottie/cli/serve.py`, add `import os` at the top, and in the `--port` branch set the env once before serving (after the lazy import, before `uvicorn.run`):

```python
    # Served meshes persist their checkpoints so an interrupt can be resumed across
    # restarts/workers. setdefault so an operator can override to "memory". Process-global
    # mutation — acceptable for a server process; set once at startup before any agent runs.
    os.environ.setdefault("LOTTIE_MESH_CHECKPOINT", "sqlite")
    uvicorn.run(build_http_app(root), host=host, port=port)
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/cli/tests/test_serve.py -q` (all serve CLI tests).

- [ ] **Step 5: Update CLAUDE.md** — under the `lottie serve` CLI section, add a line after the `serve --port` line:

```
#   resume an interrupted mesh: POST /v1/agents/{name}/resume {thread_id, decision}
#   durable across restarts when served (LOTTIE_MESH_CHECKPOINT=sqlite, set by serve --port)
```

(Match the surrounding comment style; place it with the other `serve --port` HTTP-API comment lines.)

- [ ] **Step 6: Gates** — `uv run mypy --strict src/lottie`, `uv run ruff check src/lottie` clean.

- [ ] **Step 7: Commit**

```bash
git add src/lottie/cli/serve.py src/lottie/cli/tests/test_serve.py CLAUDE.md
git commit -m "feat(cli): serve --port enables durable mesh checkpoints (LOTTIE_MESH_CHECKPOINT=sqlite)"
```

---

## Task 7: Governance on resume + base-install safety

**Files:**
- Test: `src/lottie/serve/tests/test_rest_app.py`

- [ ] **Step 1: Write the tests** — append to `src/lottie/serve/tests/test_rest_app.py`:

```python
def test_resume_writes_audit_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume run is audited (governance inherited on the resume path)."""
    _needs_mesh()
    from lottie.governance.audit import SqliteAuditLogger
    from lottie.serve.rest_app import build_rest_app

    demo = _gate_mesh_project(tmp_path, monkeypatch)
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
    client = TestClient(build_rest_app(demo))
    started = client.post("/v1/agents/gate/run", json={"task": "ship"})
    tid = started.json()["thread_id"]
    client.post(
        "/v1/agents/gate/resume",
        json={"thread_id": tid, "decision": {"action": "approve"}},
    )
    # GateMesh's class name is the audit key; at least one record exists for the run+resume.
    records = SqliteAuditLogger(demo).query(agent="GateMesh")
    assert len(records) >= 1
    assert any(r.status == "ok" for r in records)


def test_rest_modules_are_mesh_import_free() -> None:
    """rest_app/rest_schema must not import lottie.mesh at source level (keeps [api]-without-[mesh]
    importable; the ResumeDecision->ApprovalDecision convert is lazy in service.py)."""
    import pathlib

    import lottie.serve.rest_app as ra
    import lottie.serve.rest_schema as rs

    for mod in (ra, rs):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert "lottie.mesh" not in src, f"{mod.__name__} imports lottie.mesh"
```

- [ ] **Step 2: Run the tests** — `uv run pytest src/lottie/serve/tests/test_rest_app.py -q -k "audit or mesh_import_free"`.

- [ ] **Step 3: Reconcile if needed** — the audit test asserts inherited behavior. If the audit agent name differs from `GateMesh` (mesh agents may set `name` differently — `MeshAgent.__init__` takes `name="m"` in some constructions, but `GateMesh.from_project` here does NOT pass `name`, so it defaults to the class name `GateMesh`), confirm by reading the record's `.agent` and adjust the query string to the observed name with a one-line comment. Do NOT add prod code. The mesh-import-free test must pass as-is (it's a guard on this slice's own files).

- [ ] **Step 4: Base-install probe** — `uv run python -c "import lottie.serve, sys; print('starlette' in sys.modules, 'langgraph' in sys.modules)"` → `False False` (serve/__init__ pulls in neither; the mesh import in service.resume_agent is lazy).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/serve/tests/test_rest_app.py
git commit -m "test(serve): resume inherits audit; REST modules stay mesh-import-free"
```

---

## Task 8: Closeout — full gate

**Files:** none (verification only)

- [ ] **Step 1: Full test suite** — `uv run pytest -q`. Expected: PASS — all prior tests plus the new mesh/serve/cli ones; the OpenAI + REST suites and `build_openai_app`/`build_http_app` APIs unaffected.

- [ ] **Step 2: Types** — `uv run mypy --strict src`. Expected: clean. Fix inline (the `_ResumeDecision` Protocol uses read-only properties so `ApprovalDecision`/`ResumeDecision` both satisfy it; if `ApprovalDecision(action=decision.action)` trips Literal-vs-str, switch to `ApprovalDecision.model_validate({"action": decision.action, "edited_input": dict(decision.edited_input)})`).

- [ ] **Step 3: Lint** — `uv run ruff check`. Expected: clean.

- [ ] **Step 4: Manual smoke (optional, `[api,mesh]` installed)** — in a project with a mesh agent: `lottie serve --port 8000 &`; `curl -s -X POST localhost:8000/v1/agents/<mesh>/run -d '{"task":"x"}'` → `status:interrupted`, grab `thread_id`; `curl -s -X POST localhost:8000/v1/agents/<mesh>/resume -d '{"thread_id":"...","decision":{"action":"approve"}}'`. Confirm `.lottie/mesh/checkpoints.db` exists. Kill the server.

- [ ] **Step 5: Final commit (if any closeout fixes)**

```bash
git add -A
git commit -m "chore(serve): closeout fixes for durable resume (mypy/ruff)"
```

---

## Notes for the implementer

- **The cache is not load-bearing for durable resume.** `resume_agent` rebuilds the agent by name; durability comes from the engine's sqlite saver pointing at the shared root-derived db. The cross-process test (Task 4) is the proof — don't "optimize" it by reusing one `AgentService`.
- **Never leak a raw langgraph/pydantic exception** for an unknown thread — the engine's `get_state` pre-check raises the typed `ThreadNotFoundError`, mapped to `ThreadNotFound` (404). (FG-1 discipline.)
- **`rest_app`/`rest_schema` must never import `lottie.mesh`.** The decision conversion is lazy in `service.py`. Task 7 guards this.
- **Privacy:** error messages echo only the agent name / a generic label; never the decision or payload.
- **No second gate:** resume reuses `AgentService.resume_agent` → `_check_output` + `BaseAgent.run`. Do not add gating in `rest_app.py`.
- **Deferred (do not build):** streaming; applying `edited_input` on approve (engine simplification — accepted in the body, not applied); distributed multi-host resume. Spec §1.
```
