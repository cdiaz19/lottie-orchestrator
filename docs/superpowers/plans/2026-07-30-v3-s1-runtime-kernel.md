# V3 S1 — Runtime Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `src/lottie/runtime/` — an abort-capable middleware chain plus a fail-open event stream — with nothing consuming it yet, so V3 S2 can swap `BaseAgent.run` onto it without any behavior change.

**Architecture:** Two primitives. `Middleware` are ordered onion wrappers that receive `(ctx, nxt)`; calling `nxt` runs the rest of the chain, not calling it aborts the run, and a `finally` around `nxt` guarantees cleanup. `EventBus` is a fail-open observation stream the `Pipeline` emits from its **innermost** frame. `ModuleRegistry` composes factories into a chain and rejects order conflicts at registration. The kernel is standalone: it imports nothing from `lottie.core`, `lottie.governance`, `lottie.memory`, `lottie.security`, or `lottie.llm`.

**Tech Stack:** Python 3.12+, pydantic v2, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-30-v3-runtime-kernel-design.md` §4 and §7 (slice S1).

## Global Constraints

- **Zero new runtime dependencies.** Kernel uses stdlib + pydantic only.
- **Zero behavior change.** Nothing outside `src/lottie/runtime/` is modified. No existing test changes. `BaseAgent` is not touched — that is S2.
- **The kernel must not import `lottie.core`, `lottie.governance`, `lottie.memory`, `lottie.security`, or `lottie.llm`.** `src/lottie/core/__init__.py:1-5` eagerly imports `base_agent`; once S2 makes `BaseAgent` import the kernel, any kernel→core import becomes a circular import at package-init time. Task 1 ships a test that enforces this. Tests may import anything — they are not in the package import path.
- **`mypy --strict` clean, no `Any`.** The spec anticipated one `Any` seam at `Next`; this plan avoids it — every runnable output is a pydantic model (rule 2), so `BaseModel` is the true bound and one `cast` at the `Pipeline` boundary suffices.
- **Line length 100.** Ruff `select = ["E", "F", "I", "UP", "B", "SIM"]`.
- **`runtime/__init__.py` stays empty**, matching `governance/__init__.py`. Consumers import from submodules (`from lottie.runtime.pipeline import Pipeline`). This is the project's cycle-avoidance convention.
- **Local gate before push (rule 7b):** `uv sync --dev --all-extras`, then `uv run ruff check .`, `uv run mypy --strict src`, `uv run pytest -q`. Run from the project directory.
- **Branch before editing.** `git checkout -b feat/v3-s1-runtime-kernel` off `main`.
- **Do not start this slice until `v2.0.0` is tagged** (spec §1).

## File Structure

| File | Responsibility |
|---|---|
| `src/lottie/runtime/__init__.py` | Empty. Package marker only. |
| `src/lottie/runtime/context.py` | `ExecutionContext`, `RunKind`, `UsageAccumulator` Protocol. The per-run carrier. |
| `src/lottie/runtime/events.py` | `RunEvent` hierarchy, `Subscriber` Protocol, `EventBus`. The Event Runtime. |
| `src/lottie/runtime/middleware.py` | `Next`, `Middleware` Protocol, `Order` constants. The Lifecycle Hooks contract. |
| `src/lottie/runtime/pipeline.py` | `Pipeline` — compiles and executes the onion, emits lifecycle events. |
| `src/lottie/runtime/registry.py` | `Deps`, `Mountable`, `ModuleFactory` Protocol, `ModuleRegistry`. |
| `src/lottie/runtime/ARCHITECTURE.md` | Module doc: responsibilities, interfaces, extension points, lifecycle. |
| `src/lottie/runtime/tests/__init__.py` | Test package marker (matches `governance/tests/`). |
| `src/lottie/runtime/tests/test_imports.py` | Enforces the no-subsystem-imports constraint via AST. |
| `src/lottie/runtime/tests/test_context.py` | `ExecutionContext` behavior + parity pins against `core.metrics`. |
| `src/lottie/runtime/tests/test_events.py` | Bus isolation, ordering, and the hash-only contract. |
| `src/lottie/runtime/tests/test_pipeline.py` | Chain ordering, abort, `finally`, exception propagation. |
| `src/lottie/runtime/tests/test_pipeline_events.py` | Innermost-frame emission and `RunBlocked`. |
| `src/lottie/runtime/tests/test_registry.py` | Registration, conflict detection, build. |
| `src/lottie/runtime/tests/test_perf.py` | Per-run dispatch overhead budget. |

---

### Task 1: Package skeleton, `ExecutionContext`, and the import-hygiene guard

The import guard comes first because it is the constraint every later task must not violate.

**Files:**
- Create: `src/lottie/runtime/__init__.py`
- Create: `src/lottie/runtime/context.py`
- Create: `src/lottie/runtime/tests/__init__.py`
- Test: `src/lottie/runtime/tests/test_imports.py`
- Test: `src/lottie/runtime/tests/test_context.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunKind` (`Literal["agent", "skill"]`), `UsageAccumulator` (Protocol with `input_tokens: int`, `output_tokens: int`, `cost_usd: float`, `turns: int`), `ExecutionContext` (dataclass: `runnable: str`, `kind: RunKind`, `input: BaseModel`, `run_id: str`, `usage: UsageAccumulator`, `state: dict[str, object]`; method `scoped(module: str) -> dict[str, object]`).

- [ ] **Step 1: Create the empty package markers**

```bash
mkdir -p src/lottie/runtime/tests
touch src/lottie/runtime/__init__.py src/lottie/runtime/tests/__init__.py
```

- [ ] **Step 2: Write the failing import-hygiene test**

Create `src/lottie/runtime/tests/test_imports.py`:

```python
"""The kernel must not import the subsystems that mount onto it.

`src/lottie/core/__init__.py` eagerly imports `base_agent`. Once S2 makes `BaseAgent`
import the kernel, a kernel -> core import becomes a circular import at package-init
time. This test is the structural guarantee behind the dependency reversal in V3 spec
section 5.1 — it fails loudly the moment someone reintroduces the coupling.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = ("lottie.core", "lottie.governance", "lottie.memory", "lottie.security", "lottie.llm")

RUNTIME_DIR = Path(__file__).resolve().parent.parent


def _kernel_modules() -> list[Path]:
    """Every .py file in the kernel package, excluding its own tests."""
    return [p for p in RUNTIME_DIR.rglob("*.py") if "tests" not in p.parts]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


class TestKernelImportHygiene:
    def test_finds_the_kernel_modules(self) -> None:
        # Guard against the glob silently matching nothing and the suite passing vacuously.
        names = {p.name for p in _kernel_modules()}
        assert "context.py" in names

    def test_no_kernel_module_imports_a_subsystem(self) -> None:
        offenders: list[str] = []
        for path in _kernel_modules():
            for imported in _imported_modules(path):
                if any(imported == f or imported.startswith(f + ".") for f in FORBIDDEN):
                    offenders.append(f"{path.name} imports {imported}")
        assert offenders == []
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest src/lottie/runtime/tests/test_imports.py -v`
Expected: FAIL on `test_finds_the_kernel_modules` with `AssertionError` — `context.py` does not exist yet.

- [ ] **Step 4: Write the failing `ExecutionContext` test**

Create `src/lottie/runtime/tests/test_context.py`:

```python
"""Unit tests for the per-run carrier threaded through the middleware chain."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import BaseModel

from lottie.core.metrics import Kind, RunContext
from lottie.runtime.context import ExecutionContext, RunKind, UsageAccumulator


class _Input(BaseModel):
    text: str


def _ctx() -> ExecutionContext:
    return ExecutionContext(runnable="Demo", kind="agent", input=_Input(text="hi"), run_id="r1")


class TestRunKindParity:
    def test_run_kind_mirrors_core_metrics_kind(self) -> None:
        # RunKind is duplicated rather than imported to keep the kernel free of a
        # `lottie.core` dependency. This pins the duplication so it cannot drift.
        assert set(get_args(RunKind)) == set(get_args(Kind))


class TestUsageAccumulator:
    def test_core_run_context_satisfies_the_protocol(self) -> None:
        # The kernel depends on this structural view, never on the concrete dataclass.
        assert isinstance(RunContext(), UsageAccumulator)

    def test_default_usage_starts_at_zero(self) -> None:
        ctx = _ctx()
        assert ctx.usage.input_tokens == 0
        assert ctx.usage.output_tokens == 0
        assert ctx.usage.cost_usd == 0.0
        assert ctx.usage.turns == 0

    def test_accepts_an_injected_accumulator(self) -> None:
        usage = RunContext()
        usage.input_tokens = 7
        ctx = ExecutionContext(
            runnable="Demo", kind="agent", input=_Input(text="hi"), run_id="r1", usage=usage
        )
        assert ctx.usage.input_tokens == 7


class TestScopedState:
    def test_scoped_creates_a_slice_on_first_use(self) -> None:
        ctx = _ctx()
        ctx.scoped("cost")["handle"] = 42
        assert ctx.state == {"cost": {"handle": 42}}

    def test_scoped_is_stable_across_calls(self) -> None:
        ctx = _ctx()
        ctx.scoped("cost")["handle"] = 42
        assert ctx.scoped("cost")["handle"] == 42

    def test_two_modules_do_not_collide_on_the_same_key(self) -> None:
        ctx = _ctx()
        ctx.scoped("cost")["token"] = "a"
        ctx.scoped("depth")["token"] = "b"
        assert ctx.scoped("cost")["token"] == "a"
        assert ctx.scoped("depth")["token"] == "b"

    def test_scoped_rejects_a_clobbered_slot(self) -> None:
        ctx = _ctx()
        ctx.state["cost"] = "not-a-dict"
        with pytest.raises(TypeError):
            ctx.scoped("cost")
```

- [ ] **Step 5: Run it to verify it fails**

Run: `uv run pytest src/lottie/runtime/tests/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lottie.runtime.context'`.

- [ ] **Step 6: Implement `context.py`**

Create `src/lottie/runtime/context.py`:

```python
"""Per-run carrier threaded through the middleware chain.

Deliberately independent of `lottie.core`: `core/__init__.py` eagerly imports
`base_agent`, so once `BaseAgent` mounts the kernel (S2) a `lottie.core` import here
would be a circular import at package-init time. The kernel therefore depends on a
structural Protocol and a mirrored Literal instead, both pinned by `test_context.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

RunKind = Literal["agent", "skill"]
"""Mirrors `lottie.core.metrics.Kind`. Duplicated, not imported — see module docstring."""


@runtime_checkable
class UsageAccumulator(Protocol):
    """Structural view of `lottie.core.metrics.RunContext`.

    The kernel only carries and reads usage, never constructs it, so it depends on this
    Protocol rather than on the concrete dataclass.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    turns: int


@dataclass
class NullUsage:
    """Zeroed accumulator used when no real one is injected (kernel unit tests)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    turns: int = 0


@dataclass
class ExecutionContext:
    """Mutable per-run carrier. One instance per `Pipeline.execute` call."""

    runnable: str
    kind: RunKind
    input: BaseModel
    run_id: str
    usage: UsageAccumulator = field(default_factory=NullUsage)
    state: dict[str, object] = field(default_factory=dict)

    def scoped(self, module: str) -> dict[str, object]:
        """Return `module`'s private slice of `state`, creating it on first use.

        Middleware never touch `state` directly. Namespacing by module name is what stops
        two independently-authored modules from silently colliding on a key.
        """
        slot = self.state.setdefault(module, {})
        if not isinstance(slot, dict):
            raise TypeError(f"ExecutionContext.state[{module!r}] is not a dict: {type(slot)}")
        return slot
```

- [ ] **Step 7: Run both test files to verify they pass**

Run: `uv run pytest src/lottie/runtime/tests/ -v`
Expected: PASS, 9 tests.

- [ ] **Step 8: Run the lint and type gate**

Run: `uv run ruff check src/lottie/runtime && uv run mypy --strict src/lottie/runtime`
Expected: both clean.

- [ ] **Step 9: Commit**

```bash
git add src/lottie/runtime
git commit -m "feat(runtime): ExecutionContext + kernel import-hygiene guard (V3 S1)"
```

---

### Task 2: Event Runtime — `EventBus` and the hash-only contract

**Files:**
- Create: `src/lottie/runtime/events.py`
- Test: `src/lottie/runtime/tests/test_events.py`

**Interfaces:**
- Consumes: `RunKind` from `lottie.runtime.context`.
- Produces: `RunEvent` (frozen base: `run_id: str`, `runnable: str`, `kind: RunKind`); subclasses `RunStarted` (`+input_sha256: str`), `RunCompleted` (`+input_sha256: str`, `output_sha256: str | None`, `input_tokens: int`, `output_tokens: int`, `cost_usd: float`, `latency_ms: float`), `RunFailed` (`+input_sha256: str`, `error: str`, `latency_ms: float`), `RunBlocked` (`+input_sha256: str`, `blocked_by: str`, `error: str`); `Subscriber` Protocol (`name: str`, `on_event(event: RunEvent) -> None`); `EventBus` (`subscribe(sub)`, `emit(event)`).

- [ ] **Step 1: Write the failing test**

Create `src/lottie/runtime/tests/test_events.py`:

```python
"""Unit tests for the Event Runtime.

Two properties are structural, not conventional, and are the reason this module exists:
a subscriber can never break a run, and no event ever carries raw content.
"""

from __future__ import annotations

import types
import typing
import warnings

import pytest

from lottie.runtime.events import (
    EventBus,
    RunBlocked,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
)

ALL_EVENTS = [RunStarted, RunCompleted, RunFailed, RunBlocked]

SCALARS = {str, int, float, bool, type(None)}

# Names that would signal a raw payload rode along on the bus.
FORBIDDEN_FIELD_NAMES = {
    "input",
    "output",
    "content",
    "payload",
    "prompt",
    "messages",
    "response",
    "text",
    "data",
}


class _Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.seen.append(event)


class _Exploder:
    name = "exploder"

    def on_event(self, event: RunEvent) -> None:
        raise RuntimeError("subscriber blew up")


def _started() -> RunStarted:
    return RunStarted(run_id="r1", runnable="Demo", kind="agent", input_sha256="a" * 64)


class TestEventBus:
    def test_emit_reaches_every_subscriber(self) -> None:
        bus = EventBus()
        first, second = _Recorder(), _Recorder()
        bus.subscribe(first)
        bus.subscribe(second)
        bus.emit(_started())
        assert len(first.seen) == 1
        assert len(second.seen) == 1

    def test_emit_dispatches_in_registration_order(self) -> None:
        order: list[str] = []

        class _Named:
            def __init__(self, label: str) -> None:
                self.name = label

            def on_event(self, event: RunEvent) -> None:
                order.append(self.name)

        bus = EventBus()
        bus.subscribe(_Named("first"))
        bus.subscribe(_Named("second"))
        bus.emit(_started())
        assert order == ["first", "second"]

    def test_a_failing_subscriber_never_propagates(self) -> None:
        bus = EventBus()
        bus.subscribe(_Exploder())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bus.emit(_started())  # must not raise

    def test_a_failing_subscriber_warns_and_names_itself(self) -> None:
        bus = EventBus()
        bus.subscribe(_Exploder())
        with pytest.warns(UserWarning, match="exploder"):
            bus.emit(_started())

    def test_a_failing_subscriber_does_not_starve_the_next_one(self) -> None:
        bus = EventBus()
        survivor = _Recorder()
        bus.subscribe(_Exploder())
        bus.subscribe(survivor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bus.emit(_started())
        assert len(survivor.seen) == 1

    def test_emit_with_no_subscribers_is_a_no_op(self) -> None:
        EventBus().emit(_started())


class TestEventsAreFrozen:
    @pytest.mark.parametrize("model", ALL_EVENTS)
    def test_events_cannot_be_mutated_by_a_subscriber(self, model: type[RunEvent]) -> None:
        assert model.model_config.get("frozen") is True


class TestHashOnlyContract:
    """V3 spec D6: the bus is an observation surface the Plugin SDK opens to third
    parties, so raw payloads on it would be an exfiltration channel."""

    @pytest.mark.parametrize("model", ALL_EVENTS)
    def test_every_field_is_a_scalar(self, model: type[RunEvent]) -> None:
        offenders: list[str] = []
        for field_name, field in model.model_fields.items():
            annotation = field.annotation
            parts = (
                set(typing.get_args(annotation))
                if typing.get_origin(annotation) in (typing.Union, types.UnionType)
                else {annotation}
            )
            # Literal aliases (RunKind) resolve to their str members — also scalar.
            if typing.get_origin(annotation) is typing.Literal:
                parts = {type(arg) for arg in typing.get_args(annotation)}
            if not parts <= SCALARS:
                offenders.append(f"{model.__name__}.{field_name}: {annotation}")
        assert offenders == []

    @pytest.mark.parametrize("model", ALL_EVENTS)
    def test_no_field_is_named_like_a_raw_payload(self, model: type[RunEvent]) -> None:
        leaked = set(model.model_fields) & FORBIDDEN_FIELD_NAMES
        assert leaked == set()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/lottie/runtime/tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lottie.runtime.events'`.

- [ ] **Step 3: Implement `events.py`**

Create `src/lottie/runtime/events.py`:

```python
"""Event Runtime — the fail-open observation stream the pipeline emits onto.

Two rules, both structural rather than conventional:

1. **A subscriber can never break a run.** `EventBus.emit` wraps every dispatch; an
   exception becomes a warning and the next subscriber still runs. This replaces the
   hand-rolled try/except that best-effort observers repeat today.
2. **Events carry scalars and hashes only, never raw content.** The bus is an
   observation surface that the V3 Plugin SDK (E7) opens to third parties, so a raw
   payload here would be an exfiltration channel. `test_events.py` enforces it against
   every event model — a subscriber that genuinely needs content must be a middleware,
   where trust is explicit.
"""

from __future__ import annotations

import warnings
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from lottie.runtime.context import RunKind


class RunEvent(BaseModel):
    """Base for every lifecycle event.

    Frozen so one subscriber cannot mutate what later subscribers observe.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    runnable: str
    kind: RunKind


class RunStarted(RunEvent):
    """Emitted from the innermost frame, immediately before the real work runs."""

    input_sha256: str


class RunCompleted(RunEvent):
    """Emitted from the innermost frame after a successful run, before any post-phase."""

    input_sha256: str
    output_sha256: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class RunFailed(RunEvent):
    """The core execution raised."""

    input_sha256: str
    error: str
    latency_ms: float


class RunBlocked(RunEvent):
    """A middleware aborted the run before the core frame was ever entered.

    `blocked_by` is the name of the last middleware entered — the one that refused.
    """

    input_sha256: str
    blocked_by: str
    error: str


class Subscriber(Protocol):
    """A fail-open observer. Raising is tolerated and warned, never propagated."""

    name: str

    def on_event(self, event: RunEvent) -> None: ...


class EventBus:
    """Synchronous, in-process, registration-ordered fan-out."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []

    def subscribe(self, sub: Subscriber) -> None:
        self._subs.append(sub)

    def emit(self, event: RunEvent) -> None:
        """Dispatch to every subscriber, isolating each failure.

        Deliberately swallows: an observer must never be able to fail a run, and one
        broken observer must never starve the ones registered after it.
        """
        for sub in self._subs:
            try:
                sub.on_event(event)
            except Exception as exc:
                warnings.warn(
                    f"subscriber {sub.name!r} failed on {type(event).__name__}: {exc}",
                    stacklevel=2,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/runtime/tests/test_events.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 5: Run the lint and type gate**

Run: `uv run ruff check src/lottie/runtime && uv run mypy --strict src/lottie/runtime`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/runtime
git commit -m "feat(runtime): EventBus + hash-only event contract (V3 S1)"
```

---

### Task 3: Lifecycle Hooks — `Middleware` Protocol and `Order` constants

Small task, but it defines the contract Tasks 4–6 build against, and a reviewer can reject the order table independently of the executor.

**Files:**
- Create: `src/lottie/runtime/middleware.py`
- Test: `src/lottie/runtime/tests/test_middleware.py`

**Interfaces:**
- Consumes: `ExecutionContext` from `lottie.runtime.context`.
- Produces: `Next` (`Callable[[ExecutionContext], BaseModel]`), `Middleware` Protocol (`name: str`, `order: int`, `__call__(ctx: ExecutionContext, nxt: Next) -> BaseModel`), `Order` (int constants: `SECURITY_INPUT=10`, `POLICY=20`, `COST=30`, `DEPTH=40`, `CAPABILITY=50`, `RECALL=60`, `REFLECT=70`, `SECURITY_OUTPUT=75`, `VERIFY=80`).

- [ ] **Step 1: Write the failing test**

Create `src/lottie/runtime/tests/test_middleware.py`:

```python
"""Tests pinning the Lifecycle Hooks contract and the chain order table.

The order values are not arbitrary: unrolled through onion nesting they must reproduce
`BaseAgent.run`'s existing sequence exactly (V3 spec section 4.5). S2 depends on that.
"""

from __future__ import annotations

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Middleware, Next, Order


class _Output(BaseModel):
    text: str


class _Noop:
    name = "noop"
    order = Order.POLICY

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        return nxt(ctx)


class TestMiddlewareProtocol:
    def test_a_conforming_class_satisfies_the_protocol(self) -> None:
        # Structural typing: a middleware never inherits from anything.
        mw: Middleware = _Noop()
        assert mw.name == "noop"


class TestOrderTable:
    def test_pre_phase_order_matches_the_spec_sequence(self) -> None:
        # Pre-phases run low -> high.
        assert [
            Order.SECURITY_INPUT,
            Order.POLICY,
            Order.COST,
            Order.DEPTH,
            Order.CAPABILITY,
            Order.RECALL,
            Order.REFLECT,
            Order.SECURITY_OUTPUT,
            Order.VERIFY,
        ] == sorted(
            [
                Order.SECURITY_INPUT,
                Order.POLICY,
                Order.COST,
                Order.DEPTH,
                Order.CAPABILITY,
                Order.RECALL,
                Order.REFLECT,
                Order.SECURITY_OUTPUT,
                Order.VERIFY,
            ]
        )

    def test_all_orders_are_distinct(self) -> None:
        values = [
            Order.SECURITY_INPUT,
            Order.POLICY,
            Order.COST,
            Order.DEPTH,
            Order.CAPABILITY,
            Order.RECALL,
            Order.REFLECT,
            Order.SECURITY_OUTPUT,
            Order.VERIFY,
        ]
        assert len(set(values)) == len(values)

    def test_output_side_concerns_unroll_as_verify_then_output_gate_then_reflect(
        self,
    ) -> None:
        # Post-phases run in REVERSE order, so today's `_verify -> check_output ->
        # _maybe_reflect` sequence requires VERIFY > SECURITY_OUTPUT > REFLECT.
        assert Order.VERIFY > Order.SECURITY_OUTPUT > Order.REFLECT

    def test_cost_settles_outside_capability_and_depth_cleanup(self) -> None:
        # Cost's `finally` must be the outermost of the three, matching today's
        # `_cost.settle(handle)` running last in BaseAgent.run.
        assert Order.COST < Order.DEPTH < Order.CAPABILITY
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/lottie/runtime/tests/test_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lottie.runtime.middleware'`.

- [ ] **Step 3: Implement `middleware.py`**

Create `src/lottie/runtime/middleware.py`:

```python
"""Lifecycle Hooks — the ordered, abort-capable wrapper contract.

A middleware receives the context and a `nxt` callable. Calling `nxt(ctx)` runs the rest
of the chain (ultimately the real work); NOT calling it aborts the run. Code before
`nxt` is the pre-phase, code after is the post-phase, and a `try/finally` around `nxt`
is how a concern guarantees cleanup — the property a pure event bus cannot express,
and the reason gates are middleware rather than subscribers.

Post-phases run in REVERSE order of pre-phases (onion nesting). The `Order` values below
are chosen so that unrolling reproduces `BaseAgent.run`'s existing sequence exactly.
See V3 spec section 4.5; `test_middleware.py` pins the relationships that matter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext

type Next = Callable[[ExecutionContext], BaseModel]
"""Continuation into the rest of the chain.

Typed as `BaseModel` rather than `Any`: every runnable output is a pydantic model
(rule 2), so `BaseModel` is the true bound. `Pipeline` narrows it back to the concrete
`OutputT` with a single `cast` at its public boundary.
"""


class Middleware(Protocol):
    """One lifecycle hook. Lower `order` runs earlier in the pre-phase.

    Structural — a middleware never inherits from this.
    """

    name: str
    order: int

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel: ...


class Order:
    """Canonical chain positions.

    Unrolled, the standard chain produces:

        check_input -> policy -> reserve -> depth -> capability -> recall
          -> [core frame: run + emit RunCompleted]
          -> verify -> check_output -> reflect -> recall clear
          -> capability reset -> depth reset -> cost settle

    which is `core/base_agent.py:231-266`. Third-party modules (E7) pick values between
    these; the registry rejects collisions at registration.
    """

    SECURITY_INPUT = 10
    POLICY = 20
    COST = 30
    DEPTH = 40
    CAPABILITY = 50
    RECALL = 60
    REFLECT = 70
    SECURITY_OUTPUT = 75
    VERIFY = 80
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/runtime/tests/test_middleware.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Run the lint and type gate**

Run: `uv run ruff check src/lottie/runtime && uv run mypy --strict src/lottie/runtime`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/runtime
git commit -m "feat(runtime): Middleware protocol + chain order table (V3 S1)"
```

---

### Task 4: `Pipeline` — chain compilation and execution

Events are deliberately deferred to Task 5 so ordering and abort semantics can be reviewed on their own.

**Files:**
- Create: `src/lottie/runtime/pipeline.py`
- Test: `src/lottie/runtime/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `ExecutionContext`, `NullUsage`, `RunKind`, `UsageAccumulator` from `lottie.runtime.context`; `Middleware`, `Next` from `lottie.runtime.middleware`; `EventBus` from `lottie.runtime.events`.
- Produces: `Pipeline[InputT: BaseModel, OutputT: BaseModel]` with constructor keywords `runnable: str`, `kind: RunKind`, `core: Callable[[InputT], OutputT]`, `hasher: Callable[[BaseModel], str]`, `middleware: Sequence[Middleware] = ()`, `bus: EventBus | None = None`, `usage_factory: Callable[[], UsageAccumulator] = NullUsage`; and method `execute(data: InputT) -> OutputT`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/runtime/tests/test_pipeline.py`:

```python
"""Execution semantics of the middleware onion.

These are the tests S2 leans on when it claims the swap-in is behavior-preserving:
ordering, abort, cleanup-on-exception, and reverse post-phase order.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Next
from lottie.runtime.pipeline import Pipeline


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


def _hasher(model: BaseModel) -> str:
    return f"h:{model.model_dump_json()}"


def _core(data: _Input) -> _Output:
    return _Output(text=data.text.upper())


class _Tracer:
    """Records its own pre and post phases into a shared log."""

    def __init__(self, label: str, order: int, log: list[str]) -> None:
        self.name = label
        self.order = order
        self._log = log

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._log.append(f"pre:{self.name}")
        try:
            return nxt(ctx)
        finally:
            self._log.append(f"post:{self.name}")


class _Aborter:
    """A gate that refuses: never calls `nxt`."""

    name = "aborter"

    def __init__(self, order: int) -> None:
        self.order = order

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        raise PermissionError("denied")


def _pipeline(*mw: object, log: list[str] | None = None) -> Pipeline[_Input, _Output]:
    return Pipeline(
        runnable="Demo",
        kind="agent",
        core=_core,
        hasher=_hasher,
        middleware=list(mw),  # type: ignore[arg-type]  # structural Middleware in tests
    )


class TestBareExecution:
    def test_runs_the_core_with_no_middleware(self) -> None:
        assert _pipeline().execute(_Input(text="hi")).text == "HI"

    def test_returns_the_concrete_output_type(self) -> None:
        result = _pipeline().execute(_Input(text="hi"))
        assert isinstance(result, _Output)


class TestOrdering:
    def test_pre_phases_run_low_order_first(self) -> None:
        log: list[str] = []
        _pipeline(_Tracer("a", 10, log), _Tracer("b", 20, log)).execute(_Input(text="x"))
        assert log[:2] == ["pre:a", "pre:b"]

    def test_post_phases_run_in_reverse(self) -> None:
        log: list[str] = []
        _pipeline(_Tracer("a", 10, log), _Tracer("b", 20, log)).execute(_Input(text="x"))
        assert log[-2:] == ["post:b", "post:a"]

    def test_registration_order_does_not_matter(self) -> None:
        log: list[str] = []
        # Registered high-order first; must still run low-order first.
        _pipeline(_Tracer("b", 20, log), _Tracer("a", 10, log)).execute(_Input(text="x"))
        assert log == ["pre:a", "pre:b", "post:b", "post:a"]

    def test_full_onion_sequence(self) -> None:
        log: list[str] = []
        _pipeline(
            _Tracer("a", 10, log), _Tracer("b", 20, log), _Tracer("c", 30, log)
        ).execute(_Input(text="x"))
        assert log == [
            "pre:a",
            "pre:b",
            "pre:c",
            "post:c",
            "post:b",
            "post:a",
        ]


class TestAbort:
    def test_a_middleware_that_does_not_call_next_aborts_the_run(self) -> None:
        with pytest.raises(PermissionError):
            _pipeline(_Aborter(20)).execute(_Input(text="x"))

    def test_abort_skips_the_core(self) -> None:
        ran: list[str] = []

        def _tracking_core(data: _Input) -> _Output:
            ran.append("core")
            return _Output(text=data.text)

        pipe: Pipeline[_Input, _Output] = Pipeline(
            runnable="Demo",
            kind="agent",
            core=_tracking_core,
            hasher=_hasher,
            middleware=[_Aborter(20)],  # type: ignore[list-item]
        )
        with pytest.raises(PermissionError):
            pipe.execute(_Input(text="x"))
        assert ran == []

    def test_outer_post_phases_still_run_when_an_inner_middleware_aborts(self) -> None:
        # This is the cost-settle guarantee: a denied run still releases its reservation.
        log: list[str] = []
        with pytest.raises(PermissionError):
            _pipeline(_Tracer("outer", 10, log), _Aborter(20)).execute(_Input(text="x"))
        assert log == ["pre:outer", "post:outer"]


class TestCoreFailure:
    def test_core_exception_propagates(self) -> None:
        def _boom(data: _Input) -> _Output:
            raise ValueError("kaboom")

        pipe: Pipeline[_Input, _Output] = Pipeline(
            runnable="Demo", kind="agent", core=_boom, hasher=_hasher
        )
        with pytest.raises(ValueError, match="kaboom"):
            pipe.execute(_Input(text="x"))

    def test_post_phases_still_run_when_the_core_raises(self) -> None:
        log: list[str] = []

        def _boom(data: _Input) -> _Output:
            raise ValueError("kaboom")

        pipe: Pipeline[_Input, _Output] = Pipeline(
            runnable="Demo",
            kind="agent",
            core=_boom,
            hasher=_hasher,
            middleware=[_Tracer("a", 10, log)],  # type: ignore[list-item]
        )
        with pytest.raises(ValueError):
            pipe.execute(_Input(text="x"))
        assert log == ["pre:a", "post:a"]


class TestContext:
    def test_each_run_gets_a_fresh_context(self) -> None:
        seen: list[str] = []

        class _Grabber:
            name = "grabber"
            order = 10

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                seen.append(ctx.run_id)
                return nxt(ctx)

        pipe: Pipeline[_Input, _Output] = Pipeline(
            runnable="Demo",
            kind="agent",
            core=_core,
            hasher=_hasher,
            middleware=[_Grabber()],  # type: ignore[list-item]
        )
        pipe.execute(_Input(text="x"))
        pipe.execute(_Input(text="x"))
        assert len(set(seen)) == 2

    def test_context_carries_the_runnable_identity(self) -> None:
        captured: list[ExecutionContext] = []

        class _Grabber:
            name = "grabber"
            order = 10

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                captured.append(ctx)
                return nxt(ctx)

        pipe: Pipeline[_Input, _Output] = Pipeline(
            runnable="Demo",
            kind="agent",
            core=_core,
            hasher=_hasher,
            middleware=[_Grabber()],  # type: ignore[list-item]
        )
        pipe.execute(_Input(text="hi"))
        assert captured[0].runnable == "Demo"
        assert captured[0].kind == "agent"
        assert captured[0].input == _Input(text="hi")

    def test_state_written_by_a_pre_phase_is_visible_in_its_post_phase(self) -> None:
        class _Stateful:
            name = "stateful"
            order = 10

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                ctx.scoped("stateful")["handle"] = "H1"
                try:
                    return nxt(ctx)
                finally:
                    assert ctx.scoped("stateful")["handle"] == "H1"

        pipe: Pipeline[_Input, _Output] = Pipeline(
            runnable="Demo",
            kind="agent",
            core=_core,
            hasher=_hasher,
            middleware=[_Stateful()],  # type: ignore[list-item]
        )
        pipe.execute(_Input(text="x"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/lottie/runtime/tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lottie.runtime.pipeline'`.

- [ ] **Step 3: Implement `pipeline.py`**

Create `src/lottie/runtime/pipeline.py`:

```python
"""Execution core — compiles middleware into an onion and runs it.

`Pipeline` is what `InstrumentedRunnable.run` becomes in S2. It owns exactly two
responsibilities: order the chain, and run it. Everything else is a mounted module.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import cast

from pydantic import BaseModel

from lottie.runtime.context import (
    ExecutionContext,
    NullUsage,
    RunKind,
    UsageAccumulator,
)
from lottie.runtime.events import (
    EventBus,
    RunBlocked,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from lottie.runtime.middleware import Middleware


class Pipeline[InputT: BaseModel, OutputT: BaseModel]:
    """An ordered middleware chain wrapping a core execution function.

    `hasher` is required rather than defaulted so the kernel never has to guess at a
    hashing scheme: S2 injects `lottie.governance.audit.hash_model`, keeping event
    hashes byte-identical to the ones already in the audit ledger.
    """

    def __init__(
        self,
        *,
        runnable: str,
        kind: RunKind,
        core: Callable[[InputT], OutputT],
        hasher: Callable[[BaseModel], str],
        middleware: Sequence[Middleware] = (),
        bus: EventBus | None = None,
        usage_factory: Callable[[], UsageAccumulator] = NullUsage,
    ) -> None:
        self._runnable = runnable
        self._kind = kind
        self._core = core
        self._hasher = hasher
        self._chain = sorted(middleware, key=lambda m: m.order)
        self._bus = bus if bus is not None else EventBus()
        self._usage_factory = usage_factory

    def execute(self, data: InputT) -> OutputT:
        """Run `data` through the chain and return the core's typed output.

        The single `cast` is the only place typing is narrowed: the chain is
        heterogeneous and speaks `BaseModel`, but the core function is typed
        `Callable[[InputT], OutputT]`, so the value flowing back out is an `OutputT`
        by construction.
        """
        ctx = ExecutionContext(
            runnable=self._runnable,
            kind=self._kind,
            input=data,
            run_id=uuid.uuid4().hex,
            usage=self._usage_factory(),
        )
        entered: list[str] = []
        reached_core = False

        def step(index: int) -> Callable[[ExecutionContext], BaseModel]:
            def call(c: ExecutionContext) -> BaseModel:
                nonlocal reached_core
                if index == len(self._chain):
                    reached_core = True
                    return self._core_frame(c)
                mw = self._chain[index]
                entered.append(mw.name)
                return mw(c, step(index + 1))

            return call

        try:
            return cast(OutputT, step(0)(ctx))
        except Exception as exc:
            if not reached_core:
                # A gate refused before the work ever started — today's `_write_block`.
                self._emit_blocked(ctx, entered, exc)
            raise

    def _core_frame(self, ctx: ExecutionContext) -> BaseModel:
        """Innermost frame: run the real work, emit lifecycle events from HERE.

        Emitting here rather than from `execute` is load-bearing. It means observers see
        the completed run BEFORE any middleware post-phase, so an audit subscriber
        records the real cost before the cost middleware's `finally` settles the
        reservation — the invariant documented at `core/base_agent.py:259`.
        """
        input_hash = self._hasher(ctx.input)
        self._bus.emit(
            RunStarted(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                input_sha256=input_hash,
            )
        )
        start = perf_counter()
        try:
            output = self._core(cast(InputT, ctx.input))
        except Exception as exc:
            self._bus.emit(
                RunFailed(
                    run_id=ctx.run_id,
                    runnable=ctx.runnable,
                    kind=ctx.kind,
                    input_sha256=input_hash,
                    error=repr(exc),
                    latency_ms=(perf_counter() - start) * 1000,
                )
            )
            raise
        self._bus.emit(
            RunCompleted(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                input_sha256=input_hash,
                output_sha256=self._hasher(output),
                input_tokens=ctx.usage.input_tokens,
                output_tokens=ctx.usage.output_tokens,
                cost_usd=ctx.usage.cost_usd,
                latency_ms=(perf_counter() - start) * 1000,
            )
        )
        return output

    def _emit_blocked(
        self, ctx: ExecutionContext, entered: list[str], exc: Exception
    ) -> None:
        self._bus.emit(
            RunBlocked(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                input_sha256=self._hasher(ctx.input),
                blocked_by=entered[-1] if entered else "unknown",
                error=str(exc),
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/runtime/tests/test_pipeline.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Run the lint and type gate**

Run: `uv run ruff check src/lottie/runtime && uv run mypy --strict src/lottie/runtime`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/runtime
git commit -m "feat(runtime): Pipeline — onion execution, ordering, abort semantics (V3 S1)"
```

---

### Task 5: Pipeline event emission — the innermost-frame invariant

The implementation landed in Task 4; this task proves the emission ordering that S2's audit-before-settle correctness depends on. A reviewer can reject these guarantees while accepting Task 4's execution semantics.

**Files:**
- Test: `src/lottie/runtime/tests/test_pipeline_events.py`

**Interfaces:**
- Consumes: `Pipeline` from Task 4; the event types from Task 2.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/runtime/tests/test_pipeline_events.py`:

```python
"""Lifecycle-event emission, including the ordering invariant S2 depends on.

`RunCompleted` fires from the INNERMOST frame, so a subscriber (audit) observes the run
before any middleware post-phase (cost settle). Today that ordering is hand-maintained in
`BaseAgent.run`'s nested finally blocks and documented at `core/base_agent.py:259`.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.events import (
    EventBus,
    RunBlocked,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
)
from lottie.runtime.middleware import Next
from lottie.runtime.pipeline import Pipeline


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


def _hasher(model: BaseModel) -> str:
    return f"h:{model.model_dump_json()}"


def _core(data: _Input) -> _Output:
    return _Output(text=data.text.upper())


class _Recorder:
    name = "recorder"

    def __init__(self, log: list[str] | None = None) -> None:
        self.seen: list[RunEvent] = []
        self._log = log

    def on_event(self, event: RunEvent) -> None:
        self.seen.append(event)
        if self._log is not None:
            self._log.append(f"event:{type(event).__name__}")

    def only(self, model: type[RunEvent]) -> list[RunEvent]:
        return [e for e in self.seen if isinstance(e, model)]


def _pipe(
    bus: EventBus,
    *mw: object,
    core: object = _core,
) -> Pipeline[_Input, _Output]:
    return Pipeline(
        runnable="Demo",
        kind="agent",
        core=core,  # type: ignore[arg-type]
        hasher=_hasher,
        middleware=list(mw),  # type: ignore[arg-type]
        bus=bus,
    )


class TestHappyPath:
    def test_emits_started_then_completed(self) -> None:
        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        _pipe(bus).execute(_Input(text="hi"))
        assert [type(e).__name__ for e in rec.seen] == ["RunStarted", "RunCompleted"]

    def test_completed_carries_the_output_hash(self) -> None:
        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        _pipe(bus).execute(_Input(text="hi"))
        completed = rec.only(RunCompleted)[0]
        assert isinstance(completed, RunCompleted)
        assert completed.output_sha256 == _hasher(_Output(text="HI"))

    def test_started_and_completed_share_one_run_id(self) -> None:
        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        _pipe(bus).execute(_Input(text="hi"))
        assert len({e.run_id for e in rec.seen}) == 1

    def test_completed_reports_latency(self) -> None:
        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        _pipe(bus).execute(_Input(text="hi"))
        completed = rec.only(RunCompleted)[0]
        assert isinstance(completed, RunCompleted)
        assert completed.latency_ms >= 0.0


class TestInnermostFrameInvariant:
    def test_completed_fires_before_any_middleware_post_phase(self) -> None:
        """The audit-before-settle guarantee, asserted directly."""
        log: list[str] = []
        bus = EventBus()
        bus.subscribe(_Recorder(log))

        class _Settler:
            name = "cost"
            order = 30

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                log.append("pre:cost")
                try:
                    return nxt(ctx)
                finally:
                    log.append("settle:cost")

        _pipe(bus, _Settler()).execute(_Input(text="hi"))
        assert log == [
            "pre:cost",
            "event:RunStarted",
            "event:RunCompleted",
            "settle:cost",
        ]

    def test_started_fires_after_every_pre_phase(self) -> None:
        log: list[str] = []
        bus = EventBus()
        bus.subscribe(_Recorder(log))

        class _Gate:
            name = "gate"
            order = 10

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                log.append("pre:gate")
                return nxt(ctx)

        _pipe(bus, _Gate()).execute(_Input(text="hi"))
        assert log.index("pre:gate") < log.index("event:RunStarted")


class TestFailure:
    def test_core_failure_emits_run_failed(self) -> None:
        def _boom(data: _Input) -> _Output:
            raise ValueError("kaboom")

        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        with pytest.raises(ValueError):
            _pipe(bus, core=_boom).execute(_Input(text="hi"))
        assert len(rec.only(RunFailed)) == 1

    def test_core_failure_emits_no_run_completed(self) -> None:
        def _boom(data: _Input) -> _Output:
            raise ValueError("kaboom")

        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        with pytest.raises(ValueError):
            _pipe(bus, core=_boom).execute(_Input(text="hi"))
        assert rec.only(RunCompleted) == []


class TestBlocked:
    def test_a_gate_abort_emits_run_blocked(self) -> None:
        class _Denier:
            name = "policy"
            order = 20

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                raise PermissionError("denied")

        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        with pytest.raises(PermissionError):
            _pipe(bus, _Denier()).execute(_Input(text="hi"))
        blocked = rec.only(RunBlocked)
        assert len(blocked) == 1

    def test_run_blocked_names_the_refusing_middleware(self) -> None:
        class _Denier:
            name = "policy"
            order = 20

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                raise PermissionError("denied")

        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        with pytest.raises(PermissionError):
            _pipe(bus, _Denier()).execute(_Input(text="hi"))
        blocked = rec.only(RunBlocked)[0]
        assert isinstance(blocked, RunBlocked)
        assert blocked.blocked_by == "policy"

    def test_a_blocked_run_emits_no_started_or_completed(self) -> None:
        class _Denier:
            name = "policy"
            order = 20

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                raise PermissionError("denied")

        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        with pytest.raises(PermissionError):
            _pipe(bus, _Denier()).execute(_Input(text="hi"))
        assert rec.only(RunStarted) == []
        assert rec.only(RunCompleted) == []

    def test_a_core_failure_is_not_reported_as_blocked(self) -> None:
        # The distinction that makes RunBlocked meaningful: the work never started.
        def _boom(data: _Input) -> _Output:
            raise ValueError("kaboom")

        bus, rec = EventBus(), _Recorder()
        bus.subscribe(rec)
        with pytest.raises(ValueError):
            _pipe(bus, core=_boom).execute(_Input(text="hi"))
        assert rec.only(RunBlocked) == []


class TestSubscriberIsolationEndToEnd:
    def test_a_broken_subscriber_cannot_fail_a_run(self) -> None:
        class _Exploder:
            name = "exploder"

            def on_event(self, event: RunEvent) -> None:
                raise RuntimeError("blew up")

        bus = EventBus()
        bus.subscribe(_Exploder())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _pipe(bus).execute(_Input(text="hi"))
        assert result.text == "HI"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/lottie/runtime/tests/test_pipeline_events.py -v`
Expected: PASS if Task 4's implementation is complete and correct. If any test fails, fix `pipeline.py` — the failure is a real defect in Task 4's emission ordering, not a missing feature.

- [ ] **Step 3: Run the full kernel suite**

Run: `uv run pytest src/lottie/runtime -v`
Expected: PASS, 57 tests.

- [ ] **Step 4: Run the lint and type gate**

Run: `uv run ruff check src/lottie/runtime && uv run mypy --strict src/lottie/runtime`
Expected: both clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/runtime
git commit -m "test(runtime): pin innermost-frame emission + RunBlocked semantics (V3 S1)"
```

---

### Task 6: `ModuleRegistry` — composition and order-conflict detection

**Files:**
- Create: `src/lottie/runtime/registry.py`
- Test: `src/lottie/runtime/tests/test_registry.py`

**Interfaces:**
- Consumes: `EventBus`, `Subscriber` from `lottie.runtime.events`; `Middleware` from `lottie.runtime.middleware`.
- Produces: `Deps` (frozen dataclass: `bus: EventBus`, `root: Path`); `Mountable` (`Middleware | Subscriber`); `ModuleFactory[CfgT]` Protocol (`name: str`, `order: int`, `build(cfg: CfgT, deps: Deps) -> Mountable | None`); `ModuleRegistry[CfgT]` (`register(factory)`, `build(cfg, deps) -> tuple[list[Middleware], list[Subscriber]]`); `ModuleConflictError`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/runtime/tests/test_registry.py`:

```python
"""Module Orchestrator support.

Conflicts are rejected at REGISTRATION, not at run time: a plugin must not be able to
silently take a security middleware's slot, and the failure should surface at startup
where an operator sees it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.events import EventBus, RunEvent
from lottie.runtime.middleware import Next, Order
from lottie.runtime.registry import (
    Deps,
    ModuleConflictError,
    ModuleRegistry,
    Mountable,
)


class _Config(BaseModel):
    audit_enabled: bool = True
    policy_enabled: bool = True


class _PolicyMiddleware:
    name = "policy"
    order = Order.POLICY

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        return nxt(ctx)


class _AuditSubscriber:
    name = "audit"

    def on_event(self, event: RunEvent) -> None:
        return None


class _PolicyFactory:
    name = "policy"
    order = Order.POLICY

    def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
        return _PolicyMiddleware() if cfg.policy_enabled else None


class _AuditFactory:
    name = "audit"
    order = 900

    def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
        return _AuditSubscriber() if cfg.audit_enabled else None


def _deps(tmp_path: Path) -> Deps:
    return Deps(bus=EventBus(), root=tmp_path)


class TestRegistration:
    def test_registers_a_factory(self) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        assert reg.names() == ["policy"]

    def test_duplicate_name_is_rejected(self) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        with pytest.raises(ModuleConflictError, match="policy"):
            reg.register(_PolicyFactory())

    def test_duplicate_order_is_rejected_at_registration(self) -> None:
        class _Impostor:
            name = "impostor"
            order = Order.POLICY

            def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
                return None

        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        with pytest.raises(ModuleConflictError) as exc:
            reg.register(_Impostor())
        # The message must name both claimants so the operator can act on it.
        assert "impostor" in str(exc.value)
        assert "policy" in str(exc.value)

    def test_distinct_orders_coexist(self) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_AuditFactory())
        assert reg.names() == ["policy", "audit"]


class TestBuild:
    def test_sorts_middleware_into_a_chain(self, tmp_path: Path) -> None:
        class _Early:
            name = "early"
            order = Order.SECURITY_INPUT

            def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
                class _MW:
                    name = "early"
                    order = Order.SECURITY_INPUT

                    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                        return nxt(ctx)

                return _MW()

        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_Early())
        middleware, _ = reg.build(_Config(), _deps(tmp_path))
        assert [m.name for m in middleware] == ["early", "policy"]

    def test_separates_subscribers_from_middleware(self, tmp_path: Path) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_AuditFactory())
        middleware, subscribers = reg.build(_Config(), _deps(tmp_path))
        assert [m.name for m in middleware] == ["policy"]
        assert [s.name for s in subscribers] == ["audit"]

    def test_a_factory_returning_none_is_not_mounted(self, tmp_path: Path) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_AuditFactory())
        middleware, subscribers = reg.build(
            _Config(policy_enabled=False, audit_enabled=False), _deps(tmp_path)
        )
        assert middleware == []
        assert subscribers == []

    def test_build_on_an_empty_registry_returns_empty_lists(self, tmp_path: Path) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        assert reg.build(_Config(), _deps(tmp_path)) == ([], [])

    def test_build_is_repeatable(self, tmp_path: Path) -> None:
        # The orchestrator builds one chain per agent; registration state must not be
        # consumed by the first build.
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        first, _ = reg.build(_Config(), _deps(tmp_path))
        second, _ = reg.build(_Config(), _deps(tmp_path))
        assert [m.name for m in first] == [m.name for m in second]


class TestDeps:
    def test_deps_is_frozen(self, tmp_path: Path) -> None:
        deps = _deps(tmp_path)
        with pytest.raises(Exception):
            deps.root = tmp_path  # type: ignore[misc]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/lottie/runtime/tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lottie.runtime.registry'`.

- [ ] **Step 3: Implement `registry.py`**

Create `src/lottie/runtime/registry.py`:

```python
"""Module Orchestrator support — the registry S6 wires `instantiate_agent` onto.

A module declares a factory; the registry composes factories into a chain. Adding a
cross-cutting concern becomes "register one factory" instead of the four-file edit
that `BaseAgent.__init__` + `run` + `instantiate_agent` + `AgentConfig` requires today.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lottie.runtime.events import EventBus, Subscriber
from lottie.runtime.middleware import Middleware


class ModuleConflictError(RuntimeError):
    """Two modules claim the same name or the same chain position.

    Raised at registration rather than at run time so the failure surfaces at startup,
    and so a plugin can never silently displace a security middleware.
    """


@dataclass(frozen=True)
class Deps:
    """Constructor dependencies handed to a module factory.

    Deliberately minimal. S1 has no real modules, so adding fields now would be
    inventing API ahead of need; S6 extends this once the migrated modules declare what
    they actually require. Dependencies are injected here rather than imported by the
    module, which is what keeps the subsystem-to-runtime edge one-directional.
    """

    bus: EventBus
    root: Path


type Mountable = Middleware | Subscriber


class ModuleFactory[CfgT](Protocol):
    """Builds a module from configuration, or declines to.

    Generic over the config type so the kernel never imports `AgentConfig` from
    `lottie.project` — S6 instantiates this as `ModuleFactory[AgentConfig]`.
    """

    name: str
    order: int

    def build(self, cfg: CfgT, deps: Deps) -> Mountable | None: ...


class ModuleRegistry[CfgT]:
    """Ordered collection of module factories."""

    def __init__(self) -> None:
        self._factories: list[ModuleFactory[CfgT]] = []

    def names(self) -> list[str]:
        """Registered module names, in registration order."""
        return [f.name for f in self._factories]

    def register(self, factory: ModuleFactory[CfgT]) -> None:
        """Add `factory`, rejecting name and order collisions.

        Order uniqueness is enforced across all factories, including ones that produce
        subscribers. Subscribers do not strictly need a distinct order, but a global
        rule is trivially satisfiable and leaves no ambiguity about who owns a slot.
        """
        for existing in self._factories:
            if existing.name == factory.name:
                raise ModuleConflictError(f"module name {factory.name!r} is already registered")
            if existing.order == factory.order:
                raise ModuleConflictError(
                    f"module {factory.name!r} claims order {factory.order}, "
                    f"already held by {existing.name!r}"
                )
        self._factories.append(factory)

    def build(self, cfg: CfgT, deps: Deps) -> tuple[list[Middleware], list[Subscriber]]:
        """Instantiate every enabled module, split into chain and observers.

        A factory returning None is disabled by configuration and costs nothing.
        """
        middleware: list[Middleware] = []
        subscribers: list[Subscriber] = []
        for factory in self._factories:
            mounted = factory.build(cfg, deps)
            if mounted is None:
                continue
            if isinstance(mounted, Middleware):
                middleware.append(mounted)
            else:
                subscribers.append(mounted)
        middleware.sort(key=lambda m: m.order)
        return middleware, subscribers
```

> **Implementation note for the executor:** `isinstance(mounted, Middleware)` requires
> `Middleware` to be `@runtime_checkable`. If Task 3's Protocol is not decorated, add
> `@runtime_checkable` to it in `middleware.py` and re-run Task 3's tests. Prefer that over
> a `hasattr(mounted, "__call__")` check — a `Subscriber` could plausibly gain a
> `__call__` later, whereas the `order` attribute is what actually distinguishes them.
> If `runtime_checkable` proves awkward under `mypy --strict`, discriminate on
> `hasattr(mounted, "order")` instead and note the choice in `ARCHITECTURE.md`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/runtime/tests/test_registry.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Run the lint and type gate**

Run: `uv run ruff check src/lottie/runtime && uv run mypy --strict src/lottie/runtime`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/runtime
git commit -m "feat(runtime): ModuleRegistry with registration-time conflict detection (V3 S1)"
```

---

### Task 7: Performance microbenchmark and overhead budget

The spec makes performance a gate for every later slice, which requires a number to gate against. This task produces it.

**Files:**
- Test: `src/lottie/runtime/tests/test_perf.py`

**Interfaces:**
- Consumes: `Pipeline` from Task 4.
- Produces: the documented overhead budget referenced by S2–S7.

- [ ] **Step 1: Write the benchmark**

Create `src/lottie/runtime/tests/test_perf.py`:

```python
"""Dispatch-overhead budget for the middleware chain.

The V3 spec makes performance a gate for every slice, which requires a baseline. The
bound below is deliberately generous — roughly two orders of magnitude above the
measured cost — because this runs on shared CI where a tight bound would be flaky. It
is a runaway-regression guard, not a precision instrument: it catches someone adding an
I/O call or an O(n^2) walk to the hot path, which is exactly the failure mode that
matters.
"""

from __future__ import annotations

from time import perf_counter

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.events import EventBus, RunEvent
from lottie.runtime.middleware import Next
from lottie.runtime.pipeline import Pipeline

ITERATIONS = 1000
CHAIN_DEPTH = 10
SUBSCRIBERS = 3

# Budget: total wall time per run, chain + events, excluding the core function.
MAX_OVERHEAD_MS = 1.0


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


def _hasher(model: BaseModel) -> str:
    return "h" * 64  # constant: this benchmark measures dispatch, not hashing


def _core(data: _Input) -> _Output:
    return _Output(text=data.text)


class _Passthrough:
    def __init__(self, order: int) -> None:
        self.name = f"mw{order}"
        self.order = order

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        return nxt(ctx)


class _Sink:
    name = "sink"

    def on_event(self, event: RunEvent) -> None:
        return None


def _measure(pipe: Pipeline[_Input, _Output]) -> float:
    """Mean milliseconds per `execute` call."""
    data = _Input(text="benchmark")
    pipe.execute(data)  # warm up import/attribute caches
    start = perf_counter()
    for _ in range(ITERATIONS):
        pipe.execute(data)
    return ((perf_counter() - start) * 1000) / ITERATIONS


class TestDispatchOverhead:
    def test_full_chain_stays_within_budget(self) -> None:
        bus = EventBus()
        for _ in range(SUBSCRIBERS):
            bus.subscribe(_Sink())
        pipe: Pipeline[_Input, _Output] = Pipeline(
            runnable="Bench",
            kind="agent",
            core=_core,
            hasher=_hasher,
            middleware=[_Passthrough((i + 1) * 10) for i in range(CHAIN_DEPTH)],
            bus=bus,
        )
        per_run_ms = _measure(pipe)
        assert per_run_ms < MAX_OVERHEAD_MS, (
            f"{CHAIN_DEPTH} middleware + {SUBSCRIBERS} subscribers cost "
            f"{per_run_ms:.4f} ms/run, budget {MAX_OVERHEAD_MS} ms"
        )

    def test_chain_cost_grows_no_worse_than_linearly(self) -> None:
        """Guards the shape of the cost, which a wall-clock bound alone cannot."""

        def _build(depth: int) -> Pipeline[_Input, _Output]:
            return Pipeline(
                runnable="Bench",
                kind="agent",
                core=_core,
                hasher=_hasher,
                middleware=[_Passthrough((i + 1) * 10) for i in range(depth)],
            )

        shallow = _measure(_build(2))
        deep = _measure(_build(20))
        # 10x the middleware must not cost more than 20x the time. Loose enough to
        # survive CI jitter, tight enough to catch quadratic dispatch.
        assert deep < shallow * 20 + MAX_OVERHEAD_MS
```

- [ ] **Step 2: Run it and record the measured number**

Run: `uv run pytest src/lottie/runtime/tests/test_perf.py -v -s`
Expected: PASS. Note the actual `ms/run` figure — it goes into `ARCHITECTURE.md` in Task 8 as the recorded S1 baseline.

- [ ] **Step 3: Run it three times to confirm it is not flaky**

Run: `uv run pytest src/lottie/runtime/tests/test_perf.py -q && uv run pytest src/lottie/runtime/tests/test_perf.py -q && uv run pytest src/lottie/runtime/tests/test_perf.py -q`
Expected: PASS all three times. If any run fails, the budget is too tight for this machine — raise `MAX_OVERHEAD_MS` and record the reason in `ARCHITECTURE.md` rather than deleting the test.

- [ ] **Step 4: Run the lint and type gate**

Run: `uv run ruff check src/lottie/runtime && uv run mypy --strict src/lottie/runtime`
Expected: both clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/runtime
git commit -m "test(runtime): dispatch overhead budget + linearity guard (V3 S1)"
```

---

### Task 8: `ARCHITECTURE.md` and the full-repo gate

**Files:**
- Create: `src/lottie/runtime/ARCHITECTURE.md`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: the module documentation the methodology's Definition of Done requires.

- [ ] **Step 1: Write the module documentation**

Create `src/lottie/runtime/ARCHITECTURE.md`. Fill `<measured>` with the figure recorded in Task 7 Step 2.

```markdown
# `lottie.runtime` — the execution kernel

## Responsibility

Order and run the cross-cutting concerns that wrap every agent and skill run. The kernel
owns exactly two things: **what runs in what order**, and **who gets told about it**.
Everything else is a mounted module living in its own subsystem.

## The two primitives

| | Middleware (Lifecycle Hooks) | Subscriber (Event Runtime) |
|---|---|---|
| Can abort a run | **Yes** — by not calling `nxt`, or by raising | **No** — exceptions are warned and swallowed |
| Ordering | explicit `order` int, low runs first | registration order |
| Can span `_execute` | **Yes** — `try/finally` around `nxt` | No |
| Sees raw input/output | Yes, via `ctx.input` | **No** — events carry scalars and hashes only |
| Use for | security, policy, cost, capability, recall, reflect, verify | audit, otel, benchmark, telemetry |

Choosing between them is not a style question. A concern that must **block** or must
**guarantee cleanup** is a middleware. A concern that merely **observes** is a
subscriber, and making it one is what structurally prevents it from breaking a run.

## Lifecycle

```
Pipeline.execute(data)
  |
  +- build ExecutionContext (run_id, usage, state)
  |
  +- pre-phases, ascending order .......... 10, 20, 30, ...
  |
  +- CORE FRAME  emit RunStarted
  |              run the real work
  |              emit RunCompleted / RunFailed
  |
  +- post-phases, descending order ........ ..., 30, 20, 10
  |
  +- on abort before the core frame: emit RunBlocked
```

### Why events fire from the innermost frame

`RunCompleted` is emitted **inside** the core frame, before any middleware post-phase.
That ordering is load-bearing: it means an audit subscriber records the run's real cost
before the cost middleware's `finally` settles the reservation — the invariant currently
documented by hand at `core/base_agent.py:259`. Moving the emission out to `execute`
would silently invert it. `tests/test_pipeline_events.py::TestInnermostFrameInvariant`
pins it.

## Public interfaces

| Symbol | Module | Purpose |
|---|---|---|
| `ExecutionContext` | `context` | Per-run carrier. `scoped(name)` gives a module its private state slice. |
| `UsageAccumulator` | `context` | Structural view of `core.metrics.RunContext`. |
| `Middleware`, `Next`, `Order` | `middleware` | The hook contract and the canonical chain positions. |
| `RunEvent` and subclasses, `Subscriber`, `EventBus` | `events` | The observation stream. |
| `Pipeline` | `pipeline` | Compiles and runs the onion. |
| `ModuleFactory`, `ModuleRegistry`, `Deps` | `registry` | Composition and conflict detection. |

## Extension points

Add a cross-cutting concern by writing a `ModuleFactory` and registering it. Do **not**
add a step to a runnable's `run` method — that is the coupling V3 exists to remove.

- Pick an `order` between the `Order` constants. The registry rejects collisions at
  registration, so a clash fails at startup rather than silently reordering a gate.
- Return `None` from `build` when configuration disables the module; it then costs
  nothing at run time.
- Dependencies arrive through `Deps`. A module never imports the kernel's consumers.

## Hard constraint: no subsystem imports

Nothing under `lottie/runtime/` may import `lottie.core`, `lottie.governance`,
`lottie.memory`, `lottie.security`, or `lottie.llm`. `core/__init__.py` eagerly imports
`base_agent`, so a kernel-to-core import becomes a circular import at package-init time
the moment `BaseAgent` mounts the kernel. `tests/test_imports.py` enforces this by AST
scan.

Two consequences, both pinned by tests rather than left to discipline:

- `RunKind` mirrors `core.metrics.Kind` instead of importing it.
- `Pipeline` takes an injected `hasher` instead of importing
  `governance.audit.hash_model`.

## Performance

Recorded S1 baseline, 10 middleware + 3 subscribers: **`<measured>` ms/run**.
Budget enforced by `tests/test_perf.py`: **1.0 ms/run**, roughly two orders of magnitude
of headroom. The bound is a runaway-regression guard for shared CI, not a precision
instrument; `test_chain_cost_grows_no_worse_than_linearly` guards the *shape* of the cost,
which a wall-clock bound alone cannot.

## Status

S1 ships the kernel with **no consumers**. `BaseAgent` and `BaseSkill` are untouched;
S2 swaps `InstrumentedRunnable.run` onto `Pipeline` using thin adapters over the code
that already exists. See `docs/superpowers/specs/2026-07-30-v3-runtime-kernel-design.md`.
```

- [ ] **Step 2: Verify the kernel is genuinely unconsumed**

Run: `grep -rn "lottie.runtime" src/lottie --include="*.py" | grep -v "src/lottie/runtime/"`
Expected: no output. Any hit means S2 work leaked into S1.

- [ ] **Step 3: Run the full repository gate**

Run: `uv sync --dev --all-extras && uv run ruff check . && uv run mypy --strict src && uv run pytest -q`
Expected: all clean. The pre-existing test count must be **unchanged plus the kernel's new tests** — no existing test may have been modified or removed.

- [ ] **Step 4: Confirm no file outside the kernel changed**

Run: `git diff --name-only main...HEAD`
Expected: only paths under `src/lottie/runtime/` and this plan file.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/runtime
git commit -m "docs(runtime): ARCHITECTURE.md — kernel responsibilities, lifecycle, extension points (V3 S1)"
```

- [ ] **Step 6: Push and open the PR**

```bash
gh auth switch --user cdiaz19
git push -u origin feat/v3-s1-runtime-kernel
gh pr create --base main --title "feat(runtime): V3 S1 — execution kernel (middleware chain + event bus)"
```

- [ ] **Step 7: Confirm CI is green before merging**

Run: `gh pr checks`
Expected: all green. Per rule 7b, do not squash-merge on red.

- [ ] **Step 8: Build lab round R28 before merging**

S1 does not merge until its lab round is green in `cdiaz19/lottie-lab`. R28 validates the
kernel standalone — it has no orchestrator consumers yet, so the round exercises
`Pipeline` directly:

1. A chain of realistic gates (a denier, a budget-style reserve/settle, a ContextVar
   window) proving order and cleanup end-to-end.
2. A subscriber that raises on every event, proving a run still completes.
3. A subscriber attempting to read raw input from an event, proving D6 holds.
4. A plugin-style factory claiming an occupied `order`, proving startup rejection.

---

## Self-Review

**Spec coverage (§4 and §7, slice S1):**

| Spec item | Task |
|---|---|
| §4.1 `ExecutionContext`, reuses `RunContext`, does not rename it | 1 |
| §4.2 `Middleware` Protocol, `Next` | 3 |
| §4.3 `events.py`, frozen models, wrapped dispatch, D6 hash-only contract test | 2 |
| §4.4 `pipeline.py`, onion, innermost-frame emission | 4, 5 |
| §4.5 order table reproduces `BaseAgent.run` | 3 (constants), 4 (semantics), 5 (emission) |
| §4.6 event fires from innermost frame so audit precedes settle | 5 |
| §4.7 mounts on `InstrumentedRunnable`, one path for agents and skills | Deferred to S2 by design — S1 has no consumers. `Pipeline` is generic over `InputT`/`OutputT` and takes `kind`, so it is already skill-ready. |
| §6 `ModuleRegistry`, `Deps`, order-conflict detection | 6 |
| §7 S1 row: "ships the perf microbenchmark + overhead budget" | 7 |
| §7 "zero behavior change" | 8 Steps 2 and 4 verify it mechanically |
| §9 "zero new runtime dependencies" | Global Constraints; no `pyproject.toml` edit in any task |
| §10 DoD "Documentation updated" | 8 |

**Two deviations from the spec, both improvements, both deliberate:**

1. **Spec §4.2 anticipated one `Any` seam at `Next`.** This plan types it `BaseModel`
   instead. Every runnable output is a pydantic model (rule 2), so `BaseModel` is the
   true bound; `Pipeline.execute` narrows back to `OutputT` with a single documented
   `cast`. Result: the kernel has no `Any` at all, so rule 6 needs no exception.
2. **Spec §4.1 implied importing `core.metrics`.** Verification found
   `core/__init__.py:1-5` eagerly imports `base_agent`, which makes that a package-init
   cycle once S2 lands. The kernel therefore mirrors `RunKind` and depends on a
   structural `UsageAccumulator` Protocol, with `test_context.py` pinning both against
   the real `core.metrics` so they cannot drift. `test_imports.py` enforces the rule
   generally.

**One addition beyond the spec:** `RunBlocked` emission is implemented in S1 rather than
left to S3. The pipeline is the only component that can tell "a gate refused before the
work started" from "the work failed", so the detection has to live here. This directly
serves today's `_write_block` behavior (`base_agent.py:293-318`).

**Placeholder scan:** none. Every step contains runnable code or an exact command. The
single `<measured>` token in Task 8 is filled from Task 7 Step 2's output, and that
dependency is stated in both tasks.

**Type consistency:** `RunKind` (not `Kind`) throughout; `blocked_by` (not `reason`) on
`RunBlocked` in both `events.py` and its tests; `Mountable = Middleware | Subscriber` used
identically in `registry.py` and `test_registry.py`; `hasher: Callable[[BaseModel], str]`
matches every `_hasher` stub in the test files; `Order` constants referenced by the same
names in Tasks 3 and 6.

**One risk flagged for the executor:** Task 6's `isinstance(mounted, Middleware)` needs
`Middleware` to be `@runtime_checkable`. Task 6 Step 3 carries an implementation note with
the fallback if that proves awkward under `mypy --strict`.
