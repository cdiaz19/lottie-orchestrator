# `lottie.runtime` — the execution kernel

## Responsibility

Order and run the cross-cutting concerns that wrap every agent and skill run. The kernel
owns exactly two things: **what runs in what order**, and **who gets told about it**.
Everything else is a mounted module.

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
documented by hand at `core/base_agent.py:461-466`. Moving the emission out to `execute`
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

This is not hypothetical: V2 S5b hit exactly this cycle
(`core → session → security → core.__init__ → base_agent`) and had to break it with a
lazy import.

Two consequences, both pinned by tests rather than left to discipline:

- `RunKind` mirrors `core.metrics.Kind` instead of importing it.
- `Pipeline` takes an injected `hasher` instead of importing
  `governance.audit.hash_model`.

## Performance

Recorded S1 baseline, 10 middleware + 3 subscribers: **0.00906 ms/run**.
Budget enforced by `tests/test_perf.py`: **1.0 ms/run** — roughly 110× headroom. The bound
is a runaway-regression guard for shared CI, not a precision instrument;
`test_chain_cost_grows_no_worse_than_linearly` guards the *shape* of the cost, which a
wall-clock bound alone cannot.

## Status

S1 ships the kernel with **no consumers**. `BaseAgent` and `BaseSkill` are untouched;
S2 swaps `InstrumentedRunnable.run` onto `Pipeline` using thin adapters over the code
that already exists. See `docs/superpowers/specs/2026-07-30-v3-runtime-kernel-design.md`.
