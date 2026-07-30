# Lottie V3 — Runtime Kernel (Event Runtime + Lifecycle Hooks)

> Epic design. Target: **v3.0.0**. Date: 2026-07-30.
> Theme: extract a runtime kernel so cross-cutting concerns become **registered modules**
> instead of hand-sequenced steps inside `BaseAgent.run` — the step that moves Lottie from
> a framework toward an Agent Operating System.
> Methodology: `docs/METHODOLOGY.md` (design-first, modular, incremental, lab-validated).

---

## 1. Precondition

**V3 does not start until `v2.0.0` is tagged.** V2 (Phase 5) still has S3 distillation,
S4 learning-delta eval, S5 long-running harness, S6 release + red-team outstanding. Two
concurrent epics would both touch `core/base_agent.py` and conflict continuously.

### 1.1 The one constraint V3 places on V2 S5

V2 S5 ships context compaction. E4 (Context Compiler, v3.1) must be able to **absorb** it
rather than rewrite it, so S5 builds compaction as a **pure function with a single call
site**:

```python
# memory/compaction.py — pure, no BaseAgent state
def compact(
    messages: list[Message],
    *,
    max_tokens: int,
    keep_recent: int,
    pinned: Callable[[Message], bool],       # load-bearing messages survive
    summarize: Callable[[list[Message]], str],  # injected, not self.complete
) -> list[Message]: ...
```

`BaseAgent.complete()` calls it in exactly one place. E4 later moves the *call site* into
the compiler (`CompactionTransform.apply` delegates to `compact`); the function itself is
never touched. "Pure function over messages" stays correct regardless of what E4's compiler
turns out to look like, so no V3 abstraction has to be guessed at during V2.

Three constraints S5 must satisfy **for its own correctness**, independent of V3:

1. **No recursion.** The summarization LLM call must go through `self.llm.complete`, not
   `self.complete` — the latter re-enters compaction unboundedly. Usage is hand-accrued into
   `_active_ctx`, the pattern `_maybe_reflect` already uses (`base_agent.py:184-189`).
2. **Load-bearing content is pinned.** Compaction must never drop the recall system message
   or the task. Recall-as-data is a security contract (V2 S2a); silently compacting it away
   degrades the anti-poisoning story with no signal. Hence the explicit `pinned` predicate.
3. **Injected `summarize` means zero-LLM unit tests.** Compaction logic is testable with a
   stub summarizer — no `MockLLMProvider` wiring needed for the boundary cases.

---

## 2. Motivation — the measured problem

`BaseAgent.run` (`src/lottie/core/base_agent.py:231`) is a hand-sequenced 11-step pipeline:

```
check_input → policy.check → cost.reserve → load_recall → depth.set → capability.set
  → super().run (otel span + metrics + _execute)
  → _verify → check_output → _maybe_reflect
  → [finally] write_audit → cost.settle
```

Every V1 and V2 slice appended a step to this method. Consequences today:

1. **The kernel knows every subsystem.** `core/base_agent.py` imports `governance`,
   `memory`, `llm`, `security`. Dependency direction is inverted from what the
   architecture intends.
2. **Duplicated execution paths.** `run_stream` (`:268`) re-implements a subset of the same
   sequence. The methodology explicitly forbids this.
3. **A four-file tax per concern.** Adding one cross-cutting concern requires editing
   `BaseAgent.__init__` (field), `BaseAgent.run` (correctly ordered call),
   `project/discovery.py:instantiate_agent` (wiring), and `AgentConfig` (schema).
4. **Best-effort is a convention, not a guarantee.** `_write_audit`, `_write_block`, and the
   OTel calls each hand-roll `try/except → warnings.warn`. Nothing structural stops a future
   observer from breaking a run.
5. **No extension point.** A third party cannot add a concern at all.

### 2.1 Existing → V3 module map

| V3 module | Today | Gap |
|---|---|---|
| **Policy Engine** | `governance/policy.py` — `PolicyGate`, precedence deny>escalate>allow | Exists. Becomes a registered module (E2 S3). |
| **Memory Harness** | `memory/` — store, gateway, recall, reflect (V2 S0–S2b) | Exists. Becomes registered modules (E2 S5). |
| **Execution Planner** | `mesh/router.py` + `MeshEngine` (Local/LangGraph) | Partial — routing exists, no `Plan` object; single-agent runs have no planner. → E6 |
| **Provider Router** | `llm/__init__.py:build_provider` — 6 lines, always `LiteLLMProvider` | No routing, fallback, or cost policy. → E5 |
| **Module Orchestrator** | `project/discovery.py:instantiate_agent` — 80 lines, 7 hardcoded `set_*` | Closed to extension. → E3 |
| **Context Compiler** | `self._recall_prefix` string prepended inside `complete()` (`:383`) | No compiler; assembly is ad-hoc. → E4 |
| **Event Runtime** | none | → E1 |
| **Lifecycle Hooks** | none | → E1 |
| **Plugin SDK** | none | → E7 |

Policy Engine and Memory Harness are **not epics** — they already exist and become
consumers of the kernel in E2.

---

## 3. Decisions (settled in brainstorming)

| # | Decision | Rationale |
|---|---|---|
| D1 | **V2 finishes and tags before V3 starts.** | Avoids two half-built epics colliding on `core/base_agent.py`. |
| D2 | **The 9-module tree is the real V3 target**, mapped against what already exists (§2.1). | Four of nine already exist under other names; V3 is scoped to the genuine gaps. |
| D3 | **Strangler compat contract.** `BaseAgent.run` keeps its exact signature and observable behavior; internally it becomes "execute a compiled pipeline". | The existing ~950 (+V2) tests become the correctness proof. Every slice stays independently reviewable. |
| D4 | **Two kernel primitives: an abort-capable middleware chain + a never-aborting event stream.** | Matches the two semantics already present in the code (fail-closed gates vs best-effort observers) and maps 1:1 onto "Lifecycle Hooks" + "Event Runtime". A pure event bus cannot express `finally`-scoped concerns (cost `settle`, ContextVar reset) without leaking them. |
| D5 | **Incremental minors.** E1+E2+E3+E8 = v3.0.0; E4 = 3.1, E5 = 3.2, E6 = 3.3, E7 = 3.4. | Each minor is independently useful, releasable, and lab-validated. Plugin SDK ships last so the public interface is frozen only after internal modules prove it. |
| D6 | **Events carry scalars and sha256 hashes only — never raw content.** | The bus is a new exfiltration surface, and E7 opens it to third parties. Extends the existing hash-only audit discipline. |

---

## 4. Architecture — E1, the kernel

New package `src/lottie/runtime/`. **Zero runtime dependencies** beyond stdlib + pydantic.

### 4.1 `runtime/context.py`

```python
@dataclass
class ExecutionContext:
    runnable: str                 # agent/skill name
    kind: Kind                    # "agent" | "skill"
    input: BaseModel              # frozen
    usage: RunContext             # EXISTING core/metrics accumulator, reused as-is
    run_id: str
    state: dict[str, object]      # module-private scratch, key-namespaced by module name
```

`core/metrics.RunContext` (the token accumulator) is **not renamed**. `ExecutionContext`
holds it. Zero churn on metrics, benchmark, or anything reading `last_metrics`.

### 4.2 `runtime/middleware.py`

```python
type Next = Callable[[ExecutionContext], Any]

class Middleware(Protocol):
    name: str
    order: int
    def __call__(self, ctx: ExecutionContext, nxt: Next) -> Any: ...
```

The chain is heterogeneous, so `Next` carries **one documented `Any` seam** — justified in
the docstring per rule 6. `Pipeline[InputT, OutputT]` stays fully typed at its boundary, so
no `Any` escapes to callers.

### 4.3 `runtime/events.py`

Frozen pydantic event models — `RunStarted`, `RunCompleted`, `RunFailed`, `RunBlocked` —
plus:

```python
class Subscriber(Protocol):
    name: str
    def on_event(self, event: RunEvent) -> None: ...

class EventBus:
    def subscribe(self, sub: Subscriber) -> None: ...
    def emit(self, event: RunEvent) -> None: ...   # each call wrapped
```

Sync, in-process, registration-ordered. **Every subscriber invocation is wrapped**; an
exception becomes `warnings.warn` and never propagates. Fail-open is structural.

Per **D6**, event models carry scalars and hashes only. A contract test asserts every field
of every event model is a scalar or a hash — a subscriber that needs raw content must be a
middleware instead, where trust is explicit.

### 4.4 `runtime/pipeline.py`

Sorts middleware by `order`, executes onion-style, and emits lifecycle events **from the
innermost frame** (load-bearing — see §4.6).

### 4.5 Order table — the correctness proof

Onion post-order is the reverse of pre-order. Reproducing today's sequence faithfully
therefore forces one split: the current single security gate becomes `SecurityInput` +
`SecurityOutput` (better single-responsibility regardless).

| order | middleware | pre-`nxt` | post-`nxt` |
|---|---|---|---|
| 10 | `SecurityInput` | `check_input` | — |
| 20 | `Policy` | `policy.check()` | — |
| 30 | `Cost` | `reserve()` | `settle()` (in `finally`) |
| 40 | `Depth` | `_audit_depth.set` | reset (in `finally`) |
| 50 | `Capability` | `_active_capabilities.set` | reset (in `finally`) |
| 60 | `Recall` | `_load_recall` | clear prefix |
| 70 | `Reflect` | — | `_maybe_reflect` |
| 75 | `SecurityOutput` | — | `check_output` |
| 80 | `Verify` | — | `_verify` |
| — | **core frame** | `run_span` + `_execute` + `_record`, emit `RunCompleted` | |

Unrolled:

```
check_input → policy → reserve → depth → cap → recall
  → _execute → [emit RunCompleted → AuditSubscriber, OtelSubscriber, BenchmarkSubscriber]
  → verify → check_output → reflect → clear → cap reset → depth reset → settle
```

Identical to `base_agent.py:231-266`.

### 4.6 Why the event fires from the innermost frame

`base_agent.py:259` documents a load-bearing ordering: **cost settles _after_ audit records
the real cost.** Emitting `RunCompleted` from the innermost frame means `AuditSubscriber`
runs before `Cost`'s `finally`, preserving that invariant exactly.

### 4.7 Mount point

`Pipeline` mounts on `InstrumentedRunnable`, **not** on `BaseAgent`. `BaseSkill` therefore
gets the same kernel with a smaller default chain (capability + metrics). One execution
path for agents and skills — which is the point.

`run()` and `run_stream()` both become a single call over the same chain. The duplicated
execution path is deleted, not maintained.

---

## 5. E2 — Strangler migration

Migrating one concern at a time would leave `run()` in a hybrid half-pipeline /
half-hardcoded state that cannot be verified. Instead: **sequencing moves first, ownership
moves second.**

**S2 — Pipeline swap-in (the risky slice).** `run`/`run_stream` delegate to `Pipeline`.
Every middleware is a *thin adapter* over code that already exists — `PolicyMiddleware`
calls `self._policy.check()` and nothing else. Zero logic relocated, zero fields removed.
Only the sequencer changes. Adds recording-middleware / recording-bus tests that assert
**call order explicitly**, so the invariant is pinned rather than incidental.

**S3 — Fail-closed modules take ownership.** `security`, `policy`, `cost`, `capability`
middleware move into their owning subsystems. `BaseAgent` sheds `_security`, `_policy`,
`_cost`, `_capabilities`; the `set_*` methods remain as deprecated shims.

**S4 — Observers become subscribers.** `audit`, `otel`, `benchmark` stop being inline calls.
`_write_audit`, `_write_block`, and `span_set_*` become `EventBus` subscribers.
`BaseAgent` sheds `_audit`. Best-effort stops being a convention repeated in three places.

**S5 — Memory + verify modules.** `recall`, `reflect`, `verify`. `_maybe_reflect`
(48 lines inside `BaseAgent` today) becomes `ReflectMiddleware` depending on an `LLMCaller`
Protocol rather than on `BaseAgent` — depend on abstractions.

### 5.1 The dependency reversal

Middleware live in their **owning subsystem** (`governance/middleware.py`,
`security/middleware.py`, `memory/middleware.py`), each importing only `lottie.runtime`
protocols:

```
today:  core ──imports──> governance, memory, llm, security
after:  core ──> runtime <── governance, memory, security, llm
```

`core/base_agent.py` drops from 6 subsystem imports to 1. `run()` goes from a 36-line
hand-sequenced method to one line. **These two numbers are the pass/fail criteria for V3.**

---

## 6. E3 — Module Orchestrator + config surface

`instantiate_agent` becomes registry-driven composition.

```python
type Mountable = Middleware | Subscriber

@dataclass(frozen=True)
class Deps:
    """Constructor dependencies a module may need — injected, never imported.
    Populated by the orchestrator: llm, memory client, audit logger, event bus."""

class ModuleFactory(Protocol):
    name: str
    order: int
    def build(self, cfg: AgentConfig, root: Path, deps: Deps) -> Mountable | None: ...
        # None == disabled by config; never mounted, zero cost
```

`Deps` is the dependency-injection carrier — a module receives what it needs rather than
importing it, which is what keeps the subsystem→`runtime` edge one-directional (§5.1).

`instantiate_agent` collapses to: resolve registry → call each factory → mount what is
returned. Adding a cross-cutting concern goes from **editing 4 files to registering 1
factory**.

**Order conflicts fail at registration, not at runtime.** Two modules claiming order 30
raise on startup — fail-closed, and a plugin cannot silently reorder a security gate.

### 6.1 Config — no second way to configure built-ins

Existing keys (`budget_usd`, `max_run_usd`, `max_run_tokens`, `max_turns`, `capabilities`,
`policies`, `memory.*`, `chat`, `workers`) keep working verbatim. **Zero migration for every
existing project.** The new optional block covers enable/disable and third-party config
only:

```yaml
modules:
  audit:   { enabled: false }        # was LOTTIE_DISABLE_AUDIT
  otel:    { enabled: false }        # was LOTTIE_DISABLE_OTEL
  my_tracer: { enabled: true, endpoint: "..." }   # third-party, E7
```

Side benefit: three ad-hoc env vars (`LOTTIE_DISABLE_AUDIT`, `LOTTIE_DISABLE_BENCHMARKS`,
`LOTTIE_DISABLE_OTEL`) unify into one mechanism. The env vars remain as overrides so CI and
the test suite are unaffected.

### 6.2 CLI

- **`lottie modules [<agent>]`** — mounted modules with order, source (builtin/plugin), and
  enabled state. This is the primary lab-validation surface for E1–E3: the chain becomes
  visible rather than inferred.
- **`lottie doctor`** gains checks for order conflicts, unknown module names in config, and
  plugins that failed to load.

---

## 7. Slice plan — v3.0.0

**One slice = one PR = one lab round. A slice does not merge until its round is green in
`cdiaz19/lottie-lab`.** Same cadence as V1 (R15–R21) and V2 (R22–R27). V3 starts at R28.

| Slice | Epic | Delivers | Gated by | Lab round |
|---|---|---|---|---|
| **S1 Kernel** | E1 | `runtime/` — `ExecutionContext`, `Middleware`, `EventBus`, `Pipeline`, `ModuleRegistry`. Nothing consumes it yet; zero behavior change. Ships the perf microbenchmark + overhead budget. | — | **R28** |
| **S2 Pipeline swap-in** | E2 | `run`/`run_stream` execute the chain; middleware are thin adapters over existing code. Explicit call-order tests. | S1 | **R29** |
| **S3 Fail-closed modules** | E2 | security / policy / cost / capability own their middleware; `BaseAgent` sheds 4 fields; `set_*` deprecated shims. | S2 | **R30** |
| **S4 Observers** | E2 | audit / otel / benchmark become `EventBus` subscribers; `BaseAgent` sheds `_audit`. | S2 | **R31** |
| **S5 Memory + verify** | E2 | recall / reflect / verify middleware; `ReflectMiddleware` on an `LLMCaller` Protocol. | S2 | **R32** |
| **S6 Orchestrator** | E3 | registry-driven `instantiate_agent`, `modules:` config block, `lottie modules`, doctor checks. | S3–S5 | **R33** |
| **S7 Release + red-team** | E8 | bump 3.0.0, CHANGELOG, `runtime/ARCHITECTURE.md`, migration notes, perf report, **tag v3.0.0**. | all | **R34** (full regression) |

### 7.1 Red-team scenarios (R34, mandatory)

- Malicious subscriber raises → run completes unaffected; warning emitted.
- Subscriber attempts content exfiltration → blocked by D6 (no raw content on the bus);
  contract test proves it.
- Plugin declares a conflicting order → registration fails at startup.
- Plugin attempts to displace a security middleware → rejected.

---

## 8. Post-3.0 minors — scoped, not designed

Each gets its own architecture review → design proposal → spec → plan, per
`docs/METHODOLOGY.md`. Scope is fixed here; design is not.

**E4 Context Compiler → v3.1.** `ContextSource` Protocol + `ContextCompiler`: ordered,
typed, token-budgeted message assembly with a deterministic drop policy at ceiling. Sources:
system prompt · knowledge graph · recalled memory (as data) · skill descriptions · task ·
compaction summary. **Absorbs V2 S5's context compaction as a compiler transform** (see §1).

**E5 Provider Router → v3.2.** Replaces the 6-line `build_provider`. Routing rules, fallback
chains on error/ratelimit, cost/latency-aware selection — all behind the unchanged
`LLMProvider` interface, so no agent changes. Emits `provider_selected` /
`fallback_triggered` onto the bus, making observability free. Routing policy is config,
never code; the runtime stays provider-agnostic. Lab R6 already hit this wall
("`build_provider` always returns `LiteLLMProvider`, so the CLI can't script the
supervisor").

**E6 Execution Planner → v3.3.** A typed `Plan` DAG that both single-agent (a 1-node plan)
and mesh compile to. Unlocks `lottie plan <agent>` dry-run/explain, pre-execution cost
estimation, deterministic replay. **Highest-risk epic** — it touches mesh, the most complex
subsystem, and is the most likely to be re-scoped or split after its own review.

**E7 Plugin SDK → v3.4.** Public versioned extension API over entry points
(`lottie.modules`): third-party middleware, subscribers, context sources, providers. Last on
purpose.

> **Documented security limit, not to be glossed:** plugins load in-process with full trust.
> The capability gate constrains *agents calling skills*; it does not sandbox a malicious
> plugin. E7 ships an explicit trust statement, `lottie doctor` plugin listing, and
> **opt-in-by-explicit-name loading — never auto-discovery of anything installed.**

---

## 9. Cross-cutting invariants (every slice)

- **Rule 7b gate:** `uv run ruff check .`, `uv run mypy --strict src`, `uv run pytest -q`
  under `uv sync --dev --all-extras`, green before push; `gh pr checks` green before
  squash-merge.
- **One slice = one PR = one lab round.** Squash-merge. **Branch before editing** (V1 S7
  lesson).
- **Existing tests are the regression suite and are never weakened to make a slice pass.**
  A test that must change *is* a compatibility break and requires explicit sign-off in the
  PR. This is the strangler's only real safety rail.
- **Zero new runtime dependencies** in the kernel — stdlib + pydantic. Base install stays
  lean.
- **Gates fail closed (middleware); observers fail open (subscribers).** Structural, never
  conventional.
- **Rules 8 / 9 / 11 / 13b unchanged in effect.** The extraction moves *where* they run,
  never *whether*.
- **Events are hash-only (D6).** Enforced by contract test.
- **Performance is a gate.** S1 ships a microbenchmark and sets the per-run overhead budget;
  every later slice checks against it.

### 9.1 New CLAUDE.md rules

- **S2:** **Gates are middleware (fail-closed); observers are subscribers (fail-open).**
  Never mix the two. (Added when the swap-in establishes both roles.)
- **S4:** **Events carry scalars and hashes only**, never raw input/output content.
  (Added when observers become subscribers and the bus starts carrying run data.)
- **S6:** Cross-cutting concerns are **modules registered with the runtime registry** —
  never inline steps in `run()`. (Added when the registry becomes the wiring path.)

---

## 10. Definition of Done — v3.0.0

| Methodology check | Concrete criterion |
|---|---|
| Architecture reviewed | E1/E2/E3 designs approved before each implementation round |
| Implementation completed | `run()` is one line; `run_stream` shares the same chain |
| Fully tested | all pre-V3 tests green **unchanged**, plus kernel unit / order / failure suites |
| Lab scenarios | R28–R34 green, including the R34 red-team set (§7.1) |
| Documentation updated | `runtime/ARCHITECTURE.md`, per-module docs, migration notes |
| Performance evaluated | measured overhead within the budget set in S1 |
| Backward compatibility | **zero changes required** in `agents/`, `skills/`, or lab R1–R27 |
| Public APIs reviewed | `set_*` deprecated-not-removed; `run` signature identical |
| Ready for enterprise usage | `lottie modules` + `doctor` module checks ship |
| **Architectural outcome** | `core/base_agent.py`: 6 subsystem imports → 1 |
| **Release** | `v3.0.0` tagged |
