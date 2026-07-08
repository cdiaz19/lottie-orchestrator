# S1 Plan — CapabilityEnforcer (rule 11)

Design: `docs/superpowers/specs/2026-07-07-capability-enforcer-design.md`. TDD (test first) per task. `mypy --strict` + `ruff` clean after each. Baseline 871 tests → grow.

## Task 1 — CapabilityGate primitives (`governance/capability.py`)
- Test first (`governance/tests/test_capability.py`): `CapabilityGate.check` allows declared, raises `CapabilityDenied` for undeclared (message names skill, no payload); `NullCapabilityGate.check` no-ops; `build_capability_gate` empty→Null, non-empty→strict frozenset.
- Impl: `CapabilityDenied(Exception)`, `CapabilityGate` (ABC or concrete + Null), `build_capability_gate(*, capabilities)`. Core-free, primitive args. Module-level `_active_capabilities: ContextVar[CapabilityGate]` default `NullCapabilityGate()` + `set_active_capabilities`/reset helpers (or expose the ContextVar for base_agent to set/reset directly).
- Verify acyclicity: `governance/capability.py` imports nothing from `core`/`project`/`serve`.

## Task 2 — `capability_name` on BaseSkill (`core/base_skill.py`)
- Test first: `capability_name` ClassVar default `None`; a resolver classmethod returns explicit value if set, else strip-trailing-`"Skill"`+lowercase (`RetrievalSkill`→`retrieval`, `SummarizerSkill`→`summarizer`, a no-`Skill`-suffix class → full lowercase). Explicit override wins.
- Impl: add `capability_name: ClassVar[str | None] = None` + `resolved_capability_name() -> str`.

## Task 3 — Enforce in `BaseSkill.run` (`core/base_skill.py`)
- Test first: with `_active_capabilities` set to a strict gate, `skill.run(input)` raises `CapabilityDenied` for an undeclared skill and passes for a declared one; with the default Null gate, always passes.
- Impl: override `BaseSkill.run(data)` → `_active_capabilities.get().check(self.resolved_capability_name())` then `return super().run(data)`. (Keeps instrumentation from `InstrumentedRunnable.run`.)

## Task 4 — Scope the ContextVar in `BaseAgent` (`core/base_agent.py`)
- Test first: agent with a strict gate calling a declared skill inside `_execute` → ok; undeclared → `CapabilityDenied` surfaced as the run's error (audit `status="error"`). After the run, `_active_capabilities.get()` is the default again (no leak). **Framework-exemption:** a skill invoked OUTSIDE `_execute` (simulate the S2 gate calling `InputSanitizerSkill` in the `run` wrapper) is NOT blocked even under a narrow whitelist.
- Impl: `self._capabilities: CapabilityGate = NullCapabilityGate()` + `set_capability_gate`. In `run`: set `_active_capabilities` to `self._capabilities` immediately around `super().run(data)` (line ~107), reset in `finally`. Same tight scoping in `run_stream` around the streamed body.

## Task 5 — Nesting / mesh (`core` or `mesh/tests`)
- Test first (mirror `mesh/tests/test_audit_root_parallel.py`): a mesh worker with its own capability gate enforces independently of the supervisor; a real parallel path preserves per-worker context (langgraph copies contextvars). Supervisor caps do not leak into workers and vice-versa.
- Impl: none expected beyond Task 4 (contextvar nesting). If parallel workers mis-share, fix by ensuring set/reset are balanced per run.

## Task 6 — Attach in `instantiate_agent` (`project/discovery.py`)
- Test first: an agent built via `instantiate_agent` with `capabilities=[retrieval]` blocks a `SummarizerSkill` call; with `capabilities=[]` blocks nothing.
- Impl: `agent.set_capability_gate(build_capability_gate(capabilities=config.capabilities))` beside the existing `set_policy`/`set_cost_gate`.

## Task 7 — CLAUDE.md note + regression + exports + full gate
- Update CLAUDE.md line ~97 (security module listing) + the `security/` line in the module map: note the capability gate lives in `governance/capability.py` (core-free) to avoid the `core↔security` cycle; rule 11 semantics unchanged.
- Run: `uv run ruff check .`, `uv run mypy --strict src`, `uv run pytest -q` with `uv sync --dev --all-extras`.
- Confirm `research` (declares retrieval+summarizer) and `assistant` (empty caps, agent-only calls) suites stay green.
- Export `CapabilityGate`/`NullCapabilityGate`/`build_capability_gate`/`CapabilityDenied` from `governance/__init__.py` if it re-exports siblings.
- Final self-review + whole-diff review (subagent) before PR.

## Out of scope (deferred, noted in journal)
- Distinct `capability_denied` audit status (surfaces as `error` now).
- Serve HTTP mapping of `CapabilityDenied` (bubbles as 500 config-error; distinct code deferred).
- Enforcing agent-to-agent/worker calls (rule 11 = skills; SupervisorRouter already validates workers).

## Lab R15 (after merge)
Round validating: declared skill ok; undeclared skill → CapabilityDenied; empty caps → no enforcement; framework security skills exempt; mesh worker independent caps. Driver mirrors prior rounds; validate locally (lab CI red on ORCH_REPO_TOKEN — known non-bug).
