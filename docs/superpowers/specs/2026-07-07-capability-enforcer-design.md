# S1 Design — CapabilityEnforcer (rule 11, per-skill-CALL)

- **Date:** 2026-07-07
- **Epic:** v1.0.0 hardening — slice S1 (see `2026-07-07-v1-hardening-epic-design.md`)
- **Lab round:** R15
- **Closes:** rule-11 deferral (from security-gate PR #10, context.md)

---

## 1. Goal

Enforce CLAUDE.md **rule 11** at runtime: *an agent may only call skills declared in its `config.yaml` `capabilities` list; an undeclared skill call is blocked, fail-closed.* Today `capabilities` only feeds the policy gate (per-declared, global) — there is **no per-CALL check**.

Non-goal: changing what `capabilities` means for the policy gate, or touching any agent's `_execute` code.

---

## 2. Grounding (from the seam investigation)

- Agents call skills as `skill.run(input)` on injected instances — `BaseSkill.run` = `InstrumentedRunnable.run` (`core/runnable.py:55`). Skills do not know their caller; no agent→skill registry.
- `AgentConfig.capabilities: list[str]` (`project/config.py:41`); `instantiate_agent` (`project/discovery.py:181`) already loads it and passes it to `build_policy_gate`.
- Existing gate pattern to mirror: `BaseAgent.set_policy` / `set_cost_gate` → checked in `_pre_run_gates` before `_execute`; attached by `instantiate_agent`. ContextVar precedent: `_audit_depth` (`base_agent.py:38`), which langgraph copies into mesh-worker threads.
- **Capability vocabulary is lowercase logical names, not class names:** `research/config.yaml` → `capabilities: [retrieval, summarizer]`, skills are `RetrievalSkill`/`SummarizerSkill`. `assistant/config.yaml` → `capabilities: []` and calls *agents* (workers), not skills.

---

## 3. Design

### 3.1 Components (`src/lottie/governance/capability.py`)

**Module placement — deviation from CLAUDE.md line 97 (which lists `security/capability_enforcer.py`), with rationale:** the gate is imported by `core/base_agent.py` AND `core/base_skill.py`. `security/` imports `core` (its skills extend `BaseSkill`), so `core → security` would create a **cycle**. `governance/` is core-free and already hosts the sibling gates (`policy.py`, `cost.py`) that `base_agent` imports acyclically. The capability gate is functionally identical to those (a declarative allow-list checked at the chokepoint), so it lives beside them: `governance/capability.py`. Rule 11's intent (block undeclared skill calls at runtime) is fully satisfied; only the file's module differs from the note. **Flagged for user sign-off at spec review.**

- `CapabilityDenied(Exception)` — governance-local exception (mirrors `PolicyViolation` / `BudgetExceeded`; NOT a subclass of `serve.errors.SecurityViolation`, which would cycle). Raised when an undeclared skill is invoked. Message names the skill + the calling agent; **no payload**.
- `CapabilityGate` — holds `allowed: frozenset[str]`; `check(capability: str) -> None` raises `CapabilityDenied` if `capability not in allowed`. Fail-closed.
- `NullCapabilityGate` — `check` is a no-op (allow all). Default for direct construction / unit tests.
- `build_capability_gate(*, capabilities: list[str]) -> CapabilityGate` — **whitelist-when-nonempty**: empty/absent list → `NullCapabilityGate`; non-empty → `CapabilityGate(frozenset(capabilities))`. Primitive args only (no core/project import → acyclic), mirroring `build_policy_gate`.

### 3.2 Skill identity — `capability_name`

Add to `BaseSkill`:
```python
capability_name: ClassVar[str | None] = None  # override to set explicitly
```
Resolution (a classmethod/property): explicit `capability_name` if set, else derive from class name — strip a trailing `"Skill"`, lowercase. `RetrievalSkill → "retrieval"`, `SummarizerSkill → "summarizer"`. Matches existing config vocabulary with **zero config edits**.

### 3.3 Enforcement point — _execute-scoped ContextVar

- Module-level `ContextVar[CapabilityGate]` (default `NullCapabilityGate()`), e.g. `_active_capabilities`.
- `BaseAgent` stores `self._capabilities: CapabilityGate = NullCapabilityGate()` + `set_capability_gate(gate)` (like `set_policy`).
- **`BaseAgent.run` / `run_stream`: set the ContextVar to `self._capabilities` scoped TIGHTLY around `super().run()` (the `_execute` call), reset in `finally`.** Only skills the agent invokes *inside its own `_execute`* see the active gate.
- **`BaseSkill.run` override:** read `_active_capabilities.get()`, call `.check(cls.resolved_capability_name())`, then `return super().run(data)`.

**Why _execute-scoped is load-bearing:** the security gate (S2) and knowledge-ingest invoke framework skills (`InputSanitizerSkill`, `SecretDetectionSkill`, `PromptInjectionScanSkill`) *outside* the `_execute` window (in the `BaseAgent.run` wrapper / standalone CLI). Those calls see the default `NullCapabilityGate` → allowed. Only agent-authored skill calls are enforced. This avoids the framework blocking its own security skills.

**Nesting:** agent A's `_execute` calls worker B (`B.run`) → B's `run` sets the ContextVar to B's gate around B's `_execute` → B's skill calls checked vs B's caps → reset restores A's context. ContextVar nests correctly; langgraph copies context into worker threads (same property proven by the audit-depth fix `e99d42e`).

### 3.4 Attachment — `instantiate_agent`

In `project/discovery.py`, alongside the existing `set_policy` / `set_cost_gate` calls, add:
```python
agent.set_capability_gate(build_capability_gate(capabilities=config.capabilities))
```
Live for the CLI (`lottie run`) and serve paths. Direct construction leaves `NullCapabilityGate` → back-compat.

### 3.5 Auditing

A blocked skill call raises `CapabilityDenied` from inside `_execute`, so it surfaces as the agent run's `error` (status `"error"`) via the existing `_write_audit` path — no new audit status needed. (Distinct `status="capability_denied"` is possible but deferred; the error string is explicit.) Rationale documented in the plan.

---

## 4. Test plan (grow from 871)

Unit (`security/tests/test_capability_enforcer.py`):
- `build_capability_gate`: empty → NullCapabilityGate; non-empty → strict gate.
- `CapabilityGate.check`: allowed passes; undeclared raises `CapabilityDenied`; message names skill + no payload.
- `capability_name` resolution: derive-from-class-name + explicit override.

Integration (`core/tests/`):
- Agent with `capabilities=[retrieval]` calling a `RetrievalSkill` → ok; calling `SummarizerSkill` → `CapabilityDenied`.
- Empty caps → any skill call passes (NullCapabilityGate).
- **Framework-exemption test:** an agent whose `_execute` runs, with a security gate wired to call `InputSanitizerSkill` outside `_execute`, does NOT block the framework skill even when the agent declares a narrow whitelist. (Guards the _execute-scoping invariant.)
- **Nesting test:** mesh worker with its own caps enforced independently of the supervisor's caps (real parallel path, mirrors `test_audit_root_parallel.py`).
- ContextVar reset: after a run, `_active_capabilities.get()` is the default again (no leak).

Regression: existing `research`/`assistant` integration suites stay green (research declares its two skills; assistant empty + agent-only calls).

---

## 5. Files touched

- New: `src/lottie/governance/capability.py`, `src/lottie/governance/tests/test_capability.py`.
- Edit: `src/lottie/core/base_skill.py` (`capability_name` ClassVar + resolution + `run` override), `src/lottie/core/base_agent.py` (`_active_capabilities` ContextVar, `set_capability_gate`, scope around `super().run()` in run + run_stream), `src/lottie/project/discovery.py` (attach the gate), `src/lottie/governance/__init__.py` (exports if present).
- The `_active_capabilities` ContextVar + gate types live in `governance/capability.py`; both `base_agent` and `base_skill` import them (`core → governance`, acyclic — already the pattern for policy/cost).

---

## 6. Risks / decisions

- **Cycle → module placement (RESOLVED):** gate lives in `governance/capability.py` (core-free, acyclic), not `security/`. `CapabilityDenied` is governance-local. `build_capability_gate` takes primitives. See §3.1.
- **Test dir:** unit tests move to `governance/tests/test_capability.py` (co-located with policy/cost tests), not `security/tests/`.
- **capability_name convention** could mismatch a skill whose config name isn't its de-Skill-ed class name → they override the ClassVar. Documented.
- **Deferred:** distinct `capability_denied` audit status; enforcing agent-to-agent (worker) calls (out of scope — rule 11 is skills; mesh SupervisorRouter already validates workers).
