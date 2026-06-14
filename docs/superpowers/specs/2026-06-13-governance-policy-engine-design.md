# Governance — Policy Engine (slice 2) — Design

> Declarative allow/deny/escalate governance over an agent's declared **capabilities**, evaluated
> at the run chokepoint. A denied/escalated run is blocked with a typed error and recorded in the
> audit trail with a distinct status.

- **Date:** 2026-06-13
- **Phase:** Governance, slice 2. **Stacked on `feat/governance-audit-trail` (PR #11).** Merge order:
  #11 (audit) → then this. Reuses the audit trail built in slice 1.
- **Branch:** `feat/governance-policy-engine` (off `feat/governance-audit-trail`).

---

## 1. Goal & scope

Agents declare `policies: [base]` and `capabilities: [...]` in `config.yaml`, and projects ship
`policies/<name>.yaml` with `allow`/`deny`/`escalate` lists — but nothing loads or enforces them. Build
a policy engine that, at the run chokepoint, evaluates an agent's **declared capabilities** against
the merged rules of its declared policies, and blocks fail-closed: a `deny` match → `PolicyDenied`, an
`escalate` match → `PolicyEscalation` (distinct), with the blocked attempt logged to the audit trail.

**Decisions (locked in brainstorming):** rules target **declared capabilities** · `escalate` **blocks**
(distinct from deny) · branch **stacked on audit** · enforcement on the **top-level** `BaseAgent.run`.

## 2. Rule model & loading — `src/lottie/governance/policy.py`

A policy file (`policies/<name>.yaml`) already has the shape (from the `lottie init` template):
```yaml
name: base
allow: []     # capability names; non-empty ⇒ whitelist (anything not listed is denied)
deny: []      # capability names that are forbidden
escalate: []  # capability names that require human approval (blocked, distinct from deny)
```

```python
class Policy(BaseModel):
    name: str = ""
    allow: list[str] = []
    deny: list[str] = []
    escalate: list[str] = []


def load_policy(root: Path, name: str) -> Policy:
    """Read root/policies/<name>.yaml. Empty/0-byte file → empty Policy(name).
    A declared-but-missing file raises PolicyConfigError (governance config must be valid)."""
```

- **Empty file** (the repo's current `base.yaml` is 0 bytes) → `yaml.safe_load` returns `None` →
  `Policy(name=name)` with all-empty lists ⇒ no rules ⇒ nothing blocked. **Backward-compatible**:
  existing agents declaring `policies: [base]` keep running unchanged.
- **Missing file** → `PolicyConfigError` (a declared policy that doesn't exist is an error).

## 3. `PolicyGate` + evaluation

```python
class PolicyGate:
    def __init__(self, capabilities: list[str], allow: set[str],
                 deny: set[str], escalate: set[str]) -> None: ...
    def check(self) -> None:
        """Raise PolicyDenied / PolicyEscalation on a violation; else return."""

class NullPolicyGate(PolicyGate):
    def __init__(self) -> None:           # no-arg: the BaseAgent default
        super().__init__([], set(), set(), set())
    def check(self) -> None:              # no-op (no policies / direct construction)
        return


def build_policy_gate(root: Path, config: AgentConfig) -> PolicyGate:
    """Merge every declared policy file (union of allow/deny/escalate) and bind
    config.capabilities. Returns NullPolicyGate when config has no policies."""
```

**Evaluation (`check`)** over the bound declared capabilities, precedence **deny > escalate >
allow-whitelist**, deterministic (sort capabilities; first hit wins):
1. any capability in `deny` → `PolicyDenied(capability, policy-context)`.
2. else any capability in `escalate` → `PolicyEscalation(capability)`.
3. else if `allow` is non-empty and some capability ∉ `allow` → `PolicyDenied(capability)` (not
   whitelisted).
4. else return (allowed).

Merging multiple policy files: `deny`/`escalate`/`allow` are unioned across files. Most projects
declare one policy (`base`); union keeps multi-policy behaviour predictable. Error messages name the
offending capability and are safe to surface (capability names are not secrets).

## 4. Errors — `src/lottie/governance/policy.py`

```python
class PolicyViolation(Exception): ...        # base
class PolicyDenied(PolicyViolation): ...      # deny match / not whitelisted
class PolicyEscalation(PolicyViolation): ...  # escalate match — needs human approval
class PolicyConfigError(Exception): ...       # declared policy file missing / malformed
```

These live in governance (not serve). They propagate out of `BaseAgent.run`. On the serve path they
are currently wrapped as `AgentExecutionError` (cause preserved) — a distinct serve mapping is a
follow-up, out of scope here.

## 5. Enforcement + audit integration — `BaseAgent` (`src/lottie/core/base_agent.py`)

`BaseAgent` gains `self._policy: PolicyGate = NullPolicyGate()` (default, set in `__init__`) and a
`set_policy(self, gate: PolicyGate) -> None` setter. The existing audited `run` override gets a policy
pre-check at the very top:

```python
    def run(self, data: InputT) -> OutputT:
        try:
            self._policy.check()
        except PolicyViolation as exc:
            self._write_policy_block(data, exc)   # best-effort distinct audit record
            raise
        # ... existing depth + super().run + _write_audit (unchanged) ...
```

`_write_policy_block(data, exc)` builds an `AuditRecord` with `status="denied"` (for `PolicyDenied`)
or `"escalated"` (for `PolicyEscalation`), `root=True` (the check runs before any nesting — a blocked
run is a top-level event), `input_sha256=hash_model(data)`, `output_sha256=None`, zeroed
tokens/cost/latency, `ts=datetime.now(UTC).isoformat()`, `error=str(exc)`; then `self._audit.log(...)`.
It is wrapped in the same best-effort `try/except → warnings.warn` so auditing can never convert a
policy block into a different failure or suppress the `PolicyViolation`.

The check runs **before** the instrumented `super().run`, so a blocked run never reaches `_execute`
(input never reaches the agent — matches the spec's "policy check before agent receives input"). The
`AuditRecord.status` vocabulary widens from `ok|error` to `ok|error|denied|escalated` (status is a
free-form `str`; no schema change required).

## 6. Wiring — `instantiate_agent` (`src/lottie/project/discovery.py`)

`instantiate_agent(agent_cls, *, llm, root, config, ...)` is the single canonical construction point
for both `lottie run` and `AgentService`. After building the agent (via `from_project` or the plain
constructor), attach the gate:

```python
    agent = (... from_project(...) or agent_cls(llm=llm, ...))
    agent.set_policy(build_policy_gate(root, config))
    return agent
```

Post-construct attach (not a constructor kwarg) because `from_project` signatures are fixed and can't
forward a policy. This enforces policy on the **top-level** agent/mesh for CLI + serve. Directly
constructed agents (unit tests, in-process callers) keep `NullPolicyGate` — no behaviour change.
Mesh **workers** are not individually policy-checked this slice; they are already governed by the
supervisor router's declared `workers:` allow-set.

## 7. Testing

- **load_policy** (unit): a real `policies/x.yaml` with deny/allow/escalate → populated `Policy`;
  empty/0-byte file → empty `Policy`; missing file → `PolicyConfigError`.
- **PolicyGate.check** (unit): clean caps → no raise; a capability in `deny` → `PolicyDenied`; in
  `escalate` (and not denied) → `PolicyEscalation`; deny beats escalate when a cap is in both;
  non-empty `allow` with an unlisted cap → `PolicyDenied`; `allow` superset → ok; `NullPolicyGate` →
  never raises.
- **build_policy_gate** (unit): no policies → `NullPolicyGate`; merges two policy files (union).
- **BaseAgent enforcement** (integration, injected audit logger + `set_policy`): an agent whose
  capability is denied → `run` raises `PolicyDenied`, `_execute` never ran (assert via a side-effect
  flag), and ONE audit record with `status="denied"` exists; escalate → `PolicyEscalation` +
  `status="escalated"`; `NullPolicyGate` (default) → run proceeds, normal `ok` record; a failing
  `set_policy` audit write never masks the `PolicyViolation`.
- **instantiate_agent** (integration): a scaffolded project whose `base.yaml` denies the echo agent's
  capability → `instantiate_agent(...).run(...)` raises `PolicyDenied`; empty `base.yaml` → runs
  normally.
- **Full gate:** `pytest -q`, `mypy --strict src`, `ruff check` green; existing suite unaffected
  (empty `base.yaml` everywhere ⇒ no new blocks).

## 8. Out of scope (YAGNI / later)

- Action/resource policies (file-write, shell, network) — needs an interceptable action layer.
- Runtime per-skill-**call** enforcement (rule 11 `CapabilityEnforcerSkill`) — this slice governs
  *declared* capabilities, not each call.
- Interactive/HITL approval for `escalate` (it blocks for now).
- Distinct serve-path error mapping for `PolicyViolation` (wraps as `AgentExecutionError` for now).
- Per-mesh-worker policy; a `lottie policy` CLI (rules are visible via `lottie inspect`).
- Cost-budget policies and OpenTelemetry — separate governance slices.

## 9. Definition of done

Declared capabilities are evaluated against merged declared policies at `BaseAgent.run`; `deny` →
`PolicyDenied`, `escalate` → `PolicyEscalation`, `allow`-whitelist enforced; blocked runs are audited
with `status="denied"/"escalated"` and never reach `_execute`; empty/absent rules block nothing
(backward-compatible); `instantiate_agent` attaches the gate for CLI + serve; `NullPolicyGate` default
keeps direct construction unchanged; `pytest`/`mypy --strict src`/`ruff` green. Commit on the feature
branch; do not push until the user approves.
