# S2 Design — Security gate on the BaseAgent/CLI path

- **Date:** 2026-07-08
- **Epic:** v1.0.0 hardening — slice S2. Lab round R16.
- **Closes:** BaseAgent/CLI-path security-gate enforcement (context.md deferral).

## 1. Goal

Today only the **serve** path runs the input/output security gate (rules 8 & 9). `lottie run`
(`cli/run.py:46` → `agent.run(data)`) and any instantiate-built agent used off the serve path
are **ungated**. Make `BaseAgent.run` the gating chokepoint so the CLI passes through the same
input-sanitize + injection-scan / output-validate + secret-scan gate — fail-closed — **without
double-gating the serve path or regressing any serve behaviour.**

## 2. Grounding (import graph)

- `serve → project` (one-way). `project` imports **neither** serve nor security. `cli → serve` OK.
- The gate (`serve/security.py:SecurityGate`) runs security skills and raises
  `serve.errors.{Input,Output}SecurityViolation` (subclasses of `ServeError`). It must stay in
  `serve/` — `security/` can't import `serve.errors` (cycle) and the violations must remain
  `ServeError` subclasses (MCP + mesh CLI catch `ServeError` generically).
- Therefore `core`/`project` cannot import the concrete gate. Use **dependency injection**: a
  core-free Protocol, with the concrete gate passed in by a caller that may import serve (`cli`).

## 3. Design

### 3.1 Core (`core/security_gate.py`, core-free)
- `SecurityGateProtocol` (structural): `check_input(text: str) -> None`, `check_output(text: str) -> None`.
- `NullSecurityGate`: both no-ops. The BaseAgent default.

`serve.security.SecurityGate` already satisfies the Protocol structurally — no change to it.

### 3.2 BaseAgent (`core/base_agent.py`)
- `self._security: SecurityGateProtocol = NullSecurityGate()` + `set_security_gate(gate)`.
- In `run()`:
  - **input gate first** — `self._security.check_input(data.model_dump_json())` at the very top
    (rule 8: external input screened before anything else), before `_pre_run_gates`.
  - run as today.
  - **output gate** — `self._security.check_output(output.model_dump_json())` after
    `super().run()` returns, before returning to the caller (inside the audit scope, so the
    executed run is still audited; the withheld output then propagates).
- BaseAgent never imports the violation types — it calls the injected gate and lets whatever it
  raises propagate. `run_stream` is **not** gated here (only serve streams, and serve owns the
  incremental `StreamingSecretGate`); documented.

### 3.3 Attachment (`project/discovery.instantiate_agent`)
- New param `security_gate: SecurityGateProtocol | None = None`; when provided,
  `agent.set_security_gate(security_gate)`. `project` imports only the core Protocol (no serve).

### 3.4 Callers
- `cli/run.py`: `instantiate_agent(..., security_gate=SecurityGate())` (import from `serve.security`)
  → `lottie run` now gates. Catch `SecurityViolation` around `agent.run` → clean refusal (no
  payload echo), distinct exit; do not surface a raw traceback.
- `cli/mesh.py` run path + `benchmark` run path: same `security_gate=SecurityGate()` for uniform
  CLI coverage.
- `serve/service.py`: **unchanged** — calls `instantiate_agent` WITHOUT `security_gate`, so its
  agents keep `NullSecurityGate` and serve's existing external gate stays authoritative. No
  double-gate, no serve behaviour change.

### 3.5 Double-gate avoidance
Serve gates externally (in `AgentService`) and builds agents with no injected gate → BaseAgent
gate is Null on the serve path. CLI builds with the gate → BaseAgent gates. One `SecurityGate`
class, gated exactly once on every path.

## 4. Tests (grow from 895)
- Core: `NullSecurityGate` no-ops; a BaseAgent with a stub gate blocks a bad input before
  `_execute`; withholds a bad output after `_execute` (run still audited); default construction
  is ungated (back-compat).
- instantiate: `security_gate=` attaches; omitted → Null.
- CLI: `lottie run` with a poisoned input → refusal exit, no payload echo; clean input → output.
- Serve regression: existing serve/HTTP/stream tests stay green (no double-gate); an explicit
  test that a serve run is NOT gated twice (gate invoked once).

## 5. Files
- New: `core/security_gate.py` + tests.
- Edit: `core/base_agent.py`, `core/__init__.py` (export), `project/discovery.py`, `cli/run.py`,
  `cli/mesh.py`, `cli/benchmark.py` (whichever build+run), and the CLI error handling.

## 6. Risks / deferred
- Security-blocked CLI runs are **not** audited in this slice (policy/cost blocks are). The gate
  raises a serve error; auditing it from core would need a broad catch. Deferred; noted.
- `run_stream` security gating stays serve-only (incremental gate). Unchanged.
- Input text gated is `data.model_dump_json()` (typed) vs serve's raw `json.dumps(payload)` —
  equivalent JSON of the same input; acceptable.
