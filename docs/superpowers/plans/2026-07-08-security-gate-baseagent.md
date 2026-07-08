# S2 Plan — Security gate on BaseAgent/CLI path

Design: `2026-07-08-security-gate-baseagent-design.md`. TDD, mypy --strict + ruff clean per task. Baseline 895.

- **T1** `core/security_gate.py`: `SecurityGateProtocol` + `NullSecurityGate` (+ export from `core/__init__`). Test: Null no-ops; a stub gate raising on a marker string is honored.
- **T2** `BaseAgent`: `_security` + `set_security_gate`; `check_input(data.model_dump_json())` at top of `run` (before `_pre_run_gates`); `check_output(output.model_dump_json())` after `super().run()`. Tests: input-block before `_execute`; output-withhold after `_execute` (run audited); default ungated.
- **T3** `instantiate_agent(security_gate=None)` attaches when provided. Test: attaches / omitted→Null.
- **T4** CLI wiring: `cli/run.py` (+ `cli/mesh.py`, `benchmark` run paths) pass `security_gate=SecurityGate()`; catch `SecurityViolation` → clean refusal, no payload echo. Test: `lottie run` poisoned input → refusal; clean → output.
- **T5** Serve regression + no-double-gate: full suite green; assert a serve run invokes the gate exactly once (serve external gate; BaseAgent Null on serve path).
- **T6** Full gate (`uv sync --dev --all-extras`; ruff; mypy --strict src; pytest -q) + whole-diff review.

## Lab R16
Downstream: `lottie run` (or instantiate + run) with a poisoned input → refused fail-closed; a secret in output → withheld; a clean run → ok; serve path still single-gated. Validate locally.
