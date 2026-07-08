# S6 Design — Agentic-loop hygiene: max_turns + verify hook

- Date: 2026-07-08 · Epic: v1.0.0 S6 · Lab R20 · From Anthropic harness guidance. Keep minimal;
  defaults off/generous; typed Pydantic config.

## Goal
Two cheap safety rails on the agentic loop:
1. **max_turns** — cap the number of LLM completions in a single run (bounds a runaway tool loop
   by call COUNT, complementary to S3's max_run_tokens which bounds by token VOLUME).
2. **verify hook** — an optional post-`_execute` check an agent can override to reject a bad
   output BEFORE the run declares success (fail-closed).

## Design
### max_turns (`AgentConfig.max_turns: int | None`, None = unlimited)
- `RunContext.turns` counts completions. `BaseAgent.complete()` / `stream_complete()` increment
  it, then enforce: `turns > max_turns` -> `TurnLimitExceeded` (aborts before the next call).
- Attached via `instantiate_agent` -> `set_run_limits(..., max_turns=...)`.
- Mirrors the S3 token-cap plumbing exactly (same accrual sites), so no new call surface.

### verify hook (`BaseAgent._verify`)
- `def _verify(self, data: InputT, output: OutputT) -> None: ...` — default no-op. An agent
  overrides it to assert post-conditions; raising fails the run (fail-closed). Called in `run()`
  after `super().run()` (the `_execute` result) and BEFORE the output security gate, so a
  self-rejected output never even reaches the gate/caller. Enabling = overriding (no config).
- Not wired into `run_stream` (a stream has no single typed output to verify); documented.

`TurnLimitExceeded(RuntimeError)` beside `NotStreamable` in base_agent.

## Tests (grow from 938)
- max_turns: a 3-completion run with max_turns=2 -> TurnLimitExceeded on the 3rd; under/None ->
  clean. instantiate attaches it.
- verify: an agent whose `_verify` raises -> run fails, output withheld from caller, called after
  _execute; default no-op -> unaffected; `_verify` runs before the security output gate.

## Files
- core/metrics.py (RunContext.turns — done), core/base_agent.py (`_max_turns`, set_run_limits,
  count+enforce in complete/stream_complete, `_verify` + call in run), project/config.py
  (max_turns), project/discovery.py (attach). Tests alongside.

## Risks
- verify runs before the security output gate (agent semantics first, then rule-9 screen). Both
  can raise; documented. max_turns bounds completions, not wall-clock/tool-calls generally.
