# E5 — Provider Router

> Epic design. Target: **v3.2.0**. Date: 2026-08-11.
> Theme: make `providers.fallback` mean something, and give provider selection the same
> observability every other cross-cutting concern already has.
> Follows `2026-07-30-v3-runtime-kernel-design.md` §8 (E5).

---

## 1. Architecture analysis

`build_provider` is six lines:

```python
def build_provider(model: str) -> LLMProvider:
    return LiteLLMProvider(model)
```

### The headline finding

**`lottie.yaml` has declared `providers.fallback` since Phase 0. Nothing reads it.**

```python
class Providers(BaseModel):
    default: str
    fallback: str | None = None   # ← never consumed
```

A user who sets a fallback today gets nothing. That is worse than an absent feature: the
config *claims* a resilience property the runtime does not have, and the failure mode is
discovering it during an outage. Closing that gap is the point of E5; routing rules are
the smaller half.

### Supporting gaps

- **`LiteLLMProvider` has zero error handling.** Any litellm exception propagates raw, so
  E5 is defining error policy from scratch rather than adjusting it.
- **No observability on selection.** Nothing records which provider served a run, so a
  silent fallback would be invisible — the exact property that makes fallbacks dangerous.
- Lab R6 hit this years ago in project time: *"`build_provider` always returns
  `LiteLLMProvider`, so the CLI can't script the supervisor."*

### What is already in place

**Seven call sites already funnel through one function** — `cli/run`, `cli/distill`,
`cli/reflect`, `cli/create`, `serve/service`, `benchmark/runner`, `benchmark/learning`.
A real chokepoint; no plumbing needed.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Fallback triggers on transient/availability failures only** — rate limits, timeouts, 5xx, connection errors. Not bad requests, not auth errors, **not content-policy refusals**. | Security-specific to this codebase: Lottie is a governed framework with fail-closed gates, and falling back on a content-policy refusal would launder a provider's safety decision. A bad request or bad key fails identically on the fallback, so retrying only doubles the spend before failing anyway. |
| D2 | **The fallback emits `fallback_triggered` on the E1 event bus.** | A silent fallback is the dangerous kind. The bus already exists, subscribers are fail-open, and events are hash-only — so provenance costs nothing and cannot break a run. |
| D3 | **Audit records the model that actually served the response; cost accounting includes both attempts.** | The ledger should say what happened, not what was intended. Both attempts burned tokens, so both must count — under-counting a budget is the unsafe direction. |
| D4 | **`RoutedProvider` implements `LLMProvider`.** No agent, skill, or transport changes. | The interface is the seam that has held since Phase 0. Routing is a provider concern, not a runtime one. |

---

## 3. Architecture

### 3.1 `RoutedProvider(LLMProvider)`

Wraps an ordered chain of providers. `complete` tries each in turn, advancing **only** on
a transient failure; any other exception propagates immediately.

`model` reports the provider that served the **last** call, so `agent.provider` — which
feeds the audit record and the pipeline — tells the truth about what ran.

### 3.2 `is_transient(exc) -> bool`

The whole policy, in one testable predicate. Classifies by litellm's exception taxonomy
where available and by a conservative fallback shape otherwise. **Defaults to False**: an
unrecognised error is treated as *not* worth retrying, so a new exception type cannot
silently widen the fallback surface.

### 3.3 `build_provider(model, *, fallback=None, bus=None)`

Signature stays backward compatible — `build_provider(model)` returns a plain
`LiteLLMProvider` exactly as today. A fallback is only constructed when one is configured.

### 3.4 Streaming

`stream_complete` falls back **only before the first delta**. Once bytes have reached the
caller, switching providers mid-stream would produce a spliced response from two models.
After the first delta the error propagates, which the transport already handles.

---

## 4. Slice plan

| Slice | Delivers | Lab |
|---|---|---|
| **S1** | `is_transient`, `RoutedProvider`, `build_provider(fallback=...)`, the `fallback_triggered` event, wiring at the seven call sites | **R36** |
| **S2** | Release: bump 3.2.0, CHANGELOG, tag | full regression |

---

## 5. Invariants

- **`build_provider(model)` unchanged.** No fallback configured → no wrapper, no cost.
- **Never fall back on a content-policy refusal.** Asserted by a dedicated test.
- **Never fall back mid-stream.**
- **`is_transient` defaults to False** for unknown exceptions.
- Rule 7b gate per slice, one PR, one lab round.

---

## 6. Definition of Done (v3.2.0)

- `providers.fallback` is honoured — the config stops lying.
- A fallback is visible: `fallback_triggered` on the bus, and the audit names the model
  that actually served the run.
- A content-policy refusal is never retried elsewhere.
- R36 green, full regression green, `v3.2.0` tagged.
