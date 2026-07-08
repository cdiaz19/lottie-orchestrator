# Epic Design — Lottie v1.0.0 "Complete, Secured, Documented"

- **Date:** 2026-07-07
- **Status:** approved (in-session), execution starting
- **Scope thesis:** Harden, secure, and document what already exists. **No new capabilities.** Self-learning and A2A are V2, explicitly out of scope.
- **Companion repo:** `lottie-lab` — every code slice validated downstream by one lab round before the next slice starts.

---

## 1. Goal

Close 8 known open items so `v1.0.0` means "what exists is complete, secured, and documented," then tag it.

**Definition of done (V1):**
- All 8 items merged to `main`.
- Both mains (orch + lab) linear + synced.
- A full `lottie-lab` regression round green (locally).
- Journals current with reality.
- `v1.0.0` tag pushed.

---

## 2. Working conventions (non-negotiable)

- **1 slice = 1 PR**, squash-merge only, both mains linear.
- `mypy --strict` + `ruff` clean; local gate == CI (`uv sync --dev --all-extras`; watch `importorskip` false-greens).
- Tests for every slice; **count grows from the current baseline of 871** (was ~828 before streaming #19–23 landed).
- LLM calls **only** through `lottie.llm.LLMProvider`. Never a provider SDK.
- **Fail-closed** on anything security-adjacent.
- Each slice: spec + plan committed under `docs/superpowers/{specs,plans}/`, journals updated, PR pushed → **stop for user review** → squash-merge.
- **Do NOT rotate `ORCH_REPO_TOKEN`.** Lab CI stays red on the private-clone auth until the orchestrator repo goes public — known non-bug. Lab rounds are validated **locally green**; merge on local-green.
- **Always confirm before any `git push`** (including the final tag), per standing user preference. Local commits/merges need no ask.

---

## 3. The chokepoint that shapes ordering

`BaseAgent.run` → `_pre_run_gates` (policy → cost) is the single enforcement spine, shared by `run` and `run_stream`. Items **#2, #3, #4, #7 all modify this spine**, so they are strictly sequential — each rebased on the prior. Items **#5 (HTTP)** and **#6 (HITL)** touch different surfaces (serve app / langgraph engine) and are independent, but still land one-at-a-time per convention.

The user's risk order (1→8) already respects every dependency, so slices map 1:1 to the numbered items.

---

## 4. Slice table

| Slice | Item | Surface | Lab round | Depends on |
|---|---|---|---|---|
| **S0** | Journals catch-up (#19–23) + merge lab R14 | docs only — **direct to main, no PR** | — (R14 already 6/6) | none |
| **S1** | CapabilityEnforcer (#2) | skill-invocation seam | R15 | S0 |
| **S2** | BaseAgent/CLI security gate (#3) | `BaseAgent.run` chokepoint | R16 | S1 |
| **S3** | Cost caps + TOCTOU reservation (#4) | `CostGate` + `_pre_run_gates` | R17 | S2 |
| **S4** | HTTP hardening (#5) | `http_app` middleware | R18 | S3 |
| **S5** | HITL edited_input (#6) | langgraph engine + resume | R19 | S4 |
| **S6** | Agentic hygiene (#7) | `BaseAgent` + active run ctx | R20 | S5 |
| **S7** | Release engineering (#8) | packaging + CLI | R21 (full regression) | ALL |

---

## 5. Per-slice design

### S0 — Journals catch-up + merge lab R14
Streaming slices #19–23 are already merged to orch `main` (git log confirms `562c45b`, `768fc24`, `a3e6818`, `248815c`, `e5605d8`) but the journals (JOURNEY.md ends at PR #18, context.md ends at PR #18). Document them. Lab `round-14-real-streaming` already validates them (6/6) on branch `test/round-14-real-streaming` — merge it to lab main. No code, no PR, no new round.

### S1 — CapabilityEnforcer (#2, rule 11)
Per-skill-**CALL** enforcement (the policy engine already did per-**declared**-capability). Locate the exact skill-invocation seam an agent uses to call a skill, wrap it so the call is checked against the agent's declared `capabilities`; **fail-closed** — an undeclared skill call raises before the skill runs, and the block is audited. Same chokepoint philosophy as `SecurityGate` and `PolicyGate`.

### S2 — BaseAgent/CLI security gate (#3)
Today the real input-sanitize + injection-scan / output-validate + secret-scan gate lives only in the serve `AgentService`. Share the same gate at `BaseAgent.run` so `lottie run` and direct `BaseAgent` invocation are gated too. **Must not double-gate the serve path** — decide at slice-spec time between (a) serve delegates to the BaseAgent gate, or (b) BaseAgent skips when a caller has already gated. Fail-closed either way; messages leak no payload.

### S3 — Cost caps + TOCTOU reservation (#4)
Add a **per-run token cap** and replace the "block on prior cumulative, overshoot bounded by in-flight count" behavior with an **atomic reservation**: a `BEGIN IMMEDIATE` SQLite transaction appends a reservation step and checks `SUM(spent) + SUM(reserved) < budget` atomically; the run **settles** (or releases) the reservation on completion. Cross-process safe — matches the durable-resume shared-fs model. Fail-closed: configured budget + unreadable/disabled ledger → block.

### S4 — HTTP hardening (#5)
Middleware on the shared `http_app`:
- **Auth:** accept `Authorization: Bearer <k>` **and** `X-API-Key: <k>`; valid keys from env (e.g. `LOTTIE_API_KEYS`, csv); constant-time compare. **Open when unset, fail-closed when set** (opt-in security; preserves existing tests + lab TestClient rounds).
- **Rate limiting:** in-memory token bucket per key (per-process; documented as such).
- **Pagination:** `limit`/`offset` on list endpoints (`/v1/agents`, `/v1/models`).

No new dep beyond the existing `[api]` extra where avoidable.

### S5 — HITL edited_input-on-approve (#6)
Today `edited_input` is accepted but not applied. On approve-with-edit, **validate against the resumed worker's typed Input** (fail-closed on a bad edit), then apply via langgraph `Command(update=...)` so the downstream worker sees the edited value. Closes the deferral.

### S6 — Agentic hygiene (#7) — keep minimal, defaults off/generous
- **max_turns:** a counter on the active run context; each `complete()` / `stream_complete()` increments; raise `TurnLimitExceeded` past `AgentConfig.max_turns` (default `None` = off). Bounds runaway tool-use loops inside an `_execute`.
- **verify hook:** overridable `_verify(data, output)` called in `run()` after `_execute`, before the run declares success; default no-op; fail-closed (verify failure fails the run). Typed Pydantic config to enable.

### S7 — Release engineering (#8)
Version bump to `1.0.0`; `CHANGELOG.md`; upgrade/migration notes for the new config surface (`capabilities` enforcement, `max_turns`, per-run token cap, `LOTTIE_API_KEYS`, `_verify`); deprecation sweep (remove/annotate anything stale); `lottie doctor` checks for the new config. Full lab regression (R21) green, then tag `v1.0.0`.

---

## 6. Lab-round mapping

| Round | Validates | State |
|---|---|---|
| R14 | real token streaming #19–23 | **exists, 6/6, merge in S0** |
| R15 | per-skill-call capability enforcement | new |
| R16 | BaseAgent/CLI security gate | new |
| R17 | cost cap + atomic reservation circuit-breaker | new |
| R18 | HTTP auth + rate limit + pagination | new |
| R19 | HITL edited_input applied downstream | new |
| R20 | max_turns + verify hook | new |
| R21 | full V1 regression smoke (all surfaces) | new |

Lab CI red on `ORCH_REPO_TOKEN` throughout — known non-bug; validate locally.

---

## 7. Risks / open questions carried into slice specs

- **S2:** the double-gate decision (delegate vs skip) — resolve in the S2 slice spec.
- **S1:** exact skill-invocation seam must be located before writing the enforcer.
- **S3:** reservation settle-on-crash — ensure a reservation cannot leak and permanently block a budget (release in a `finally`, or a TTL/cleanup on read).
- **S4:** per-process rate-limit + auth is "minimum viable production," not distributed — document honestly.

---

## 8. Out of scope (V2)

Self-learning / reflection loop, agent-to-agent (A2A) protocol, distributed multi-host rate limiting, and any new agent/skill/transport capability. This epic adds **zero** new capabilities.
