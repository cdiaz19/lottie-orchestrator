# Real Token Streaming — Slice 2: Streaming Secret Gate — Design

> A pure, sync generator that line-buffers a token-delta stream and emits only scanned-clean text —
> raising `OutputSecurityViolation` the instant a complete line (or capped chunk) trips the existing
> `SecretDetectionSkill`. The gate-soundness groundwork for real token streaming (rule 9 preserved on a
> stream). Security/serve layer ONLY — NO agent/transport/SSE/async wiring (that's slice 3).

- **Date:** 2026-06-19
- **Phase:** Phase 4+ — Real Token Streaming, slice 2 of 3 (provider stream [done] → **this** → SSE seam).
- **Branch:** `feat/streaming-secret-gate` (off `main`).

---

## 1. Context & decomposition

`LLMProvider.stream()` shipped (slice 1, PR #19) — providers can now yield content deltas. Real
token streaming over `/v1/chat/completions` collides with the fail-closed output gate (CLAUDE.md rule 9:
no secret/invalid output ever leaves). Brainstorming resolved it: scan the stream **line-buffered** and
emit only scanned-clean lines — sound because the configured `detect-secrets` plugin set is **line-scoped**
(per-line scanning == the non-stream whole-text scan). This slice builds that gate as a pure component;
slice 3 wires it into the SSE transport (async bridge, agent seam, OutputValidation cap).

**Locked decisions (resolved + confirmed against the code; do not relitigate):**
- **Line-buffered**, reusing `SecretDetectionSkill.scan_text` **verbatim** (no detection reimplemented).
- **Generator transformer** `scan_stream(deltas) -> Iterator[str]` — NOT a stateful feed/flush object.
- **Sync** generator (BaseAgent/provider are sync; slice 3 bridges to async SSE via `anyio.to_thread`,
  the MCP-path pattern). Deliberate.
- **Abort = raise the existing `OutputSecurityViolation`** so slice 3 catches it uniformly with the
  non-stream withhold path.

**Code-confirmation finding (FG-1 — deviation from the brief, with rationale):** the brief said
`security/stream_gate.py`. But `OutputSecurityViolation` lives in `serve/errors.py`, and the established
layering is **serve → security** (serve imports security; `security/` imports no `serve` — verified).
A gate in `security/` raising `OutputSecurityViolation` would invert that. The streaming gate is the
streaming counterpart of `serve/security.py`'s `check_output` (it gates serve output), so it belongs in
**`serve/stream_gate.py`** — serve→security is allowed, and `serve/errors.py` is right there. (`serve/errors`
is a documented dependency-free leaf; placing the gate in serve keeps the layering clean.)

## 2. Component — `serve/stream_gate.py`

```python
class StreamingSecretGate:
    """Line-buffered secret gate over a token-delta stream. Emits only scanned-clean text;
    raises OutputSecurityViolation on the first complete line/chunk that trips SecretDetection."""

    def __init__(self, secrets: SecretDetectionSkill | None = None) -> None:
        self._secrets = secrets or SecretDetectionSkill()

    def scan_stream(self, deltas: Iterable[str]) -> Iterator[str]:
        """Yield scanned-clean text as deltas arrive; raise OutputSecurityViolation on a secret."""
```

Holds a `SecretDetectionSkill` (constructor-injectable for tests). `Iterable[str]` in (slice 3 feeds it
`provider.stream(...)`), `Iterator[str]` out (the safe text slice 3 wraps in SSE chunks).

## 3. Algorithm (line-buffered; sound)

```python
    def scan_stream(self, deltas):
        buffer = ""
        for delta in deltas:
            buffer += delta
            while "\n" in buffer:
                nl = buffer.index("\n")
                line = buffer[: nl + 1]          # complete line INCLUDING its newline (\r\n preserved)
                buffer = buffer[nl + 1 :]
                self._raise_if_secret(line)
                yield line
            if len(buffer) > _MAX_LINE:           # unterminated over-long line -> fail closed, no partial emit
                raise OutputSecurityViolation(
                    f"output withheld: unterminated line exceeds {_MAX_LINE} bytes"
                )
        # FLUSH the final (complete-at-end) line, scanned whole — AFTER the loop, NEVER in a finally (see §4).
        if buffer:
            self._raise_if_secret(buffer)
            yield buffer

    def _raise_if_secret(self, text: str) -> None:
        if self._secrets.scan_text(text):
            raise OutputSecurityViolation("output withheld: secret detected")  # fixed label, NO payload
```

- **Emit ONLY complete lines** — a sub-line (partial-line) emit could split a line-scoped secret across two
  clean-looking chunks (e.g. `"-----BEGIN RSA PRIVATE "` emits clean; `"KEY-----"` held → full header leaked).
  The fix: no partial-line bytes ever leave the gate.
- **Per-line scan** = the non-stream `check_output` scan (detect-secrets reports per line; custom regexes
  are per-line via `splitlines()`). A secret split across *deltas* but within one line is caught — the line
  is fully reassembled before scanning.
- **No-newline lines** buffer until a newline arrives or the stream ends; they are scanned WHOLE at flush
  (never split). A single-line (no-newline) response is a format-level choice — sub-line streaming of a
  line-scoped-gated output is NOT sound, so it is deliberately not done.
- **`_MAX_LINE = 65536`** (64 KB): an unterminated line (no newline ever arrives) beyond this cap is
  withheld fail-closed — raises `OutputSecurityViolation`, nothing emitted (bounds memory; no partial emit).
  No `_OVERLAP_CAP` / `_split_hold_overlap` — those helpers are gone.

## 4. Two required correctness details

1. **Flush after the for-loop, never in a `finally`.** A generator cannot `yield` during `GeneratorExit`
   (raised when a consumer closes the generator early). More importantly, on early close we MUST NOT flush:
   the buffered tail is *unverified* — flushing it would emit (or scan-and-maybe-leak) bytes the consumer
   never asked for. So the flush lives after the `for`; if the consumer stops early, the held buffer is
   silently dropped (never emitted). A test asserts an early `.close()` does NOT yield the held tail.
2. **Sync, on purpose.** No `async def`. Slice 3 runs `scan_stream(provider.stream(...))` inside
   `anyio.to_thread.run_sync` (the existing MCP/transport pattern) to bridge to the async SSE response.

## 5. Soundness invariant + guard (the load-bearing assumption)

The whole argument rests on **every configured detect-secrets plugin being line-scoped** (per-line
`analyze_line`). Confirmed: the default plugin set (AWSKeyDetector, Base64/HexHighEntropyString,
PrivateKeyDetector, KeywordDetector, … 27 plugins) is entirely per-line; `_scan_custom` iterates
`splitlines()`. **If a future detect-secrets upgrade adds a multi-line/whole-document plugin, this gate
becomes unsound silently.** Guard: a test pins the configured plugin-name set; any change fails the test
→ forces a human re-review before the gate ships against a changed scanner. The module docstring states
the invariant.

## 6. Abort semantics

On a secret, `scan_stream` raises `OutputSecurityViolation` (the existing `serve.errors` type, fixed
message `"output withheld: secret detected"` — never the payload). **Already-yielded clean lines stay
emitted** — that is the guarantee: only scanned-clean bytes ever leave, so a later secret cannot
retroactively leak earlier output. Slice 3's transport catches the raise and ends the SSE with a
`content_filter` finish (parity with the non-stream 200-withhold).

## 7. Honest residual (do NOT overclaim — FG-1)

The no-newline-overflow partial-emit path **has been removed** (it was the source of confirmed secret-split
leaks). There is no longer a partial-line emit of any kind.

The new honest note: a single-line (no-newline) response buffers to completion and emits whole at flush
(= format-level for that line — sub-line streaming of a line-scoped-gated output is NOT sound, so it is
deliberately not done). An unterminated line beyond 64 KB (`_MAX_LINE`) is withheld fail-closed (raises
`OutputSecurityViolation`; nothing emitted). No secret-split leak exists under this design.

Two confirmed leak reproductions from adversarial review now have regression tests that assert each
raises `OutputSecurityViolation` with nothing leaked:
- `"x"*4000 + "-----BEGIN RSA PRIVATE KEY-----\n"` (SPACE in header, previously split at cut)
- `"x"*4000 + "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"` (keyword+value split)

## 8. Out of scope (slice 3)

- **OutputValidation** — oversized = a running total-byte counter that aborts at the threshold (having
  emitted up to it); empty = only knowable at flush → an error finish, not a withhold. Lives in the
  transport.
- **Agent/SSE wiring + the async bridge** — slice 3.
- **Perf:** `scan_text` does `mkstemp` per call, so per-line scanning is a temp-file create/write/scan/unlink
  PER LINE. Kept verbatim for soundness/parity this slice; an in-memory line scan is a deferred
  optimization — do NOT optimize now.

## 9. Testing (pure unit; no transport)

`serve/tests/test_stream_gate.py`. Inject a real `SecretDetectionSkill` (it's deterministic; no LLM).

- **Multi-line all-clean** → emits all lines; **`"".join(out)` equals the input BYTE-FOR-BYTE** (newlines
  preserved; `\r\n` preserved). Feed the same text both as one delta and split across many deltas.
- **`AKIA…` on a line** → raises `OutputSecurityViolation` AFTER the prior clean lines were emitted, and the
  secret line is NEVER yielded (collect yields up to the raise; assert the key absent).
- **Secret split across deltas within one line** (e.g. `"AKIA0000"` then `"00000000000A"` then `"\n"`) →
  caught (line reassembled before scanning).
- **Flush** — input with no trailing newline → the final partial clean line is emitted at stream end.
- **Overflow path** — a >`_MAX_LINE` no-newline buffer: emits with the trailing identifier-run held back;
  AND a secret in the over-cap buffer still raises.
- **Entropy secret** — a high-entropy token detect-secrets flags (e.g. a long Base64/hex run) on a line →
  raises (proves it's not regex-only).
- **Early close skips flush** — consume one line from the generator, then `.close()` it; assert the held
  buffer/tail was NOT yielded (no unverified bytes leak).
- **Plugin-set invariant** — assert `default_settings()`'s configured plugin name set equals the pinned
  line-scoped snapshot (fails if a future upgrade changes it).
- **Full gate**: `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` — green; existing
  ~836 tests unaffected (pure new component, nothing else touched).

## 10. Definition of done

`serve/stream_gate.py::StreamingSecretGate.scan_stream(deltas) -> Iterator[str]` line-buffers a sync delta
stream, scans each complete line + capped overflow chunk + the flush remainder via
`SecretDetectionSkill.scan_text` verbatim, yields only scanned-clean text BYTE-FOR-BYTE, and raises the
existing `OutputSecurityViolation` (fixed label, no payload) on the first secret — with already-emitted
clean output retained. Flush is after the for-loop (never `finally`); early close skips it. The
line-scoped-plugin-set invariant is documented + guarded by a pinned-set test. The no-newline-overflow
residual is documented, not overclaimed. No agent/transport/async/OutputValidation wiring (slice 3). `uv
run pytest -q` / `mypy --strict src` / `ruff check` green. (No lab round — internal component; the
real-streaming round lands with slice 3.) Commit on the feature branch; do not push until approved.
```
