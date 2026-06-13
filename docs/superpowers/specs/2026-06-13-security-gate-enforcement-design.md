# Security Gate Enforcement (serve path) — Design

> Replace the identity `SecurityGate` with a real, **fail-closed** input/output checkpoint on
> the serve path, satisfying CLAUDE.md rules 8 & 9. Build the two missing skills
> (`InputSanitizerSkill`, `OutputValidationSkill`), reuse the existing injection + secret
> scanners, and raise a typed `SecurityViolation` on any hit.

- **Date:** 2026-06-13
- **Phase framing:** first slice of the post-mesh work. Governance (audit trail, policy engine,
  OpenTelemetry) and rule 11 (`CapabilityEnforcerSkill`) are explicitly **deferred** to later
  slices. This slice closes the "secure by construction" gap only.
- **Enforcement surface:** the serve path (`AgentService.run_agent` + `resume_agent`) only. The
  CLI `lottie run` path and in-process callers stay ungated (follow-up).

---

## 1. Goal & current gap

`src/lottie/serve/security.py` ships an **identity** `SecurityGate`: `check_input`/`check_output`
return their argument unchanged with a `# TODO(phase1)` note. So on the serve path, external input
is never sanitized/injection-scanned and LLM output is never validated/secret-scanned — CLAUDE.md
rules 8 and 9 are documented but not enforced.

The detection skills mostly already exist and are tested:
- `PromptInjectionScanSkill` — **text-native** (`InjectionScanInput(content, source)` →
  `InjectionScanOutput(flagged, findings, sanitized)`).
- `SecretDetectionSkill` — **file/path-based** (`ScanInput(paths)` → `ScanOutput(findings)`); runs
  `detect-secrets` + custom regexes over files.
- `SchemaValidatorSkill` — mypy+ruff over code **files**. This is code validation, **not** LLM
  output-content validation — not reusable here.

Missing entirely: `InputSanitizerSkill`, `OutputValidationSkill`, and a way to secret-scan an output
**string** (the secret skill only takes file paths).

This slice is done when the default serve gate runs real checks, blocks fail-closed with a typed
error, the two new skills exist with tests, and the existing serve tests still pass (the gate stays
constructor-injectable so tests can supply a permissive gate).

## 2. Posture & contract

**Fail-closed.** Any of: input sanitizer reject, injection flagged, output validation reject, or
secret finding → raise `SecurityViolation`. No redaction, no sanitize-and-proceed.

**Pure detect-and-block checkpoint.** The gate operates on the **serialized string** form of the
payload/output and never rewrites the structured payload. `AgentService` already calls the gate on
the JSON string (`check_input(json.dumps(payload))`, `check_output(output.model_dump_json())`) and
ignores the return value; we keep that — sanitization is fail-closed *rejection* of bad input, not
silent fixing. `check_input`/`check_output` change from returning `str` to returning `None` and
raising on violation. (Return-type change is safe: all call sites discard the return.)

**`SecurityViolation(ServeError)`** — new error in `serve/`. Because `AgentService.run_agent`/
`resume_agent` already let `ServeError` subclasses propagate, transports map it to a refusal with no
extra wiring. Message names the failing check + a short reason; it must **not** echo the offending
content (no secret/injection payload leakage in the error string).

## 3. New skills (text-native `BaseSkill`, deterministic, no LLM)

### `InputSanitizerSkill` — `src/lottie/security/input_sanitizer.py`
- Schema (`security/schema.py`): `SanitizeInput(content: str, max_len: int = 20_000)` →
  `SanitizeOutput(ok: bool, reason: str = "")`.
- Rules (fail-closed screen, no rewriting):
  - `ok=False, reason="oversized"` when `len(content) > max_len`.
  - `ok=False, reason="control-characters"` when the content contains disallowed control/
    non-printable chars (C0 controls except `\t\n\r`, plus the C1 range). Tab/newline/CR allowed.
  - else `ok=True`.
- Pure transform-free verdict skill; the gate raises when `ok=False`.

### `OutputValidationSkill` — `src/lottie/security/output_validator.py`
- Schema: `OutputCheckInput(content: str, max_len: int = 100_000)` →
  `OutputCheckOutput(ok: bool, reason: str = "")`.
- Rules: `ok=False, reason="empty"` when content is empty/whitespace-only;
  `ok=False, reason="oversized"` when `len(content) > max_len`; else `ok=True`.
- Deliberately minimal (YAGNI) — richer output policy belongs to the deferred governance slice.

> Naming note: `SchemaValidatorSkill` already exists for code (mypy+ruff); this new skill is
> `OutputValidationSkill` in its own module to avoid confusion. CLAUDE.md rule 9 names
> "OutputValidationSkill" — we match it.

## 4. Secret scanning on a string

`SecretDetectionSkill` is file-based; the gate has an output **string**. Add a method that reuses the
existing detection path verbatim (no detection-logic change):

```python
def scan_text(self, content: str, source: str = "output") -> list[SecurityFinding]:
    """Secret-scan a string by writing it to a private temp file and reusing _execute."""
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        findings = self._execute(ScanInput(paths=[tmp])).findings
    finally:
        os.unlink(tmp)
    # Relabel so findings reference the logical source, never the temp path.
    return [f.model_copy(update={"file": source}) for f in findings]
```

This keeps full `detect-secrets` + custom-regex coverage. The temp file is created with `mkstemp`
(0600), written, scanned, and unlinked in a `finally`. No new dependency.

**Behavior-preserving by construction.** `scan_text` *wraps* `_execute` — it does not change
`_execute`, `_scan_detect_secrets`, or `_scan_custom`. The existing file-path tests therefore keep
covering the detection logic unchanged. The only new behavior is the temp-file round-trip and the
`file`-relabel, which get their own direct tests (see §7) so the gate wires to a tested surface
rather than an end-to-end-only path. A **parity test** asserts that the same content yields equivalent
findings whether scanned via `_execute(ScanInput(paths=[file]))` or via `scan_text` — locking the
wrapper to the file path it stands in for.

## 5. The real `SecurityGate` — `src/lottie/serve/security.py`

```python
class SecurityGate:
    def __init__(self) -> None:
        self._sanitizer = InputSanitizerSkill()
        self._injection = PromptInjectionScanSkill()
        self._output = OutputValidationSkill()
        self._secrets = SecretDetectionSkill()

    def check_input(self, text: str) -> None:
        s = self._sanitizer.run(SanitizeInput(content=text))
        if not s.ok:
            raise SecurityViolation(f"input rejected: {s.reason}")
        scan = self._injection.run(InjectionScanInput(content=text, source="serve-input"))
        if scan.flagged:
            raise SecurityViolation("input rejected: prompt-injection detected")

    def check_output(self, text: str) -> None:
        v = self._output.run(OutputCheckInput(content=text))
        if not v.ok:
            raise SecurityViolation(f"output withheld: {v.reason}")
        if self._secrets.scan_text(text, source="serve-output"):
            raise SecurityViolation("output withheld: secret detected")
```

- Order: sanitize → injection (input); validate → secret (output). Cheapest/most-decisive check
  first; first failure short-circuits (fail-closed).
- Skills construct without an LLM (deterministic `BaseSkill`s).
- A permissive base class for tests: keep an `_AllowAllGate` (or expose the old identity behavior as a
  test helper) so existing serve tests that don't care about security inject it. The **default**
  `AgentService` gate becomes the real `SecurityGate`.

## 6. Wiring & blast radius

- `AgentService.__init__(..., gate: SecurityGate | None = None)` already exists; default changes from
  identity to the real gate. Call sites in `run_agent`/`resume_agent` are unchanged (they already
  call `check_input`/`check_output` and ignore the return).
- Existing serve tests that run a real agent through `run_agent` must still pass: their echo/mock
  outputs are clean text (no secrets/injection, non-empty, small), so the real gate passes them. Any
  test that happens to trip a check gets a permissive gate injected, or its fixture adjusted — decide
  per failing test during implementation, do not weaken a check.
- `serve/__init__.py` exports gain `SecurityViolation`.
- `security/__init__.py` exports gain `InputSanitizerSkill`, `OutputValidationSkill`, and the new
  schema models.

## 7. Testing

- **InputSanitizerSkill** (unit): clean → ok; oversized → reject; embedded NUL/control char →
  reject; tab/newline allowed.
- **OutputValidationSkill** (unit): clean → ok; empty/whitespace → reject; oversized → reject.
- **SecretDetectionSkill.scan_text** (unit — the gate's tested surface):
  - a string containing `AKIA...`/a PRIVATE KEY block → ≥1 finding with `file == "serve-output"`
    (never a temp path);
  - clean string → no findings;
  - the temp file is removed after the call (even when findings are raised internally) — assert no
    leftover temp artifact;
  - **parity:** write the same content to a real file, scan it via
    `SecretDetectionSkill()._execute(ScanInput(paths=[file]))`, and assert the finding `kind`/`line`
    set matches `scan_text(content)` (modulo the relabeled `file`). This guards the wrapper against
    drift from the file path it reuses.
  Existing `_execute`/file-path tests remain unchanged and continue to cover the detection logic.
- **SecurityGate** (unit): clean in/out → no raise; injection string → `SecurityViolation`; secret in
  output → `SecurityViolation`; oversized input → `SecurityViolation`; error message contains no
  offending content.
- **AgentService** (integration, MockLLM): happy path still returns a `RunResult`; an agent whose
  mocked output embeds a secret → `run_agent` raises `SecurityViolation`; a payload with an injection
  marker → `run_agent` raises `SecurityViolation` before the agent runs. Reuse the existing
  `_scaffold` echo-agent harness.
- Full gate: `pytest -q`, `mypy --strict src`, `ruff check`.

## 8. Out of scope (YAGNI / deferred)

- Governance: audit trail, policy engine (allow/deny/escalate), cost tracker, OpenTelemetry.
- Rule 11 `CapabilityEnforcerSkill` (agent→skill call authorization).
- Enforcement on `BaseAgent.run` / the CLI `lottie run` path (serve only this slice).
- Redaction / sanitize-and-proceed; configurable per-check posture.
- Logging/persisting blocked attempts (lands with the governance audit trail).
- Real-LLM tests.

## 9. Definition of done

Default serve `SecurityGate` runs the four checks fail-closed; `SecurityViolation(ServeError)` raised
on any hit and propagated by `AgentService`; `InputSanitizerSkill` + `OutputValidationSkill` built and
exported; `SecretDetectionSkill.scan_text` added (existing file-scan behavior unchanged); all new +
existing tests green; `mypy --strict` + `ruff` clean; error messages leak no offending content.
Commit on a feature branch; do not push until the user approves.
