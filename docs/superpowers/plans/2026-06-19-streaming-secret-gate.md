# Streaming Secret Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure sync generator `StreamingSecretGate.scan_stream(deltas) -> Iterator[str]` that line-buffers a token-delta stream and emits only scanned-clean text, raising `OutputSecurityViolation` on the first secret.

**Architecture:** Holds a `SecretDetectionSkill`; accumulates deltas, scans each COMPLETE line (+ a capped overflow chunk + the flush remainder) via `scan_text` VERBATIM, yields scanned-clean text byte-for-byte. Sound because detect-secrets is line-scoped. Flush is after the for-loop (never `finally`), so an early consumer close skips it (no unverified bytes leak). Slice 2 of 3 for real streaming — no agent/transport/async wiring (slice 3).

**Tech Stack:** Python 3.12, detect-secrets (via the existing `SecretDetectionSkill`), pytest, `uv run` (mypy --strict, ruff).

**Design:** `docs/superpowers/specs/2026-06-19-streaming-secret-gate-design.md`

---

## File structure

- **Create** `src/lottie/serve/stream_gate.py` — `StreamingSecretGate` + the `_split_hold_overlap` helper + constants. (Placed in `serve/` not `security/`: it raises `serve.errors.OutputSecurityViolation` and is the streaming counterpart of `serve/security.py`'s `check_output`; serve→security is the established layering, `security/` imports no serve.)
- **Create** `src/lottie/serve/tests/test_stream_gate.py`.

Known facts (verified):
- `OutputSecurityViolation` is in `src/lottie/serve/errors.py` (a dependency-free leaf). `SecretDetectionSkill` is exported from `lottie.security` (`from lottie.security import SecretDetectionSkill`); `scan_text(content, source="output") -> list[SecurityFinding]` — non-empty list ⇒ a secret. It is deterministic (no LLM).
- detect-secrets default plugin set is line-scoped (27 plugins, all per-line); `aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` is flagged by KeywordDetector (a detect-secrets finding, NOT one of the custom regexes); a bare hex/base64 run is NOT flagged.
- `AKIA1234567890ABCDEF` is flagged by the custom `AWSAccessKey` regex.

---

## Task 1: Gate + line-buffered common path

**Files:**
- Create: `src/lottie/serve/stream_gate.py`
- Test: `src/lottie/serve/tests/test_stream_gate.py`

- [ ] **Step 1: Write the failing tests** — create `src/lottie/serve/tests/test_stream_gate.py`:

```python
from __future__ import annotations

import pytest

from lottie.serve.errors import OutputSecurityViolation
from lottie.serve.stream_gate import StreamingSecretGate

_AKIA = "AKIA" + "1234567890ABCDEF"


def test_multiline_all_clean_emits_byte_for_byte() -> None:
    gate = StreamingSecretGate()
    text = "first line\nsecond line\r\nthird no-newline-tail"
    # fed as one delta
    assert "".join(gate.scan_stream([text])) == text
    # fed split across many deltas -> same bytes
    deltas = ["fir", "st line\nsec", "ond line\r", "\nthird ", "no-newline-tail"]
    assert "".join(gate.scan_stream(deltas)) == text


def test_secret_on_a_line_raises_after_prior_clean_lines() -> None:
    gate = StreamingSecretGate()
    out: list[str] = []
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream([f"safe line one\nhere is {_AKIA} oops\nnever reached\n"]):
            out.append(piece)
    assert out == ["safe line one\n"]      # the clean line emitted
    assert _AKIA not in "".join(out)        # the secret line NEVER yielded


def test_secret_split_across_deltas_within_one_line_caught() -> None:
    gate = StreamingSecretGate()
    out: list[str] = []
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream(["AKIA0000", "00000000", "000A\n"]):  # one line, reassembled
            out.append(piece)
    assert "".join(out) == ""               # nothing emitted (the only line is the secret)


def test_flush_emits_final_partial_clean_line() -> None:
    gate = StreamingSecretGate()
    # no trailing newline -> the last line is emitted at flush
    assert "".join(gate.scan_stream(["a clean tail with no newline"])) == "a clean tail with no newline"
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/serve/tests/test_stream_gate.py -q` (ModuleNotFoundError: `lottie.serve.stream_gate`).

- [ ] **Step 3: Write the module (common path, no overflow yet)** — create `src/lottie/serve/stream_gate.py`:

```python
"""Line-buffered secret gate over a token-delta stream — emits only scanned-clean text.

The gate-soundness groundwork for real token streaming (slice 2 of 3): it lets `stream:true`
emit tokens while still satisfying CLAUDE.md rule 9 (no secret ever leaves). SOUNDNESS INVARIANT:
the configured detect-secrets plugin set is LINE-SCOPED (every plugin scans per line), so scanning
each complete line == the non-stream whole-text scan. If a future detect-secrets upgrade adds a
multi-line / whole-document plugin this gate becomes unsound; a pinned-plugin-set test guards it.

Lives in serve/ (not security/): it raises serve.errors.OutputSecurityViolation and is the streaming
counterpart of serve/security.py's check_output. serve -> security is the established layering.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from lottie.security import SecretDetectionSkill
from lottie.serve.errors import OutputSecurityViolation


class StreamingSecretGate:
    """Yield scanned-clean text from a delta stream; raise OutputSecurityViolation on a secret."""

    def __init__(self, secrets: SecretDetectionSkill | None = None) -> None:
        self._secrets = secrets or SecretDetectionSkill()

    def _raise_if_secret(self, text: str) -> None:
        if self._secrets.scan_text(text):
            raise OutputSecurityViolation("output withheld: secret detected")  # fixed label, no payload

    def scan_stream(self, deltas: Iterable[str]) -> Iterator[str]:
        """Line-buffer the deltas; emit only scanned-clean lines. The flush (final partial line)
        is AFTER the for-loop, never in a finally — so an early consumer .close() skips it and no
        unverified buffered bytes leak."""
        buffer = ""
        for delta in deltas:
            buffer += delta
            while "\n" in buffer:
                nl = buffer.index("\n")
                line = buffer[: nl + 1]          # complete line INCLUDING its newline (\r\n preserved)
                buffer = buffer[nl + 1 :]
                self._raise_if_secret(line)
                yield line
        if buffer:                                # flush the final partial line
            self._raise_if_secret(buffer)
            yield buffer
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/serve/tests/test_stream_gate.py -q`.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/serve` and `uv run ruff check src/lottie/serve` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/stream_gate.py src/lottie/serve/tests/test_stream_gate.py
git commit -m "feat(serve): StreamingSecretGate line-buffered scan_stream (common path)"
```

---

## Task 2: No-newline overflow path

**Files:**
- Modify: `src/lottie/serve/stream_gate.py`
- Test: `src/lottie/serve/tests/test_stream_gate.py`

- [ ] **Step 1: Write the failing tests** — append to `src/lottie/serve/tests/test_stream_gate.py`:

```python
def test_overflow_emits_with_overlap_and_reconstructs() -> None:
    gate = StreamingSecretGate()
    big = "x" * 9000  # > MAX_LINE (8192), no newline, all identifier chars
    out = list(gate.scan_stream([big]))
    assert len(out) >= 2                  # head emitted on overflow, tail at flush
    assert "".join(out) == big            # byte-for-byte reconstruction


def test_secret_in_overflow_buffer_raises_and_does_not_leak() -> None:
    gate = StreamingSecretGate()
    big = "x" * 9000 + _AKIA              # secret after a long no-newline run
    out: list[str] = []
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream([big]):
            out.append(piece)
    assert _AKIA not in "".join(out)      # the secret never streamed
```

(`_AKIA` is defined at the top of the file from Task 1.)

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/serve/tests/test_stream_gate.py -q -k overflow` (the second test may pass by accident if the whole buffer flushes; the first FAILS — without overflow handling a 9000-char no-newline buffer yields only once at flush, so `len(out) >= 2` fails).

- [ ] **Step 3: Add the overflow branch** — in `src/lottie/serve/stream_gate.py`, add `import re` and the module constants + helper (above the class):

```python
import re

# A long single-line (no-newline) response is emitted in capped chunks to bound memory. We hold
# back the trailing run of identifier/entropy characters (capped) so a partial high-entropy token —
# or a bounded regex (AKIA=20, PRIVATE-KEY header ~40) straddling the cut — is never emitted; the
# held tail is re-buffered and rescanned next round.
_MAX_LINE = 8192
_OVERLAP_CAP = 128
_TRAILING_IDENT_RUN = re.compile(r"[A-Za-z0-9+/=_\-]*$")


def _split_hold_overlap(buffer: str) -> tuple[str, str]:
    """Split a no-newline overflow buffer into (head_to_emit, tail_to_hold). `tail` is the trailing
    identifier/entropy run, capped at _OVERLAP_CAP; `head` is everything before it."""
    run = _TRAILING_IDENT_RUN.search(buffer)
    tail = run.group(0) if run else ""
    if len(tail) > _OVERLAP_CAP:
        tail = tail[-_OVERLAP_CAP:]
    head = buffer[: len(buffer) - len(tail)]
    return head, tail
```

In `scan_stream`, add the overflow check at the end of the `for` body (after the `while "\n"` loop, before the loop repeats):

```python
            if len(buffer) > _MAX_LINE:           # no-newline overflow -> emit a capped clean chunk
                head, buffer = _split_hold_overlap(buffer)
                self._raise_if_secret(head)
                yield head
```

(The flush block after the loop is unchanged — it scans + yields whatever tail remains.)

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/serve/tests/test_stream_gate.py -q` (all Task 1 + Task 2 tests).

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/serve` and `uv run ruff check src/lottie/serve` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/stream_gate.py src/lottie/serve/tests/test_stream_gate.py
git commit -m "feat(serve): StreamingSecretGate no-newline overflow with held identifier-run overlap"
```

---

## Task 3: Guard tests — early-close, detect-secrets reuse, plugin-set invariant

**Files:**
- Test: `src/lottie/serve/tests/test_stream_gate.py`

These assert behaviors of the existing code (no prod change expected). If any reveals a real gap, STOP and report.

- [ ] **Step 1: Write the tests** — append to `src/lottie/serve/tests/test_stream_gate.py`:

```python
def test_early_close_does_not_flush_held_buffer() -> None:
    """Flush is after the for-loop, NOT in a finally — so a consumer that closes the generator
    early never gets the unverified buffered tail (no leak), and close() does not raise."""
    gate = StreamingSecretGate()
    gen = gate.scan_stream(["clean line\n", "UNVERIFIED_TAIL_NO_NEWLINE"])
    first = next(gen)
    assert first == "clean line\n"
    gen.close()  # must NOT raise (a finally-flush would RuntimeError 'ignored GeneratorExit')
    with pytest.raises(StopIteration):
        next(gen)


def test_detect_secrets_keyword_secret_raises() -> None:
    """A secret flagged by detect-secrets (KeywordDetector), NOT a custom regex — proves the gate
    reuses the full scan_text, not just the bounded patterns."""
    gate = StreamingSecretGate()
    secret_line = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
    with pytest.raises(OutputSecurityViolation):
        list(gate.scan_stream([secret_line]))


_LINE_SCOPED_PLUGINS = {
    "ArtifactoryDetector", "AWSKeyDetector", "AzureStorageKeyDetector", "BasicAuthDetector",
    "CloudantDetector", "DiscordBotTokenDetector", "GitHubTokenDetector", "GitLabTokenDetector",
    "Base64HighEntropyString", "HexHighEntropyString", "IbmCloudIamDetector", "IbmCosHmacDetector",
    "IPPublicDetector", "JwtTokenDetector", "KeywordDetector", "MailchimpDetector", "NpmDetector",
    "OpenAIDetector", "PrivateKeyDetector", "PypiTokenDetector", "SendGridDetector", "SlackDetector",
    "SoftlayerDetector", "SquareOAuthDetector", "StripeDetector", "TelegramBotTokenDetector",
    "TwilioKeyDetector",
}


def test_configured_plugins_are_line_scoped() -> None:
    """Soundness guard: the gate is sound ONLY if every detect-secrets plugin is line-scoped.
    Pin the configured set; a detect-secrets upgrade that changes it fails here -> forces a human
    re-review (and verification that any new plugin is still per-line)."""
    from detect_secrets.settings import default_settings

    with default_settings() as settings:
        assert set(settings.plugins) == _LINE_SCOPED_PLUGINS
```

- [ ] **Step 2: Run the tests** — `uv run pytest src/lottie/serve/tests/test_stream_gate.py -q -k "early_close or keyword or line_scoped"`.

- [ ] **Step 3: Reconcile if needed** — these assert existing behavior. If `test_configured_plugins_are_line_scoped` fails because the installed detect-secrets exposes a different plugin set, update `_LINE_SCOPED_PLUGINS` to the OBSERVED set ONLY AFTER confirming each plugin is per-line (none multi-line/whole-document) — add a one-line comment noting the verification. If `test_early_close_...` fails (e.g. `gen.close()` raises), that means the flush is wrongly in a `finally` — FIX `scan_stream` to flush after the loop (it already should be). Do NOT weaken the early-close test.

- [ ] **Step 4: Full file green** — `uv run pytest src/lottie/serve/tests/test_stream_gate.py -q`.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/serve` and `uv run ruff check src/lottie/serve` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/tests/test_stream_gate.py
git commit -m "test(serve): stream gate early-close skip-flush, detect-secrets reuse, plugin-set invariant"
```

---

## Task 4: Closeout — full gate

**Files:** none (verification only)

- [ ] **Step 1: Full test suite** — `uv run pytest -q`. Expected: PASS — all prior ~836 tests plus the new stream-gate ones; pure new component, nothing else touched.

- [ ] **Step 2: Types** — `uv run mypy --strict src`. Expected: clean.

- [ ] **Step 3: Lint** — `uv run ruff check`. Expected: clean.

- [ ] **Step 4: Final commit (if any closeout fixes)**

```bash
git add -A
git commit -m "chore(serve): closeout fixes for StreamingSecretGate (mypy/ruff)"
```

---

## Notes for the implementer

- **Reuse `scan_text` verbatim** — do NOT reimplement secret detection. A non-empty `scan_text` result is a secret.
- **Flush AFTER the for-loop, never in a `finally`** — this is a correctness requirement: a generator can't yield during `GeneratorExit`, and on early close the buffered tail is unverified and must NOT be emitted. Task 3's `test_early_close_...` guards this.
- **Byte-for-byte:** split on `\n` keeping the newline in the emitted line (`buffer[:nl+1]`); `\r\n` is preserved because the `\r` stays in the line. Never strip/normalize.
- **Privacy:** the raise message is the fixed `"output withheld: secret detected"` — never the offending text.
- **No agent/transport/async/OutputValidation wiring** — that's slice 3. Do NOT touch `openai_app`/`http_app`/`service`/`core`/`security`. This slice is the one new file + its test.
- **Honest residual (already in the module docstring intent):** the no-newline-overflow path is the only non-fully-sound corner (a high-entropy token longer than `_OVERLAP_CAP` straddling the cap); bounded regexes are always overlap-protected; the newline-delimited path is fully sound. Don't overclaim.
```
