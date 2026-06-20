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

# Memory bound: an unterminated line beyond this cap is withheld fail-closed (raises) — never
# emitted partially. A sub-line emit could split a line-scoped secret across two clean-looking
# chunks (e.g. "-----BEGIN RSA PRIVATE " emits clean; " KEY-----" held → full header leaked).
# 64 KB is generous for any real LLM line; no-newline responses buffer to completion and emit whole.
_MAX_LINE = 65536


class StreamingSecretGate:
    """Yield scanned-clean text from a delta stream; raise OutputSecurityViolation on a secret."""

    def __init__(self, secrets: SecretDetectionSkill | None = None) -> None:
        self._secrets = secrets or SecretDetectionSkill()

    def _raise_if_secret(self, text: str) -> None:
        if self._secrets.scan_text(text):
            # fixed label — never echo payload into the message
            raise OutputSecurityViolation("output withheld: secret detected")

    def scan_stream(self, deltas: Iterable[str]) -> Iterator[str]:
        """Line-buffer the deltas; emit ONLY complete scanned-clean lines (never partial-line bytes
        — a sub-line emit could split a line-scoped secret across two clean-looking chunks). A
        no-newline line buffers until a newline or stream end and is scanned WHOLE; an unterminated
        line beyond _MAX_LINE is withheld fail-closed (bounds memory, never emits a partial line).
        The flush is AFTER the for-loop, never in a finally — so an early consumer .close() drops
        the unverified tail (no leak)."""
        buffer = ""
        for delta in deltas:
            buffer += delta
            while "\n" in buffer:
                nl = buffer.index("\n")
                # complete line INCLUDING its newline (\r\n preserved)
                line = buffer[: nl + 1]
                buffer = buffer[nl + 1 :]
                self._raise_if_secret(line)
                yield line
            # unterminated over-long line -> fail closed, no partial emit
            if len(buffer) > _MAX_LINE:
                raise OutputSecurityViolation(
                    f"output withheld: unterminated line exceeds {_MAX_LINE} bytes"
                )
        # flush the final (complete-at-end) line, scanned whole
        if buffer:
            self._raise_if_secret(buffer)
            yield buffer
