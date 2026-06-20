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

import re
from collections.abc import Iterable, Iterator

from lottie.security import SecretDetectionSkill
from lottie.serve.errors import OutputSecurityViolation

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


class StreamingSecretGate:
    """Yield scanned-clean text from a delta stream; raise OutputSecurityViolation on a secret."""

    def __init__(self, secrets: SecretDetectionSkill | None = None) -> None:
        self._secrets = secrets or SecretDetectionSkill()

    def _raise_if_secret(self, text: str) -> None:
        if self._secrets.scan_text(text):
            # fixed label — never echo payload into the message
            raise OutputSecurityViolation("output withheld: secret detected")

    def scan_stream(self, deltas: Iterable[str]) -> Iterator[str]:
        """Line-buffer the deltas; emit only scanned-clean lines. The flush (final partial line)
        is AFTER the for-loop, never in a finally — so an early consumer .close() skips it and no
        unverified buffered bytes leak."""
        buffer = ""
        for delta in deltas:
            buffer += delta
            while "\n" in buffer:
                nl = buffer.index("\n")
                line = buffer[: nl + 1]  # include newline; \r\n is preserved
                buffer = buffer[nl + 1 :]
                self._raise_if_secret(line)
                yield line
            if len(buffer) > _MAX_LINE:           # no-newline overflow -> emit a capped clean chunk
                head, buffer = _split_hold_overlap(buffer)
                self._raise_if_secret(head)
                yield head
        if buffer:                                # flush the final partial line
            self._raise_if_secret(buffer)
            yield buffer
