"""InputSanitizerSkill — fail-closed screen for external input (CLAUDE.md rule 8).

Rejects oversized or control-character-laden input. It does not rewrite content:
the gate raises on a False verdict rather than silently mutating the payload.
"""

from __future__ import annotations

from lottie.core import BaseSkill
from lottie.security.schema import SanitizeInput, SanitizeOutput

# Allowed whitespace controls; everything else in C0 (0x00-0x1F) and C1/DEL
# (0x7F-0x9F) is disallowed in user-supplied text.
_ALLOWED_CONTROLS = {"\t", "\n", "\r"}


def _has_control_chars(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if ch in _ALLOWED_CONTROLS:
            continue
        if code < 0x20 or 0x7F <= code <= 0x9F:
            return True
    return False


class InputSanitizerSkill(BaseSkill[SanitizeInput, SanitizeOutput]):
    """Screen external input; verdict drives the gate's fail-closed decision."""

    def _execute(self, data: SanitizeInput) -> SanitizeOutput:
        if len(data.content) > data.max_len:
            return SanitizeOutput(ok=False, reason="oversized")
        if _has_control_chars(data.content):
            return SanitizeOutput(ok=False, reason="control-characters")
        return SanitizeOutput(ok=True)
