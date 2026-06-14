"""OutputValidationSkill — fail-closed screen for LLM output (CLAUDE.md rule 9).

Minimal by design: non-empty + size bound. Richer output policy belongs to the
deferred governance slice, not here. (Distinct from SchemaValidatorSkill, which
type-checks/lints generated code files.)
"""

from __future__ import annotations

from lottie.core import BaseSkill
from lottie.security.schema import OutputCheckInput, OutputCheckOutput


class OutputValidationSkill(BaseSkill[OutputCheckInput, OutputCheckOutput]):
    """Screen LLM output; verdict drives the gate's fail-closed decision."""

    def _execute(self, data: OutputCheckInput) -> OutputCheckOutput:
        if not data.content.strip():
            return OutputCheckOutput(ok=False, reason="empty")
        if len(data.content) > data.max_len:
            return OutputCheckOutput(ok=False, reason="oversized")
        return OutputCheckOutput(ok=True)
