"""Single security chokepoint for external content entering/leaving a run.

Identity for now. The real InputSanitizerSkill / OutputValidationSkill /
SecretDetectionSkill (Phase 1) swap in via constructor injection without
changing any call site. See CLAUDE.md rules 8 and 9.
"""

from __future__ import annotations


class SecurityGate:
    """Identity gate. Subclass / replace to perform real scanning."""

    def check_input(self, text: str) -> str:
        # TODO(phase1): route through InputSanitizerSkill
        return text

    def check_output(self, text: str) -> str:
        # TODO(phase1): route through OutputValidationSkill + SecretDetectionSkill
        return text
