"""Single security chokepoint for content entering/leaving a serve-path run.

Fail-closed: any tripped check raises SecurityViolation (a ServeError), which
AgentService propagates and transports map to a refusal. The gate is a pure
detect-and-block screen over the serialized string — it never rewrites the payload.
Messages never echo the offending content. See CLAUDE.md rules 8 and 9.
"""

from __future__ import annotations

from lottie.security import (
    InputSanitizerSkill,
    OutputValidationSkill,
    PromptInjectionScanSkill,
    SecretDetectionSkill,
)
from lottie.security.schema import (
    InjectionScanInput,
    OutputCheckInput,
    SanitizeInput,
)
from lottie.serve.errors import InputSecurityViolation, OutputSecurityViolation


class SecurityGate:
    """Real input/output gate. Constructor-injectable so tests can swap it."""

    def __init__(self) -> None:
        self._sanitizer = InputSanitizerSkill()
        self._injection = PromptInjectionScanSkill()
        self._output = OutputValidationSkill()
        self._secrets = SecretDetectionSkill()

    def check_input(self, text: str) -> None:
        screen = self._sanitizer.run(SanitizeInput(content=text))
        if not screen.ok:
            raise InputSecurityViolation(f"input rejected: {screen.reason}")
        if self._injection.run(InjectionScanInput(content=text, source="serve-input")).flagged:
            raise InputSecurityViolation("input rejected: prompt-injection detected")

    def check_output(self, text: str) -> None:
        verdict = self._output.run(OutputCheckInput(content=text))
        if not verdict.ok:
            raise OutputSecurityViolation(f"output withheld: {verdict.reason}")
        if self._secrets.scan_text(text, source="serve-output"):
            raise OutputSecurityViolation("output withheld: secret detected")
