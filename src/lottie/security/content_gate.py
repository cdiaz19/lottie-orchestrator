"""Fail-closed content screen for untrusted text entering durable storage.

Content headed for the memory store or a distilled-skill draft is screened with the
same scanners that guard serve I/O: oversize/control-char sanitize, prompt-injection
scan, secret scan. Any trip raises `ContentRejected` — the write never happens.

This is the poisoning defence generalised: a run must not be able to write instructions
or secrets that hijack or exfiltrate from future runs, whether the vehicle is a memory
note (rule 13b) or an LLM-authored skill template (rule 10). Messages never echo the
offending content.

Imports only lottie.security — never memory/core/distill, so security stays a leaf.
"""

from __future__ import annotations

from lottie.security.injection_scanner import PromptInjectionScanSkill
from lottie.security.input_sanitizer import InputSanitizerSkill
from lottie.security.schema import InjectionScanInput, SanitizeInput
from lottie.security.secret_detector import SecretDetectionSkill


class ContentRejected(Exception):
    """Raised when content fails a write-time security check. Carries no content."""


class ContentGate:
    """Detect-and-block screen over untrusted content before it is persisted.

    `source` labels the write site in scanner findings; `error` lets a caller keep a
    domain-specific exception type without duplicating the screen itself.
    """

    def __init__(
        self,
        *,
        source: str,
        error: type[ContentRejected] = ContentRejected,
        label: str = "write",
    ) -> None:
        self._sanitizer = InputSanitizerSkill()
        self._injection = PromptInjectionScanSkill()
        self._secrets = SecretDetectionSkill()
        self._source = source
        self._error = error
        self._label = label

    def check(self, content: str) -> None:
        screen = self._sanitizer.run(SanitizeInput(content=content))
        if not screen.ok:
            raise self._error(f"{self._label} rejected: {screen.reason}")
        if self._injection.run(
            InjectionScanInput(content=content, source=self._source)
        ).flagged:
            raise self._error(f"{self._label} rejected: prompt-injection detected")
        if self._secrets.scan_text(content, source=self._source):
            raise self._error(f"{self._label} rejected: secret detected")
