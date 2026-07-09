"""Fail-closed content gate for memory writes (CLAUDE.md rules 8/9/10).

Content headed for the memory store is screened with the same scanners that guard
serve I/O: oversize/control-char sanitize, prompt-injection scan, secret scan. Any
trip raises MemoryContentRejected — the write never happens. This is the memory-
poisoning defense: a run must not be able to write instructions or secrets that
hijack or exfiltrate from future runs. Messages never echo the offending content.

Imports only lottie.security — never memory/core, so security stays a leaf.
"""

from __future__ import annotations

from lottie.security.injection_scanner import PromptInjectionScanSkill
from lottie.security.input_sanitizer import InputSanitizerSkill
from lottie.security.schema import InjectionScanInput, SanitizeInput
from lottie.security.secret_detector import SecretDetectionSkill


class MemoryContentRejected(Exception):
    """Raised when content fails a memory-write security check. Carries no content."""


class MemoryContentGate:
    """Detect-and-block screen over content before it enters the memory store."""

    def __init__(self) -> None:
        self._sanitizer = InputSanitizerSkill()
        self._injection = PromptInjectionScanSkill()
        self._secrets = SecretDetectionSkill()

    def check(self, content: str) -> None:
        screen = self._sanitizer.run(SanitizeInput(content=content))
        if not screen.ok:
            raise MemoryContentRejected(f"memory write rejected: {screen.reason}")
        if self._injection.run(
            InjectionScanInput(content=content, source="memory-write")
        ).flagged:
            raise MemoryContentRejected("memory write rejected: prompt-injection detected")
        if self._secrets.scan_text(content, source="memory-write"):
            raise MemoryContentRejected("memory write rejected: secret detected")
