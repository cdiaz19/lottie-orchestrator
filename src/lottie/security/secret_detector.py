"""SecretDetectionSkill — flag secrets in files before they are written or shipped.

Runs the `detect-secrets` plugin suite over each file and augments it with a small
set of high-signal custom regexes. Deterministic: same file content always yields
the same findings. Applied at knowledge ingest, on LLM outputs, and inside the
code-write gate (CLAUDE.md rules 9, 10, 13).
"""

from __future__ import annotations

import re
from pathlib import Path

from detect_secrets.core.secrets_collection import SecretsCollection
from detect_secrets.settings import default_settings

from lottie.core import BaseSkill
from lottie.security.schema import ScanInput, ScanOutput, SecurityFinding

_CUSTOM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PrivateKey", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("AWSAccessKey", re.compile(r"AKIA[0-9A-Z]{16}")),
]


class SecretDetectionSkill(BaseSkill[ScanInput, ScanOutput]):
    """Scan files for secrets via detect-secrets plus custom patterns."""

    def _execute(self, data: ScanInput) -> ScanOutput:
        seen: set[tuple[str, int, str]] = set()
        findings: list[SecurityFinding] = []
        for raw in data.paths:
            path = Path(raw)
            if not path.is_file():
                continue
            self._scan_detect_secrets(raw, seen, findings)
            self._scan_custom(raw, path, seen, findings)
        return ScanOutput(findings=findings)

    def _scan_detect_secrets(
        self,
        raw: str,
        seen: set[tuple[str, int, str]],
        findings: list[SecurityFinding],
    ) -> None:
        collection = SecretsCollection()
        with default_settings():
            collection.scan_file(raw)
        for _filename, secret in collection:
            key = (raw, secret.line_number, secret.type)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                SecurityFinding(
                    file=raw,
                    line=secret.line_number,
                    kind=secret.type,
                    message="potential secret",
                )
            )

    def _scan_custom(
        self,
        raw: str,
        path: Path,
        seen: set[tuple[str, int, str]],
        findings: list[SecurityFinding],
    ) -> None:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for kind, pattern in _CUSTOM_PATTERNS:
                if pattern.search(line):
                    key = (raw, lineno, kind)
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        SecurityFinding(
                            file=raw, line=lineno, kind=kind, message="potential secret"
                        )
                    )
