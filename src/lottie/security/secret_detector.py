"""SecretDetectionSkill — flag secrets in files before they are written or shipped.

Runs the `detect-secrets` plugin suite over each file and augments it with a small
set of high-signal custom regexes. Deterministic: same file content always yields
the same findings. Applied at knowledge ingest, on LLM outputs, and inside the
code-write gate (CLAUDE.md rules 9, 10, 13).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from detect_secrets.core.secrets_collection import SecretsCollection
from detect_secrets.settings import default_settings

from lottie.core import BaseSkill
from lottie.security.schema import ScanInput, ScanOutput, SecurityFinding

# Custom patterns are a version-stable safety net independent of detect-secrets'
# plugin labels — they keep firing even if the library renames or drops a detector.
# Their `kind` strings deliberately differ from detect-secrets' (e.g. "AWSAccessKey"
# vs "AWS Access Key"), so a key on the same line can surface twice. That is benign:
# the write-gate fails on any finding, and these explicit kinds are the test contract.
# Patterns use `search` (not `fullmatch`) so a key embedded in a larger token is caught.
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

    def scan_text(self, content: str, source: str = "output") -> list[SecurityFinding]:
        """Secret-scan a string by reusing the file-based _execute on a private temp file.

        Behavior-preserving: delegates to _execute; only the temp round-trip and the
        `file` relabel are new, so a finding never leaks the temp path.
        """
        fd, tmp = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            findings = self._execute(ScanInput(paths=[tmp])).findings
        finally:
            os.unlink(tmp)
        return [f.model_copy(update={"file": source}) for f in findings]

    def _scan_custom(
        self,
        raw: str,
        path: Path,
        seen: set[tuple[str, int, str]],
        findings: list[SecurityFinding],
    ) -> None:
        # errors="replace": a binary / non-UTF-8 file in the path list must not abort
        # the whole scan and silently skip the remaining files.
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
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
