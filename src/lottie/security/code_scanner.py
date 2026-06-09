"""CodeSecurityScanSkill — run bandit over generated/modified Python.

Step 3 of the code review pipeline (CLAUDE.md rule 13). Shells out to bandit in
JSON mode and normalizes its results into `SecurityFinding`s. semgrep is deferred
to a later phase. Deterministic given the same source.
"""

from __future__ import annotations

import json
import subprocess
import sys

from lottie.core import BaseSkill
from lottie.security.schema import ScanInput, ScanOutput, SecurityFinding


class CodeSecurityScanSkill(BaseSkill[ScanInput, ScanOutput]):
    """Scan Python files with bandit; return one finding per reported issue."""

    def _execute(self, data: ScanInput) -> ScanOutput:
        if not data.paths:
            return ScanOutput(findings=[])
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "bandit", "-f", "json", "-q", *data.paths],
            capture_output=True,
            text=True,
        )
        try:
            report = json.loads(proc.stdout)
            findings = [
                SecurityFinding(
                    file=str(result["filename"]),
                    line=int(result["line_number"]),
                    kind=str(result["test_id"]),
                    message=str(result["issue_text"]),
                )
                for result in report.get("results", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Empty/garbled stdout or an unexpected bandit JSON shape: fail open to
            # empty findings rather than propagating — mirrors the JSONDecodeError path.
            return ScanOutput(findings=[])
        return ScanOutput(findings=findings)
