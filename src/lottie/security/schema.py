"""Typed contracts for the security scan skills and the code-write gate.

`ScanInput`/`ScanOutput` are shared by the content scanners (secret + bandit).
`ValidateInput`/`ValidateOutput` carry the mypy+ruff verdict. `GateResult` is the
combined outcome returned by `guard_and_write`.
"""

from __future__ import annotations

from pydantic import BaseModel


class SecurityFinding(BaseModel):
    """A single issue located in a scanned file."""

    file: str
    line: int
    kind: str
    message: str


class ScanInput(BaseModel):
    """File paths to scan."""

    paths: list[str]


class ScanOutput(BaseModel):
    """Findings from a content scanner."""

    findings: list[SecurityFinding] = []


class ValidateInput(BaseModel):
    """File paths to type-check and lint."""

    paths: list[str]


class ValidateOutput(BaseModel):
    """Combined mypy + ruff verdict."""

    passed: bool
    diagnostics: str = ""


class GateResult(BaseModel):
    """Outcome of the rule-13 code-write gate."""

    passed: bool
    findings: list[SecurityFinding] = []
    diagnostics: str = ""
    files_written: list[str] = []


class InjectionScanInput(BaseModel):
    """Input for the prompt-injection scanner."""

    content: str
    source: str = "unknown"  # provenance label — e.g. a file path or URL


class InjectionScanOutput(BaseModel):
    """Output from the prompt-injection scanner."""

    flagged: bool
    findings: list[SecurityFinding] = []
    sanitized: str  # content with matched spans replaced by [REDACTED:INJECTION]


class SanitizeInput(BaseModel):
    """External input text to screen before it reaches an agent."""

    content: str
    max_len: int = 20_000


class SanitizeOutput(BaseModel):
    """Screen verdict. ok=False means the gate must reject (fail-closed)."""

    ok: bool
    reason: str = ""


class OutputCheckInput(BaseModel):
    """LLM output text to screen before it leaves Lottie."""

    content: str
    max_len: int = 100_000


class OutputCheckOutput(BaseModel):
    """Screen verdict. ok=False means the gate must withhold the output."""

    ok: bool
    reason: str = ""
