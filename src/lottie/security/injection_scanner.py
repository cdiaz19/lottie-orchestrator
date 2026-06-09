"""PromptInjectionScanSkill — detect prompt-injection markers in text.

Scans arbitrary text for common prompt-injection patterns: instruction overrides,
role/system spoofing, exfiltration requests, and data-control markers. Deterministic:
same input always yields the same findings in the same order. No LLM, no network.

Applied at knowledge ingest (CLAUDE.md rule 10) before any content reaches an agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lottie.core import BaseSkill
from lottie.security.schema import (
    InjectionScanInput,
    InjectionScanOutput,
    SecurityFinding,
)

_REDACT_MARKER = "[REDACTED:INJECTION]"


@dataclass(frozen=True)
class _Rule:
    label: str
    pattern: re.Pattern[str]


# ---------------------------------------------------------------------------
# Pattern table
# Each entry: (rule_label, raw_pattern_string).
# All compiled with re.IGNORECASE | re.DOTALL where appropriate.
# ---------------------------------------------------------------------------
_RULES: list[_Rule] = [
    # instruction-override
    _Rule(
        label="instruction-override/ignore-previous",
        pattern=re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    ),
    _Rule(
        label="instruction-override/disregard-above",
        pattern=re.compile(r"disregard\s+(the\s+)?above", re.IGNORECASE),
    ),
    _Rule(
        label="instruction-override/forget-everything",
        pattern=re.compile(
            r"forget\s+(all|everything)\s+(you|above)",
            re.IGNORECASE,
        ),
    ),
    # role / system override
    _Rule(
        label="role-override/you-are-now",
        pattern=re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    ),
    _Rule(
        label="role-override/act-as",
        pattern=re.compile(r"act\s+as\s+(a|an)\b", re.IGNORECASE),
    ),
    _Rule(
        label="role-override/system-prompt-phrase",
        pattern=re.compile(r"system\s+prompt", re.IGNORECASE),
    ),
    _Rule(
        label="role-override/fake-role-tag",
        pattern=re.compile(r"</?(?:system|assistant)>", re.IGNORECASE),
    ),
    # exfiltration / tool abuse
    _Rule(
        label="exfiltration/reveal-prompt",
        pattern=re.compile(
            r"reveal\s+(your\s+)?(system\s+)?prompt",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        label="exfiltration/print-instructions",
        pattern=re.compile(r"print\s+(your\s+)?instructions", re.IGNORECASE),
    ),
    _Rule(
        label="exfiltration/exfiltrate",
        pattern=re.compile(r"\bexfiltrate\b", re.IGNORECASE),
    ),
    _Rule(
        label="exfiltration/send-to-http",
        pattern=re.compile(r"send\s+.{0,40}?\bto\s+https?://", re.IGNORECASE | re.DOTALL),
    ),
    # data-control markers
    _Rule(
        label="data-control/begin-injection",
        pattern=re.compile(r"BEGIN\s+INJECTION", re.IGNORECASE),
    ),
    _Rule(
        label="data-control/end-injection",
        pattern=re.compile(r"END\s+INJECTION", re.IGNORECASE),
    ),
]


class PromptInjectionScanSkill(BaseSkill[InjectionScanInput, InjectionScanOutput]):
    """Scan text for prompt-injection markers; redact matched spans.

    Deterministic, no LLM, no network. Findings are ordered by ascending match
    position (character offset stored in the `line` field of `SecurityFinding`).
    """

    def _execute(self, data: InjectionScanInput) -> InjectionScanOutput:
        content = data.content
        source = data.source

        # Collect all (start_offset, end_offset, rule_label, matched_text) tuples.
        raw_hits: list[tuple[int, int, str, str]] = []
        for rule in _RULES:
            for m in rule.pattern.finditer(content):
                raw_hits.append((m.start(), m.end(), rule.label, m.group()))

        # Sort by start position for stable, position-ordered output.
        raw_hits.sort(key=lambda h: h[0])

        findings: list[SecurityFinding] = [
            SecurityFinding(
                file=source,
                line=start,  # character offset — best available position in raw text
                kind=label,
                message=(
                    f"Prompt-injection marker '{matched}' detected in {source}"
                ),
            )
            for start, _end, label, matched in raw_hits
        ]

        # Build sanitized string by applying all substitutions.
        sanitized = content
        for rule in _RULES:
            sanitized = rule.pattern.sub(_REDACT_MARKER, sanitized)

        return InjectionScanOutput(
            flagged=len(findings) > 0,
            findings=findings,
            sanitized=sanitized,
        )
