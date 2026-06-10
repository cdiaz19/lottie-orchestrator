"""SummarizerSkill — LLM-backed text summarisation.

Produces a concise prose summary plus up to ``max_points`` bullet-point
highlights from arbitrary input text.  All LLM access goes through the
injected ``LLMProvider``; no vendor SDK is imported here (CLAUDE.md rule 1).

Parsing contract
----------------
The skill asks the model to return:

1. A short prose paragraph (the *summary*).
2. Bullet points using ``-``, ``*``, ``•``, or decimal numbering (``1.``/``1)``).

Parsing is order-independent: every non-empty line is classified individually.
Lines matching the bullet regex become ``points`` (marker stripped); all other
non-empty lines become prose.

- ``summary`` = all prose lines joined with a single space (order-preserved,
  leading AND trailing prose both included).
- ``points`` = all bullet lines in order, capped to ``max_points``.
- Fallback: if there are NO prose lines (everything was bullets), ``summary``
  is set to the full stripped response text so it is never empty when the model
  returned content.
- No bullets at all → ``points=[]``, ``summary`` = full stripped response.
"""

from __future__ import annotations

import re
from pathlib import Path

from lottie.core import BaseSkill
from lottie.llm import LLMProvider, Message

from .schema import SummarizerInput, SummarizerOutput

# Matches lines that are bullet markers: -, *, •, or "N." / "N)" (1–2 digits only
# to avoid classifying prose like "100. Some sentence." as a list item).
_BULLET_RE = re.compile(r"^\s*([-*•]|\d{1,2}[.)]) +(.+)$")

_SYSTEM_PROMPT = """\
You are a precise summarisation assistant.
Return your answer in two parts, in exactly this order:
1. A concise prose paragraph summarising the key message (the "summary").
2. A list of the most important bullet points, one per line, each starting
   with "- ".

Do not add headers, preamble, or any other text.
"""


def _parse_response(text: str, max_points: int) -> tuple[str, list[str]]:
    """Split *text* into a prose summary and a capped bullet-point list.

    Parsing is order-independent: each non-empty line is classified on its own
    merit.  Lines matching ``_BULLET_RE`` become points; all other non-empty
    lines become prose.

    Parameters
    ----------
    text:
        Raw LLM response string.
    max_points:
        Maximum number of bullet points to return.

    Returns
    -------
    tuple[str, list[str]]
        ``(summary, points)`` where *summary* is all prose lines joined with a
        single space (or the full stripped response when no prose lines exist),
        and *points* is a list of at most *max_points* strings.
    """
    lines = text.splitlines()

    prose_lines: list[str] = []
    points: list[str] = []

    for line in lines:
        m = _BULLET_RE.match(line)
        if m:
            points.append(m.group(2).strip())
        else:
            stripped = line.strip()
            if stripped:
                prose_lines.append(stripped)

    # If there were no bullets, treat the entire response as prose.
    if not points:
        return text.strip(), []

    # Build summary from ALL prose lines (leading and trailing), joined.
    # Fallback: if the model returned only bullets, use the full text so that
    # summary is never empty when content was present.
    summary = " ".join(prose_lines) if prose_lines else text.strip()

    return summary, points[:max_points]


class SummarizerSkill(BaseSkill[SummarizerInput, SummarizerOutput]):
    """Summarise text into a prose paragraph plus capped bullet points.

    Parameters
    ----------
    llm:
        Injected ``LLMProvider``.  Accepts ``MockLLMProvider`` in tests.
    name:
        Optional display name forwarded to ``InstrumentedRunnable``.
    enable_benchmarks:
        If ``True``/``False``, overrides the ``LOTTIE_DISABLE_BENCHMARKS``
        env-var check.  ``None`` (default) defers to the env var.
    benchmarks_root:
        Directory under which benchmark JSONL files are appended.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        name: str | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._llm = llm

    def _execute(self, data: SummarizerInput) -> SummarizerOutput:
        """Call the LLM and parse its response into summary + points."""
        messages: list[Message] = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Summarise the following text in at most {data.max_points} "
                    f"bullet points:\n\n{data.text}"
                ),
            ),
        ]
        # TODO: LLM token usage from response.usage is NOT accumulated into a
        # RunContext here — skills lack the accumulator that BaseAgent.complete
        # provides.  As a result, last_metrics reports 0 tokens for skills.
        # This is a known architectural gap (Phase 2 tracking work item).
        response = self._llm.complete(messages)
        summary, points = _parse_response(response.content, data.max_points)
        return SummarizerOutput(summary=summary, points=points)
