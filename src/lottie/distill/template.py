"""TemplateRunnerSkill — the single executor for every distilled skill.

One generic skill runs all distilled skills: fill declared slots, call the injected
`LLMProvider`, return the result. No LLM-authored Python is ever imported or executed
(epic decision D2), so distillation adds no arbitrary-execution surface.

Rendering security
------------------
`str.format` is deliberately NOT used. A distilled template is LLM-authored, and
`"{x.__class__.__init__.__globals__}".format(x=obj)` is a well-known info-leak vector
through attribute traversal. Rendering is instead a literal replacement of `{slot}` for
*declared slots only*, so an unexpected placeholder in a template is inert text rather
than an expression.

Slot values are NOT re-screened here. They are ordinary runtime data arriving through
the calling agent, which already passed the SecurityGate input check (rule 8); screening
them again would reject legitimate content without adding a boundary. The untrusted
content in distillation is the *template*, and that is gated once at authoring time
(`store.write_draft`) and again by human review before promotion.
"""

from __future__ import annotations

import re
from pathlib import Path

from lottie.core import BaseSkill
from lottie.distill.schema import DistilledSkill, TemplateRunInput, TemplateRunOutput
from lottie.llm import LLMProvider, Message

_SLOT_RE = re.compile(r"\{([a-z][a-z0-9_]{0,39})\}")


class SlotError(ValueError):
    """A distilled skill was invoked with missing or unknown slot values."""


def render(skill: DistilledSkill, values: dict[str, str]) -> str:
    """Fill `skill.user_template`, validating slots fail-closed.

    Raises SlotError on a missing required slot or an undeclared value — a silent
    partial render would send a half-formed prompt to the model.
    """
    declared = skill.slot_names()
    unknown = set(values) - declared
    if unknown:
        raise SlotError(f"unknown slot(s) for {skill.name!r}: {sorted(unknown)}")
    missing = skill.required_slots() - set(values)
    if missing:
        raise SlotError(f"missing required slot(s) for {skill.name!r}: {sorted(missing)}")

    def _fill(match: re.Match[str]) -> str:
        name = match.group(1)
        # Undeclared placeholders stay literal — that is what keeps an authored
        # `{x.__class__...}` inert rather than evaluated.
        return values.get(name, "") if name in declared else match.group(0)

    # ONE left-to-right pass. Sequential str.replace per slot would re-scan text it
    # had already inserted, so a value containing `{other_slot}` would be expanded a
    # second time — the exact injection this function exists to prevent. (Caught by
    # CI: iterating a set made the bug appear only under some PYTHONHASHSEED values.)
    return _SLOT_RE.sub(_fill, skill.user_template)


class TemplateRunnerSkill(BaseSkill[TemplateRunInput, TemplateRunOutput]):
    """Execute any `DistilledSkill`: render slots, call the LLM, return the result."""

    #: Distilled skills share one capability name; an agent declares `distilled` to
    #: call any promoted template (rule 11). Per-skill capabilities are declared at
    #: promotion time (S3c).
    capability_name = "distilled"

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

    def _execute(self, data: TemplateRunInput) -> TemplateRunOutput:
        prompt = render(data.skill, data.values)
        response = self._llm.complete(
            [
                Message(role="system", content=data.skill.system_prompt),
                Message(role="user", content=prompt),
            ]
        )
        return TemplateRunOutput(
            result=response.content,
            skill_name=data.skill.name,
            version=data.skill.version,
        )
