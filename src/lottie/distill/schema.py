"""Typed models for skill distillation (V2 S3b).

A distilled skill is a **parameterized prompt template**, never generated Python
(epic decision D2). That sidesteps the rule-13 codegen pipeline and arbitrary-exec risk
entirely: nothing authored by an LLM is ever imported or executed, only rendered.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillSlot(BaseModel):
    """One typed hole in a distilled skill's template."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    description: str
    required: bool = True


class DistilledSkill(BaseModel):
    """A parameterized prompt template distilled from an agent's trajectories."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    description: str
    version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    system_prompt: str
    user_template: str  # contains {slot_name} placeholders for declared slots
    slots: list[SkillSlot] = []
    #: Rule-11 capability, declared by a human AT PROMOTION (S3c) — never by the model.
    #: None on a draft; set when promoted. An agent must declare BOTH `distilled` (to
    #: use the runner at all) and this name (to use this specific template).
    capability: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,39}$")

    def slot_names(self) -> set[str]:
        return {s.name for s in self.slots}

    def required_slots(self) -> set[str]:
        return {s.name for s in self.slots if s.required}


class DistillProvenance(BaseModel):
    """Where a distilled skill came from — the audit trail for a draft."""

    source_agent: str
    trajectory_count: int
    version: str
    run_ids: list[str] = []
    created_at: float | None = None  # epoch seconds; stamped at write


class TemplateRunInput(BaseModel):
    """Input to TemplateRunnerSkill: a distilled skill plus its slot values."""

    skill: DistilledSkill
    values: dict[str, str] = {}


class TemplateRunOutput(BaseModel):
    """Output of a distilled-skill execution."""

    result: str
    skill_name: str
    version: str


class PromotionRecord(BaseModel):
    """The human decision that turned a draft into a registered skill.

    Promotion is never automatic (rule 12's pattern). This record is the audit trail:
    who approved it, when, at which version, and under which capability.
    """

    skill_name: str
    capability: str
    reviewer: str
    source_version: str
    approved_at: float | None = None  # epoch seconds; stamped at promotion
