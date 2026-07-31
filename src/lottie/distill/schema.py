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
