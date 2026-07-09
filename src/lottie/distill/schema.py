"""Typed contracts for distilled template-skills (V2 S3). Pure data — pydantic only."""

from __future__ import annotations

from pydantic import BaseModel


class DistilledSkillSpec(BaseModel):
    """A reusable prompt template distilled from an agent's learned lessons."""

    name: str
    template: str            # prompt text with {slot} placeholders
    slots: list[str] = []    # slot names (extracted from template, deterministic)
    version: str = "0.1.0"


class DistillProvenance(BaseModel):
    """Where a distilled skill came from (which agent / notes produced it)."""

    source_agent: str
    namespace: str
    source_run_ids: list[str] = []
    version: str = "0.1.0"


class TemplateInput(BaseModel):
    """Input to the generic TemplateRunnerSkill: values for the template's slots."""

    slots: dict[str, str] = {}


class TemplateOutput(BaseModel):
    """Output of TemplateRunnerSkill: the LLM's response to the filled template."""

    content: str
