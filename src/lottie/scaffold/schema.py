"""Typed contracts for template rendering and AI scaffolding.

The skill boundary is fully typed (CLAUDE.md rule 2); the dict handed to Jinja is
internal to the renderer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RenderContext(BaseModel):
    """Variables injected into a scaffold template."""

    name: str
    class_name: str
    provider: str = "anthropic/claude-sonnet-4-6"


class RenderInput(BaseModel):
    """Input to TemplateRendererSkill — which template, with what context."""

    template: str
    context: BaseModel


class RenderOutput(BaseModel):
    """Rendered template content."""

    content: str


class FieldSpec(BaseModel):
    """One typed field on a generated Input/Output model."""

    name: str
    type: str
    description: str = ""


class ScaffoldRequest(BaseModel):
    """A natural-language request to generate an agent or skill."""

    kind: Literal["agent", "skill"]
    name: str
    description: str
    repair_feedback: str | None = None


class ScaffoldPlan(BaseModel):
    """Structured plan the LLM produces; renders into a unit module."""

    class_name: str
    input_fields: list[FieldSpec] = Field(min_length=1)
    output_fields: list[FieldSpec] = Field(min_length=1)
    system_prompt: str
    run_body: str
    tools: list[str] = []


class PlanRenderContext(BaseModel):
    """Full context for the `*_desc` Jinja templates."""

    name: str
    class_name: str
    provider: str
    kind: Literal["agent", "skill"]
    input_fields: list[FieldSpec]
    output_fields: list[FieldSpec]
    system_prompt: str
    run_body: str
    tools: list[str]
    input_sample: str


class ScaffoldResult(BaseModel):
    """Outcome of an AI scaffold run."""

    files_written: list[str]
    passed: bool
    diagnostics: str = ""
