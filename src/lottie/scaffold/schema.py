"""Typed contract for template rendering.

The skill boundary is fully typed (CLAUDE.md rule 2); the dict handed to Jinja is
internal to the renderer.
"""

from __future__ import annotations

from pydantic import BaseModel


class RenderContext(BaseModel):
    """Variables injected into a scaffold template."""

    name: str
    class_name: str
    provider: str = "anthropic/claude-sonnet-4-6"


class RenderInput(BaseModel):
    """Input to TemplateRendererSkill — which template, with what context."""

    template: str
    context: RenderContext


class RenderOutput(BaseModel):
    """Rendered template content."""

    content: str
