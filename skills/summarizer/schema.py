"""Typed input/output models for SummarizerSkill."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SummarizerInput(BaseModel):
    """Input for SummarizerSkill."""

    text: str
    max_points: int = Field(default=5, ge=1)


class SummarizerOutput(BaseModel):
    """Output from SummarizerSkill."""

    summary: str
    points: list[str] = []
