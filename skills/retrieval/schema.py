"""Typed input/output models for RetrievalSkill."""

from __future__ import annotations

from pydantic import BaseModel

from lottie.knowledge.schema import RetrievalQuery, RetrievalResult


class RetrievalSkillInput(BaseModel):
    """Input for RetrievalSkill."""

    query: RetrievalQuery


class RetrievalSkillOutput(BaseModel):
    """Output from RetrievalSkill."""

    result: RetrievalResult
