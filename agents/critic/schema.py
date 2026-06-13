"""Typed I/O for CriticAgent."""

from __future__ import annotations

from pydantic import BaseModel


class CriticInput(BaseModel):
    text: str


class CriticOutput(BaseModel):
    review: str
