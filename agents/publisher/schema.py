"""Typed I/O for PublisherAgent."""

from __future__ import annotations

from pydantic import BaseModel


class PublisherInput(BaseModel):
    text: str


class PublisherOutput(BaseModel):
    published: str
