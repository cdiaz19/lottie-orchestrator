"""Typed I/O for the assistant mesh (discovery-named aliases over mesh models)."""

from __future__ import annotations

from lottie.mesh.schema import MeshInput, MeshOutput


class AssistantInput(MeshInput):
    """Task input for the assistant mesh."""


class AssistantOutput(MeshOutput):
    """Final answer + step history from the assistant mesh."""
