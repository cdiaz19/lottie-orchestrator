"""Typed state and I/O models for the agent mesh (all Pydantic — CLAUDE.md rule 2)."""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

FINISH = "FINISH"
"""Sentinel returned by the supervisor to end the routing loop."""


class StepResult(BaseModel):
    """One worker invocation's outcome, recorded in mesh history."""

    worker: str
    result: str
    step: int = 0
    metadata: dict[str, str] = {}


class MeshState(BaseModel):
    """Evolving, typed state threaded through every mesh node."""

    task: str
    history: Annotated[list[StepResult], operator.add] = []
    final: str | None = None

    def with_step(self, step: StepResult) -> MeshState:
        """Return a new state with `step` appended (does not mutate self)."""
        return self.model_copy(update={"history": [*self.history, step]})


class RouteDecision(BaseModel):
    """Supervisor's choice of the next worker, or FINISH."""

    next: str


class MeshInput(BaseModel):
    """Input to a mesh agent."""

    task: str
    max_steps: int = Field(default=8, ge=1)


class MeshOutput(BaseModel):
    """Output of a mesh agent."""

    final: str
    history: list[StepResult] = []
