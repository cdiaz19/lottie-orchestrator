"""Transport-agnostic serving core: list and run agents by name."""

from __future__ import annotations

from pathlib import Path

from lottie.project.discovery import discover_agents
from lottie.serve.schema import AgentInfo
from lottie.serve.security import SecurityGate


class ServeError(Exception):
    """Base for serving-core errors. Transport-agnostic — no typer."""


class AgentNotFoundError(ServeError):
    """No agents/<name>/agent.py exists."""


class InvalidInputError(ServeError):
    """The payload failed the agent's Input validation."""


class AgentExecutionError(ServeError):
    """The agent raised while running."""


class AgentService:
    """Lists and runs agents under a project root, gated by a SecurityGate."""

    def __init__(self, root: Path, *, gate: SecurityGate | None = None) -> None:
        self._root = root
        self._gate = gate or SecurityGate()

    def list_agents(self) -> list[AgentInfo]:
        """One AgentInfo per discovered agent. Import-free."""
        return [
            AgentInfo(name=unit.name, provider=unit.provider)
            for unit in discover_agents(self._root)
        ]
