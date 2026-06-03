"""Transport-agnostic serving core: list and run agents by name."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from lottie.llm import build_provider
from lottie.project.config import load_agent_config
from lottie.project.discovery import (
    discover_agents,
    load_agent_class,
    load_input_model,
)
from lottie.serve.schema import AgentInfo, RunResult
from lottie.serve.security import SecurityGate


class ServeError(Exception):
    """Base for serving-core errors. Transport-agnostic — no typer."""


class AgentNotFoundError(ServeError):
    """No agents/<name>/agent.py exists."""


class InvalidInputError(ServeError):
    """The payload failed the agent's Input validation."""


class AgentExecutionError(ServeError):
    """The agent raised while running."""


class AgentLoadError(ServeError):
    """Agent exists but its config, schema, or module could not be loaded."""


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

    def run_agent(
        self,
        name: str,
        payload: Mapping[str, object],
        *,
        provider: str | None = None,
    ) -> RunResult:
        """Run one agent: gate input, validate, run, gate output, return metrics."""
        unit_dir = self._root / "agents" / name
        if not (unit_dir / "agent.py").is_file():
            raise AgentNotFoundError(f"agent '{name}' not found")

        self._gate.check_input(json.dumps(payload))

        try:
            cfg = load_agent_config(unit_dir)
            llm = build_provider(provider or cfg.provider)
            input_model = load_input_model(self._root, name)
        except Exception as exc:  # noqa: BLE001 — keep CLI/import errors out of the core
            raise AgentLoadError(f"cannot load agent '{name}': {exc}") from exc

        try:
            data = input_model.model_validate(payload)
        except ValidationError as exc:
            raise InvalidInputError(f"invalid input for '{name}': {exc}") from exc

        try:
            agent = load_agent_class(self._root, name)(llm=llm)
        except Exception as exc:  # noqa: BLE001 — class import/instantiation failure
            raise AgentLoadError(f"cannot load agent '{name}': {exc}") from exc

        try:
            output = agent.run(data)
        except Exception as exc:  # noqa: BLE001 — any agent failure → one typed error
            raise AgentExecutionError(f"agent '{name}' failed: {exc}") from exc

        self._gate.check_output(output.model_dump_json())

        m = agent.last_metrics
        return RunResult(
            agent=name,
            output=output.model_dump(),
            latency_ms=m.latency_ms if m else 0.0,
            input_tokens=m.input_tokens if m else 0,
            output_tokens=m.output_tokens if m else 0,
            cost_usd=m.cost_usd if m else 0.0,
        )
