"""Transport-agnostic serving core: list and run agents by name."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ValidationError

from lottie.core import BaseAgent
from lottie.llm import build_provider
from lottie.project.config import AgentConfig, load_agent_config
from lottie.project.discovery import (
    discover_agents,
    instantiate_agent,
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
        # Cache constructed agents by (name, provider) so an interrupted mesh run
        # and its later resume_agent share the SAME engine instance — and thus the
        # same memoized in-memory checkpointer. Without this, resume_agent would
        # build a fresh MemorySaver that cannot see the interrupted checkpoint.
        # Plain agents are stateless per run (BaseAgent.run resets _active_ctx),
        # so reusing them across runs is safe.
        self._agents: dict[tuple[str, str | None], BaseAgent[BaseModel, BaseModel]] = {}

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
        self._require_agent(name)
        self._gate.check_input(json.dumps(payload))

        try:
            input_model = load_input_model(self._root, name)
        except Exception as exc:  # noqa: BLE001 — keep CLI/import errors out of the core
            raise AgentLoadError(f"cannot load agent '{name}': {exc}") from exc

        try:
            data = input_model.model_validate(payload)
        except ValidationError as exc:
            raise InvalidInputError(f"invalid input for '{name}': {exc}") from exc

        agent = self._get_agent(name, provider)

        try:
            output = agent.run(data)
        except Exception as exc:  # noqa: BLE001 — any agent failure → one typed error
            raise AgentExecutionError(f"agent '{name}' failed: {exc}") from exc

        self._gate.check_output(output.model_dump_json())
        return self._result(name, output, agent.last_metrics)

    def resume_agent(
        self,
        name: str,
        thread_id: str,
        decision: object,
    ) -> RunResult:
        """Resume an interrupted mesh agent from its checkpoint.

        NOTE: with the in-memory checkpointer the checkpoint is process-local, so
        this only works within the same process that ran the agent (the cached
        agent shares its engine's memoized MemorySaver). Durable cross-process
        resume requires the sqlite checkpointer (out of scope here).
        """
        self._require_agent(name)
        agent = self._get_agent(name, None)
        resume = getattr(agent, "resume", None)
        if resume is None:
            raise AgentExecutionError(
                f"agent '{name}' does not support resume (not a mesh)"
            )
        try:
            output = resume(thread_id, decision)
        except Exception as exc:  # noqa: BLE001 — any agent failure → one typed error
            raise AgentExecutionError(f"agent '{name}' failed: {exc}") from exc

        self._gate.check_output(output.model_dump_json())
        return self._result(name, output, agent.last_metrics)

    def _require_agent(self, name: str) -> None:
        """Raise AgentNotFoundError if agents/<name>/agent.py is absent."""
        if not (self._root / "agents" / name / "agent.py").is_file():
            raise AgentNotFoundError(f"agent '{name}' not found")

    def _get_agent(
        self, name: str, provider: str | None
    ) -> BaseAgent[BaseModel, BaseModel]:
        """Get-or-build the agent for (name, provider), caching the instance.

        Caching keeps the SAME engine instance across a run + resume in this
        process, so the mesh's in-memory checkpoint survives between calls.
        """
        key = (name, provider)
        cached = self._agents.get(key)
        if cached is not None:
            return cached
        try:
            cfg: AgentConfig = load_agent_config(self._root / "agents" / name)
            llm = build_provider(provider or cfg.provider)
            agent_cls = load_agent_class(self._root, name)
            agent = instantiate_agent(agent_cls, llm=llm, root=self._root, config=cfg)
        except Exception as exc:  # noqa: BLE001 — config/import/instantiation failure
            raise AgentLoadError(f"cannot load agent '{name}': {exc}") from exc
        self._agents[key] = agent
        return agent

    def _result(
        self,
        name: str,
        output: BaseModel,
        m: object,
    ) -> RunResult:
        """Map an agent output + last metrics into a RunResult.

        status/thread_id/pending are duck-typed off the output — only mesh
        outputs carry them; plain agents fall back to the defaults.
        """
        pending_obj = getattr(output, "pending", None)
        pending = pending_obj.model_dump() if pending_obj is not None else None
        return RunResult(
            agent=name,
            output=output.model_dump(),
            latency_ms=getattr(m, "latency_ms", 0.0),
            input_tokens=getattr(m, "input_tokens", 0),
            output_tokens=getattr(m, "output_tokens", 0),
            cost_usd=getattr(m, "cost_usd", 0.0),
            status=getattr(output, "status", "complete"),
            thread_id=getattr(output, "thread_id", None),
            pending=pending,
        )
