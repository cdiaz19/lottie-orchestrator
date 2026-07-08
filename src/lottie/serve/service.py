"""Transport-agnostic serving core: list and run agents by name."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator, Mapping
from pathlib import Path
from typing import Protocol

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
from lottie.serve.errors import (
    NotResumable,
    OutputSecurityViolation,
    ServeError,
    ThreadNotFound,
)
from lottie.serve.schema import AgentInfo, RunResult
from lottie.serve.security import SecurityGate
from lottie.serve.stream_gate import StreamingSecretGate


class _ResumeDecision(Protocol):
    @property
    def action(self) -> str: ...

    @property
    def edited_input(self) -> dict[str, str]: ...


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
        self._stream_gate = StreamingSecretGate()  # incremental secret gate for the streaming path
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

        self._check_output(agent, output)
        return self._result(name, output, agent.last_metrics)

    def stream_agent(
        self,
        name: str,
        payload: Mapping[str, object],
        *,
        provider: str | None = None,
    ) -> Iterator[str] | None:
        """Real-token stream of an opt-in agent's output, secret-gated incrementally.

        Returns None if the agent does not implement `_stream` — the transport then uses the
        format-level fallback. Otherwise gates the input + validates EAGERLY (so those errors
        raise here, before the SSE starts), and returns scan_stream(run_stream(data)) — a lazy
        generator. NOTE the eager/lazy split: only the input gate + validation are eager; the
        policy/cost PRE-gates and audit live inside run_stream and fire on the FIRST pull, so a
        PolicyViolation/BudgetExceeded surfaces mid-stream (the transport maps it to an `error`
        finish), not as a pre-stream status.
        """
        self._require_agent(name)
        # _get_agent precedes check_input (unlike run_agent) because supports_streaming() needs the
        # instance; the (name, provider) cache makes this a one-time cost per agent.
        agent = self._get_agent(name, provider)
        if not agent.supports_streaming():
            return None  # not streamable -> caller falls back to format-level (no gating yet)
        self._gate.check_input(json.dumps(payload))
        try:
            input_model = load_input_model(self._root, name)
        except Exception as exc:  # noqa: BLE001 — keep CLI/import errors out of the core
            raise AgentLoadError(f"cannot load agent '{name}': {exc}") from exc
        try:
            data = input_model.model_validate(payload)
        except ValidationError as exc:
            raise InvalidInputError(f"invalid input for '{name}': {exc}") from exc
        return self._gated_stream(agent.run_stream(data))

    def _gated_stream(self, run_gen: Generator[str, None, None]) -> Iterator[str]:
        """Secret-gate run_stream's deltas, CLOSING run_stream on any exit.

        `scan_stream` loops over its source with a plain `for`, so it does NOT close `run_gen`
        when it raises (secret) or the consumer stops early — that would orphan run_stream and
        defer its audit + `_audit_depth` reset to GC (in the wrong context). The `finally` here
        closes it deterministically, in the caller's pinned context, so the reset always pairs.
        """
        try:
            yield from self._stream_gate.scan_stream(run_gen)
        finally:
            run_gen.close()

    def resume_agent(
        self,
        name: str,
        thread_id: str,
        decision: _ResumeDecision,
    ) -> RunResult:
        """Resume an interrupted mesh agent from its checkpoint.

        Durable when the engine uses the sqlite checkpointer (the `lottie serve --port`
        default, via LOTTIE_MESH_CHECKPOINT=sqlite): a fresh process rebuilds the agent by
        name and rehydrates the thread from the shared db. With the in-memory checkpointer it
        is process-local (only the process that ran the interrupt can resume).
        """
        # lazy imports: keep serve importable without the [mesh] extra
        from lottie.mesh.errors import EditedInputError, ThreadNotFoundError
        from lottie.mesh.schema import ApprovalDecision

        self._require_agent(name)
        agent = self._get_agent(name, None)
        resume = getattr(agent, "resume", None)
        if resume is None:
            raise NotResumable(f"agent '{name}' is not resumable (not a mesh)")
        # Precondition: REST handler validates ResumeRequest (action is a Literal) before this,
        # so the conversion cannot fail on the transport path; a direct caller passing a malformed
        # decision would surface a ValidationError here (acceptable — single validated transport).
        approval = ApprovalDecision.model_validate(
            {"action": decision.action, "edited_input": dict(decision.edited_input)}
        )
        try:
            output = resume(thread_id, approval)
        except ThreadNotFoundError as exc:
            raise ThreadNotFound(f"thread '{thread_id}' not found") from exc
        except EditedInputError as exc:  # bad human edit → client error, not 500
            raise InvalidInputError(f"invalid edited_input: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — any other failure → one typed error
            raise AgentExecutionError(f"agent '{name}' failed: {exc}") from exc
        self._check_output(agent, output)
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

    def _check_output(
        self,
        agent: BaseAgent[BaseModel, BaseModel],
        output: BaseModel,
    ) -> None:
        """Run the output gate; on withhold, re-raise carrying the run's token metrics
        so an HTTP transport can report `usage` for the (already-executed) run."""
        try:
            self._gate.check_output(output.model_dump_json())
        except OutputSecurityViolation as exc:
            m = agent.last_metrics
            raise OutputSecurityViolation(
                str(exc),
                input_tokens=getattr(m, "input_tokens", 0),
                output_tokens=getattr(m, "output_tokens", 0),
            ) from exc

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
