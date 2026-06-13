"""Supervisor routing: ask the LLM which worker runs next, validate the answer."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from lottie.llm import LLMResponse, Message
from lottie.mesh.errors import CapabilityViolation
from lottie.mesh.schema import FINISH, MeshState, RouteDecision

CompleteFn = Callable[[list[Message]], LLMResponse]

_SYSTEM = (
    "You are a supervisor routing a task to the single best-suited worker. "
    "Reply with ONLY the worker name, or FINISH when the task is complete. "
    "Never reply with anything else."
)


class SupervisorRouter:
    """Routes by LLM intent, constrained to a declared worker set."""

    def __init__(self, complete: CompleteFn) -> None:
        self._complete = complete

    def route(self, state: MeshState, workers: Mapping[str, str]) -> RouteDecision:
        roster = "\n".join(f"- {name}: {desc}" for name, desc in workers.items())
        done = "\n".join(f"[{s.worker}] {s.result}" for s in state.history) or "(none yet)"
        user = (
            f"Task: {state.task}\n\n"
            f"Workers:\n{roster}\n\n"
            f"Work done so far:\n{done}\n\n"
            "Next worker (or FINISH):"
        )
        raw = self._complete(
            [Message(role="system", content=_SYSTEM), Message(role="user", content=user)]
        ).content.strip()
        return RouteDecision(next=self._resolve(raw, workers))

    @staticmethod
    def _resolve(raw: str, workers: Mapping[str, str]) -> str:
        if raw.upper() == FINISH:
            return FINISH
        lowered = {name.lower(): name for name in workers}
        if raw.lower() in lowered:
            return lowered[raw.lower()]
        raise CapabilityViolation(
            f"supervisor chose undeclared worker {raw!r}; allowed: {sorted(workers)}"
        )
