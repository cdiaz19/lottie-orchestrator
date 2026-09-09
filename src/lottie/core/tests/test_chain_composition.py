"""`build_chain` composition rules — the guard that used to live in `ModuleRegistry`.

The registry was speculative API that the runtime never adopted: every real module needs
the *agent*, not the `Deps(bus, root)` it offered, so adopting it would have meant
rewriting it. It was deleted, and these tests move the one rule worth keeping — no two
modules may claim a chain position — onto the code that actually composes the chain.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.core.middleware import KNOWN_MODULES, build_chain
from lottie.llm import MockLLMProvider
from lottie.memory.middleware import RecallMiddleware
from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import ModuleConflictError, Next, Order


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent[BaseModel, BaseModel]):
    """Typed over `BaseModel` directly — `build_chain` takes the erased agent."""

    def _execute(self, data: BaseModel) -> BaseModel:
        return _Out(answer="ok")


def _agent() -> _Agent:
    return _Agent(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)


class _Squatter:
    """A module that claims a slot it does not own."""

    name = "squatter"

    def __init__(self, order: int) -> None:
        self.order = order

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:  # pragma: no cover
        raise AssertionError("a conflicting chain must never be built, let alone run")


class _SquattingAgent(_Agent):
    """Substitutes a colliding module where a real one would be mounted."""

    squat_order: int = Order.SECURITY_INPUT

    def _recall_module(self) -> RecallMiddleware:
        squatter: object = _Squatter(self.squat_order)
        return squatter  # type: ignore[return-value]  # deliberately the wrong slot


class TestOrdering:
    def test_every_module_has_a_distinct_position(self) -> None:
        orders = [m.order for m in build_chain(_agent())]
        assert len(orders) == len(set(orders))

    def test_the_input_gate_runs_before_anything_else(self) -> None:
        chain = sorted(build_chain(_agent()), key=lambda m: m.order)
        assert chain[0].name == "security_input"

    def test_the_capability_gate_is_innermost(self) -> None:
        # It must wrap the agent's own `_execute`, since that is where skill calls happen.
        chain = sorted(build_chain(_agent()), key=lambda m: m.order)
        assert chain[-1].name == "capability"

    def test_the_output_gate_runs_before_verify_declares_success(self) -> None:
        by_name = {m.name: m.order for m in build_chain(_agent())}
        assert by_name["security_output"] < by_name["verify"]


class TestConflictDetection:
    def test_a_position_collision_is_refused_at_composition(self) -> None:
        # A plugin displacing a security gate must fail at startup, never at run time.
        agent = _SquattingAgent(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        with pytest.raises(ModuleConflictError, match="already held by 'security_input'"):
            build_chain(agent)

    def test_the_error_names_both_claimants(self) -> None:
        # An operator needs to know WHO collided, not just that something did.
        agent = _SquattingAgent(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        agent.squat_order = Order.VERIFY
        with pytest.raises(ModuleConflictError) as exc:
            build_chain(agent)
        assert "squatter" in str(exc.value) and "verify" in str(exc.value)

    def test_disabling_the_incumbent_frees_the_slot(self) -> None:
        # The conflict is about the composed chain, not about the names in isolation.
        agent = _SquattingAgent(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        names = {m.name for m in build_chain(agent, disabled=frozenset({"security_input"}))}
        assert "squatter" in names


class TestDisabling:
    def test_a_disabled_module_is_absent_from_the_chain(self) -> None:
        names = {m.name for m in build_chain(_agent(), disabled=frozenset({"verify"}))}
        assert "verify" not in names

    def test_disabling_one_module_leaves_the_rest(self) -> None:
        full = {m.name for m in build_chain(_agent())}
        trimmed = {m.name for m in build_chain(_agent(), disabled=frozenset({"verify"}))}
        assert full - trimmed == {"verify"}

    def test_an_unknown_disabled_name_drops_nothing(self) -> None:
        # `doctor` and `modules` reject the typo; the builder itself stays total.
        full = build_chain(_agent())
        assert len(build_chain(_agent(), disabled=frozenset({"nope"}))) == len(full)

    def test_disabling_everything_yields_an_empty_chain(self) -> None:
        assert build_chain(_agent(), disabled=frozenset(KNOWN_MODULES)) == []
