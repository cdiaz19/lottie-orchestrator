"""Unit tests for the per-run carrier threaded through the middleware chain."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import BaseModel

from lottie.core.metrics import Kind, RunContext
from lottie.runtime.context import ExecutionContext, RunKind, UsageAccumulator


class _Input(BaseModel):
    text: str


def _ctx() -> ExecutionContext:
    return ExecutionContext(runnable="Demo", kind="agent", input=_Input(text="hi"), run_id="r1")


class TestRunKindParity:
    def test_run_kind_mirrors_core_metrics_kind(self) -> None:
        # RunKind is duplicated rather than imported to keep the kernel free of a
        # `lottie.core` dependency. This pins the duplication so it cannot drift.
        assert set(get_args(RunKind)) == set(get_args(Kind))


class TestUsageAccumulator:
    def test_core_run_context_satisfies_the_protocol(self) -> None:
        # The kernel depends on this structural view, never on the concrete dataclass.
        assert isinstance(RunContext(), UsageAccumulator)

    def test_default_usage_starts_at_zero(self) -> None:
        ctx = _ctx()
        assert ctx.usage.input_tokens == 0
        assert ctx.usage.output_tokens == 0
        assert ctx.usage.cost_usd == 0.0
        assert ctx.usage.turns == 0

    def test_accepts_an_injected_accumulator(self) -> None:
        usage = RunContext()
        usage.input_tokens = 7
        ctx = ExecutionContext(
            runnable="Demo", kind="agent", input=_Input(text="hi"), run_id="r1", usage=usage
        )
        assert ctx.usage.input_tokens == 7


class TestScopedState:
    def test_scoped_creates_a_slice_on_first_use(self) -> None:
        ctx = _ctx()
        ctx.scoped("cost")["handle"] = 42
        assert ctx.state == {"cost": {"handle": 42}}

    def test_scoped_is_stable_across_calls(self) -> None:
        ctx = _ctx()
        ctx.scoped("cost")["handle"] = 42
        assert ctx.scoped("cost")["handle"] == 42

    def test_two_modules_do_not_collide_on_the_same_key(self) -> None:
        ctx = _ctx()
        ctx.scoped("cost")["token"] = "a"
        ctx.scoped("depth")["token"] = "b"
        assert ctx.scoped("cost")["token"] == "a"
        assert ctx.scoped("depth")["token"] == "b"

    def test_scoped_rejects_a_clobbered_slot(self) -> None:
        ctx = _ctx()
        ctx.state["cost"] = "not-a-dict"
        with pytest.raises(TypeError):
            ctx.scoped("cost")
