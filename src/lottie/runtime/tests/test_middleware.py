"""Tests pinning the Lifecycle Hooks contract and the chain order table.

The order values are not arbitrary: unrolled through onion nesting they must reproduce
`BaseAgent.run`'s existing sequence (V3 spec section 4.5). S2 depends on that.
"""

from __future__ import annotations

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Middleware, Next, Order

ALL_ORDERS = [
    Order.SECURITY_INPUT,
    Order.POLICY,
    Order.COST,
    Order.SESSION,
    Order.TRAJECTORY,
    Order.DEPTH,
    Order.CAPABILITY,
    Order.RECALL,
    Order.REFLECT,
    Order.SECURITY_OUTPUT,
    Order.VERIFY,
]


class _Noop:
    name = "noop"
    order = Order.POLICY

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        return nxt(ctx)


class TestMiddlewareProtocol:
    def test_a_conforming_class_satisfies_the_protocol(self) -> None:
        # Structural typing: a middleware never inherits from anything.
        mw: Middleware = _Noop()
        assert mw.name == "noop"


class TestOrderTable:
    def test_the_declared_order_is_already_sorted(self) -> None:
        assert sorted(ALL_ORDERS) == ALL_ORDERS

    def test_all_orders_are_distinct(self) -> None:
        assert len(set(ALL_ORDERS)) == len(ALL_ORDERS)

    def test_output_side_concerns_unroll_as_verify_then_output_gate_then_reflect(
        self,
    ) -> None:
        # Post-phases run in REVERSE order, so today's
        # `_verify -> check_output -> _maybe_reflect` requires VERIFY > SECURITY_OUTPUT > REFLECT.
        assert Order.VERIFY > Order.SECURITY_OUTPUT > Order.REFLECT

    def test_cost_settles_outside_capability_and_depth_cleanup(self) -> None:
        # Cost's `finally` must be the outermost of the three, matching today's
        # `_cost.settle(handle)` running last in BaseAgent.run.
        assert Order.COST < Order.DEPTH < Order.CAPABILITY

    def test_trajectory_and_session_post_between_depth_reset_and_cost_settle(self) -> None:
        # v2.0.0 added `_persist_trajectory` then `_record_session_run` after the audit
        # write and before the settle. In reverse-unrolled post order that means both sit
        # strictly between COST and DEPTH.
        assert Order.COST < Order.SESSION < Order.TRAJECTORY < Order.DEPTH

    def test_trajectory_posts_before_session(self) -> None:
        # Higher order == earlier post phase, and today's code runs trajectory first.
        assert Order.TRAJECTORY > Order.SESSION
