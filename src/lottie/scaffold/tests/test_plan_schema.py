from __future__ import annotations

import pytest
from pydantic import ValidationError

from lottie.scaffold.schema import FieldSpec, PlanRenderContext, ScaffoldPlan, ScaffoldRequest


def test_field_spec_defaults() -> None:
    f = FieldSpec(name="query", type="str")
    assert f.description == ""


def test_scaffold_plan_requires_fields() -> None:
    with pytest.raises(ValidationError):
        ScaffoldPlan(
            class_name="FooAgent",
            input_fields=[],
            output_fields=[FieldSpec(name="result", type="str")],
            system_prompt="hi",
            run_body="return FooAgentOutput(result='x')",
        )


def test_scaffold_plan_valid() -> None:
    plan = ScaffoldPlan(
        class_name="FooAgent",
        input_fields=[FieldSpec(name="query", type="str")],
        output_fields=[FieldSpec(name="result", type="str")],
        system_prompt="hi",
        run_body="return FooAgentOutput(result='x')",
    )
    assert plan.tools == []


def test_scaffold_request_fields() -> None:
    req = ScaffoldRequest(kind="agent", name="foo", description="does foo")
    assert req.repair_feedback is None


def test_plan_render_context_accepts_field_dicts() -> None:
    ctx = PlanRenderContext(
        name="foo",
        class_name="FooAgent",
        provider="anthropic/claude-sonnet-4-6",
        kind="agent",
        input_fields=[FieldSpec(name="query", type="str")],
        output_fields=[FieldSpec(name="result", type="str")],
        system_prompt="hi",
        run_body="...",
        tools=[],
        input_sample='query="x"',
    )
    assert ctx.input_sample == 'query="x"'
