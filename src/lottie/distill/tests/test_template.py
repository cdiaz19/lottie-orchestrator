"""TemplateRunnerSkill — rendering, slot validation, and the no-format guarantee."""

from __future__ import annotations

import pytest

from lottie.distill.schema import DistilledSkill, SkillSlot, TemplateRunInput
from lottie.distill.template import SlotError, TemplateRunnerSkill, render
from lottie.governance.capability import (
    CapabilityDenied,
    CapabilityGate,
    _active_capabilities,
)
from lottie.llm import MockLLMProvider


def _skill(**kw: object) -> DistilledSkill:
    base: dict[str, object] = {
        "name": "summarise",
        "description": "summarise a document",
        "system_prompt": "You summarise.",
        "user_template": "Summarise {doc} in {style} style.",
        "slots": [
            SkillSlot(name="doc", description="the document"),
            SkillSlot(name="style", description="tone", required=False),
        ],
    }
    base.update(kw)
    return DistilledSkill.model_validate(base)


class TestRender:
    def test_fills_declared_slots(self) -> None:
        out = render(_skill(), {"doc": "the report", "style": "terse"})
        assert out == "Summarise the report in terse style."

    def test_optional_slot_omitted_renders_empty(self) -> None:
        assert render(_skill(), {"doc": "x"}) == "Summarise x in  style."

    def test_missing_required_slot_raises(self) -> None:
        with pytest.raises(SlotError, match="missing required"):
            render(_skill(), {"style": "terse"})

    def test_unknown_slot_raises(self) -> None:
        with pytest.raises(SlotError, match="unknown slot"):
            render(_skill(), {"doc": "x", "nope": "y"})

    def test_error_names_the_skill(self) -> None:
        with pytest.raises(SlotError, match="summarise"):
            render(_skill(), {})


class TestRenderingIsNotFormat:
    """`str.format` on an LLM-authored template is an attribute-traversal info leak."""

    def test_attribute_traversal_in_template_is_inert(self) -> None:
        skill = _skill(
            user_template="leak {doc.__class__.__init__.__globals__}",
            slots=[SkillSlot(name="doc", description="d")],
        )
        # The placeholder is not a declared slot spelling, so it stays literal text.
        assert "__globals__" in render(skill, {"doc": "x"})

    def test_a_value_containing_a_placeholder_is_not_re_expanded(self) -> None:
        # A value that looks like a placeholder must be inserted literally, never
        # resolved against another slot.
        out = render(_skill(), {"doc": "{style}", "style": "terse"})
        assert "{style}" in out

    def test_braces_in_a_value_do_not_raise(self) -> None:
        # `.format` would raise KeyError/IndexError on stray braces; replacement cannot.
        assert "{" in render(_skill(), {"doc": "a { b } c", "style": "s"})


class TestTemplateRunnerSkill:
    def test_runs_the_template_through_the_llm(self) -> None:
        skill = TemplateRunnerSkill(
            MockLLMProvider(responses=["a summary"]), enable_benchmarks=False
        )
        out = skill.run(TemplateRunInput(skill=_skill(), values={"doc": "d", "style": "s"}))
        assert out.result == "a summary"

    def test_output_carries_skill_identity(self) -> None:
        skill = TemplateRunnerSkill(
            MockLLMProvider(responses=["x"]), enable_benchmarks=False
        )
        out = skill.run(TemplateRunInput(skill=_skill(), values={"doc": "d"}))
        assert (out.skill_name, out.version) == ("summarise", "0.1.0")

    def test_slot_error_propagates_fail_closed(self) -> None:
        runner = TemplateRunnerSkill(
            MockLLMProvider(responses=["x"]), enable_benchmarks=False
        )
        with pytest.raises(SlotError):
            runner.run(TemplateRunInput(skill=_skill(), values={}))

    def test_capability_name_is_distilled(self) -> None:
        # Rule 11: an agent declares `distilled` to invoke promoted templates.
        assert TemplateRunnerSkill.resolved_capability_name() == "distilled"


class TestRenderIsSinglePass:
    """Sequential per-slot replacement re-scans inserted text; this pins that it doesn't.

    Regression: `render` originally looped over a set of slot names calling
    `str.replace`, so a value containing `{other_slot}` was expanded a second time.
    Set iteration order made it appear only under some PYTHONHASHSEED values — green
    locally, red on CI.
    """

    def test_value_that_looks_like_another_slot_is_not_expanded(self) -> None:
        out = render(_skill(), {"doc": "{style}", "style": "terse"})
        assert out == "Summarise {style} in terse style."

    def test_value_that_looks_like_itself_is_not_expanded(self) -> None:
        out = render(_skill(), {"doc": "{doc}", "style": "s"})
        assert out == "Summarise {doc} in s style."

    def test_result_is_order_independent(self) -> None:
        # Both slots carry each other's placeholder; a re-scanning implementation
        # produces different output depending on which slot it happens to fill first.
        out = render(_skill(), {"doc": "{style}", "style": "{doc}"})
        assert out == "Summarise {style} in {doc} style."


class TestPerSkillCapability:
    """Rule 11 at template granularity: a promoted skill carries the capability its
    human reviewer declared, so an agent needs `distilled` AND that name."""

    def test_a_draft_without_a_capability_is_unconstrained(self) -> None:
        runner = TemplateRunnerSkill(MockLLMProvider(["ok"]), enable_benchmarks=False)
        token = _active_capabilities.set(CapabilityGate(["distilled"]))
        try:
            out = runner.run(TemplateRunInput(skill=_skill(), values={"doc": "d"}))
        finally:
            _active_capabilities.reset(token)
        assert out.result == "ok"

    def test_declared_capability_is_enforced(self) -> None:
        promoted = _skill(capability="doc_summary")
        runner = TemplateRunnerSkill(MockLLMProvider(["ok"]), enable_benchmarks=False)
        token = _active_capabilities.set(CapabilityGate(["distilled"]))
        try:
            with pytest.raises(CapabilityDenied):
                runner.run(TemplateRunInput(skill=promoted, values={"doc": "d"}))
        finally:
            _active_capabilities.reset(token)

    def test_holding_both_capabilities_allows_the_run(self) -> None:
        promoted = _skill(capability="doc_summary")
        runner = TemplateRunnerSkill(MockLLMProvider(["ok"]), enable_benchmarks=False)
        token = _active_capabilities.set(CapabilityGate(["distilled", "doc_summary"]))
        try:
            out = runner.run(TemplateRunInput(skill=promoted, values={"doc": "d"}))
        finally:
            _active_capabilities.reset(token)
        assert out.result == "ok"

    def test_the_runner_capability_alone_is_still_required(self) -> None:
        runner = TemplateRunnerSkill(MockLLMProvider(["ok"]), enable_benchmarks=False)
        token = _active_capabilities.set(CapabilityGate(["something_else"]))
        try:
            with pytest.raises(CapabilityDenied):
                runner.run(TemplateRunInput(skill=_skill(), values={"doc": "d"}))
        finally:
            _active_capabilities.reset(token)
