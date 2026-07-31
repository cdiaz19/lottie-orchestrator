"""TemplateRunnerSkill — rendering, slot validation, and the no-format guarantee."""

from __future__ import annotations

import pytest

from lottie.distill.schema import DistilledSkill, SkillSlot, TemplateRunInput
from lottie.distill.template import SlotError, TemplateRunnerSkill, render
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
