from __future__ import annotations

from lottie.scaffold.renderer import TemplateRendererSkill
from lottie.scaffold.schema import FieldSpec, PlanRenderContext, RenderInput


def _ctx(kind: str) -> PlanRenderContext:
    return PlanRenderContext(
        name="greeter",
        class_name="GreeterAgent" if kind == "agent" else "GreeterSkill",
        provider="anthropic/claude-sonnet-4-6",
        kind=kind,
        input_fields=[FieldSpec(name="who", type="str")],
        output_fields=[FieldSpec(name="greeting", type="str")],
        system_prompt="Greet warmly.",
        run_body='return GreeterAgentOutput(greeting="hi")'
        if kind == "agent"
        else 'return GreeterSkillOutput(greeting="hi")',
        tools=["WebSearchSkill"],
        input_sample='who="x"',
    )


def test_agent_schema_template_renders_fields() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="agent_desc/schema.py.j2", context=_ctx("agent")))
    assert "who: str" in out.content
    assert "greeting: str" in out.content
    assert "class GreeterAgentInput(BaseModel)" in out.content


def test_agent_py_template_injects_run_body() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="agent_desc/agent.py.j2", context=_ctx("agent")))
    assert "class GreeterAgent(BaseAgent" in out.content
    assert "        return GreeterAgentOutput(greeting=\"hi\")" in out.content


def test_agent_config_lists_tools() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="agent_desc/config.yaml.j2", context=_ctx("agent")))
    assert "capabilities: [WebSearchSkill]" in out.content


def test_skill_py_template_injects_run_body() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="skill_desc/skill.py.j2", context=_ctx("skill")))
    assert "class GreeterSkill(BaseSkill" in out.content
    assert "        return GreeterSkillOutput(greeting=\"hi\")" in out.content
