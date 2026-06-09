from __future__ import annotations

import pytest
from jinja2 import StrictUndefined, TemplateError

from lottie.scaffold.renderer import TemplateRendererSkill
from lottie.scaffold.schema import RenderContext, RenderInput, RenderOutput


def test_render_context_defaults_provider() -> None:
    ctx = RenderContext(name="web_search", class_name="WebSearchSkill")
    assert ctx.provider == "anthropic/claude-sonnet-4-6"


def test_render_input_wraps_context() -> None:
    ctx = RenderContext(name="x", class_name="XSkill")
    inp = RenderInput(template="skill/skill.py.j2", context=ctx)
    assert inp.template == "skill/skill.py.j2"
    assert isinstance(inp.context, RenderContext)
    assert inp.context.class_name == "XSkill"


def test_render_output_holds_content() -> None:
    assert RenderOutput(content="hello").content == "hello"


def test_renders_class_name_into_skill_template() -> None:
    skill = TemplateRendererSkill()
    ctx = RenderContext(name="web_search", class_name="WebSearchSkill")
    out = skill.run(RenderInput(template="skill/skill.py.j2", context=ctx))
    assert "class WebSearchSkill(BaseSkill" in out.content


def test_environment_uses_strict_undefined() -> None:
    skill = TemplateRendererSkill()
    assert skill._env.undefined is StrictUndefined


def test_unknown_template_raises() -> None:
    skill = TemplateRendererSkill()
    ctx = RenderContext(name="x", class_name="XSkill")
    with pytest.raises(TemplateError):
        skill.run(RenderInput(template="nope/missing.j2", context=ctx))


def test_renders_agent_class_and_provider() -> None:
    skill = TemplateRendererSkill()
    ctx = RenderContext(name="researcher", class_name="ResearcherAgent")
    agent_py = skill.run(RenderInput(template="agent/agent.py.j2", context=ctx))
    assert "class ResearcherAgent(BaseAgent" in agent_py.content
    config = skill.run(RenderInput(template="agent/config.yaml.j2", context=ctx))
    assert "provider: anthropic/claude-sonnet-4-6" in config.content
