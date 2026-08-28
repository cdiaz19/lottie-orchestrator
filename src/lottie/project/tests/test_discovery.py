from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from lottie.cli import app
from lottie.core import BaseAgent
from lottie.project.discovery import (
    discover_agents,
    discover_skills,
    load_agent_class,
    load_input_model,
    load_schema_models,
    load_system_prompt,
    required_fields,
)

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0
    return demo


def test_discover_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    agents = discover_agents(demo)
    # `lottie init` ships a runnable hello agent, so it is always present alongside
    # any agent the project later creates.
    assert [a.name for a in agents] == ["hello", "researcher"]
    assert all(a.kind == "agent" for a in agents)
    assert all(a.provider == "anthropic/claude-sonnet-4-6" for a in agents)


def test_discover_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    skills = discover_skills(demo)
    assert [s.name for s in skills] == ["cleaner"]
    assert skills[0].kind == "skill"
    assert skills[0].provider is None


def test_discover_empty_when_no_dir(tmp_path: Path) -> None:
    assert discover_agents(tmp_path) == []
    assert discover_skills(tmp_path) == []


def test_load_agent_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    cls = load_agent_class(demo, "researcher")
    assert issubclass(cls, BaseAgent)
    assert cls.__name__ == "ResearcherAgent"


def test_load_agent_class_zero_subclasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "agents" / "researcher" / "agent.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        load_agent_class(demo, "researcher")


def test_load_agent_class_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    with pytest.raises(typer.BadParameter):
        load_agent_class(demo, "nope")


def test_load_input_model_and_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    model = load_input_model(demo, "researcher")
    assert required_fields(model) == ["query"]


def test_load_input_model_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "agents" / "researcher" / "schema.py").write_text("x = 1\n", encoding="utf-8")
    import typer

    with pytest.raises(typer.BadParameter):
        load_input_model(demo, "researcher")


def test_load_schema_models_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    in_model, out_model = load_schema_models(demo, "agent", "researcher")
    assert in_model.__name__ == "ResearcherAgentInput"
    assert out_model.__name__ == "ResearcherAgentOutput"
    assert "query" in in_model.model_fields


def test_load_schema_models_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    in_model, out_model = load_schema_models(demo, "skill", "cleaner")
    assert in_model.__name__ == "CleanerSkillInput"
    assert out_model.__name__ == "CleanerSkillOutput"
    assert "text" in in_model.model_fields


def test_load_schema_models_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    with pytest.raises(typer.BadParameter):
        load_schema_models(demo, "skill", "nope")


def test_load_system_prompt_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    prompt = load_system_prompt(demo, "researcher")
    assert prompt is not None
    assert "ResearcherAgent" in prompt


def test_load_system_prompt_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "agents" / "researcher" / "prompts.py").unlink()
    assert load_system_prompt(demo, "researcher") is None


def test_load_system_prompt_broken_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "agents" / "researcher" / "prompts.py").write_text(
        "import nonexistent_module_xyz\n", encoding="utf-8"
    )
    with pytest.raises(typer.BadParameter):
        load_system_prompt(demo, "researcher")


# --- E5: resolve_provider honours the project fallback -----------------------


def test_resolve_provider_honours_the_configured_fallback(tmp_path: Path) -> None:
    from lottie.llm.routing import RoutedProvider
    from lottie.project.discovery import resolve_provider

    (tmp_path / "lottie.yaml").write_text(
        "project: p\nproviders:\n  default: a/primary\n  fallback: b/secondary\n"
    )
    routed = resolve_provider(tmp_path, "a/primary")
    assert isinstance(routed, RoutedProvider)
    assert routed.chain == ["a/primary", "b/secondary"]


def test_resolve_provider_without_a_fallback_is_not_wrapped(tmp_path: Path) -> None:
    from lottie.llm.routing import RoutedProvider
    from lottie.project.discovery import resolve_provider

    (tmp_path / "lottie.yaml").write_text("project: p\nproviders:\n  default: a/primary\n")
    assert not isinstance(resolve_provider(tmp_path, "a/primary"), RoutedProvider)


def test_a_broken_lottie_yaml_degrades_to_no_fallback(tmp_path: Path) -> None:
    """Losing the fallback is worse-but-working; refusing to build a provider is a dead run.

    A caller that reached here already has a project, so a malformed lottie.yaml is
    `lottie status`'s problem to report — not a reason to make the run impossible.
    """
    from lottie.llm.routing import RoutedProvider
    from lottie.project.discovery import resolve_provider

    (tmp_path / "lottie.yaml").write_text("{{{ not yaml at all")
    provider = resolve_provider(tmp_path, "a/primary")
    assert provider.model == "a/primary"
    assert not isinstance(provider, RoutedProvider)


def test_the_default_notifier_warns_so_a_fallback_is_never_silent() -> None:
    """A silent fallback is the dangerous kind — the run succeeds on a different model
    at a different price and nothing says so."""
    import pytest

    from lottie.project.discovery import _warn_fallback

    with pytest.warns(UserWarning, match="falling back"):
        _warn_fallback("a/primary", "b/secondary", RuntimeError("429"))
