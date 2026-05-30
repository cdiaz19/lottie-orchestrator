from __future__ import annotations

from pathlib import Path

import pytest
import typer

from lottie.project.config import (
    AgentConfig,
    LottieConfig,
    find_project_root,
    load_agent_config,
    load_lottie_config,
)

_LOTTIE_YAML = """\
project: demo
providers:
  default: anthropic/claude-sonnet-4-6
  fallback: openai/gpt-4o
policies:
  - base
registry:
  agents: agents/
  skills: skills/
"""

_AGENT_YAML = """\
provider: anthropic/claude-sonnet-4-6
model_params:
  temperature: 0.3
capabilities: []
policies:
  - base
memory:
  enabled: false
  namespace: researcher
"""


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text(_LOTTIE_YAML, encoding="utf-8")
    nested = tmp_path / "agents" / "researcher"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path


def test_find_project_root_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter):
        find_project_root(tmp_path)


def test_load_lottie_config(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text(_LOTTIE_YAML, encoding="utf-8")
    cfg = load_lottie_config(tmp_path)
    assert isinstance(cfg, LottieConfig)
    assert cfg.project == "demo"
    assert cfg.providers.default == "anthropic/claude-sonnet-4-6"
    assert cfg.providers.fallback == "openai/gpt-4o"
    assert cfg.policies == ["base"]


def test_load_agent_config_ignores_extra(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(_AGENT_YAML, encoding="utf-8")
    cfg = load_agent_config(tmp_path)
    assert isinstance(cfg, AgentConfig)
    assert cfg.provider == "anthropic/claude-sonnet-4-6"
    assert cfg.model_params == {"temperature": 0.3}
    # `memory` is not a field — extra='ignore' must not raise.


def test_load_lottie_config_malformed_raises(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text("project: [unclosed", encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        load_lottie_config(tmp_path)
