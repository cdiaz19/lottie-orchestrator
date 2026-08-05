"""`lottie modules` and the S6 config block.

The chain is otherwise invisible: an operator can read config.yaml and *infer* what is
mounted, and inference is exactly how a disabled security gate goes unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lottie.cli.app import app
from lottie.core.middleware import KNOWN_MODULES, build_chain
from lottie.llm import MockLLMProvider
from lottie.project.config import AgentConfig
from lottie.project.discovery import instantiate_agent, load_agent_class
from lottie.runtime.registry import ModuleConflictError

runner = CliRunner()

_SCHEMA = """
from __future__ import annotations
from pydantic import BaseModel


class ProbeInput(BaseModel):
    task: str


class ProbeOutput(BaseModel):
    answer: str
"""

_AGENT = """
from __future__ import annotations
from lottie.core import BaseAgent

from .schema import ProbeInput, ProbeOutput


class ProbeAgent(BaseAgent[ProbeInput, ProbeOutput]):
    def _execute(self, data: ProbeInput) -> ProbeOutput:
        return ProbeOutput(answer="ok")
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "lottie.yaml").write_text("name: demo\n")
    agent_dir = tmp_path / "agents" / "probe"
    agent_dir.mkdir(parents=True)
    (tmp_path / "agents" / "__init__.py").write_text("")
    (agent_dir / "__init__.py").write_text("")
    (agent_dir / "agent.py").write_text(_AGENT)
    (agent_dir / "schema.py").write_text(_SCHEMA)
    (agent_dir / "config.yaml").write_text(yaml.safe_dump({"provider": "mock/sim"}))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _set_config(project: Path, cfg: dict[str, object]) -> None:
    (project / "agents" / "probe" / "config.yaml").write_text(
        yaml.safe_dump({"provider": "mock/sim", **cfg})
    )


def _agent(**cfg: object):  # type: ignore[no-untyped-def]
    root = Path.cwd()
    return instantiate_agent(
        load_agent_class(root, "probe"),
        llm=MockLLMProvider(responses=["ok"]),
        root=root,
        config=AgentConfig.model_validate({"provider": "mock/sim", **cfg}),
        enable_benchmarks=False,
    )


class TestModulesCommand:
    def test_lists_the_mounted_chain(self, project: Path) -> None:
        result = runner.invoke(app, ["modules", "probe"])
        assert result.exit_code == 0, result.output
        assert "11 module(s) mounted" in result.output

    def test_shows_modules_in_chain_order(self, project: Path) -> None:
        out = runner.invoke(app, ["modules", "probe"]).output
        assert out.index("security_input") < out.index("policy") < out.index("capability")

    def test_a_disabled_module_is_shown_as_disabled(self, project: Path) -> None:
        _set_config(project, {"modules": {"recall": {"enabled": False}}})
        out = runner.invoke(app, ["modules", "probe"]).output
        assert "disabled" in out and "10 module(s) mounted" in out

    def test_an_unknown_module_name_is_flagged(self, project: Path) -> None:
        # A typo here would otherwise silently do nothing.
        _set_config(project, {"modules": {"recal": {"enabled": False}}})
        out = runner.invoke(app, ["modules", "probe"]).output
        assert "unknown module name" in out and "recal" in out

    def test_listing_every_agent_needs_no_argument(self, project: Path) -> None:
        assert runner.invoke(app, ["modules"]).exit_code == 0

    def test_an_unknown_agent_is_refused(self, project: Path) -> None:
        assert runner.invoke(app, ["modules", "ghost"]).exit_code != 0


class TestDisabling:
    def test_a_disabled_module_is_not_constructed(self, project: Path) -> None:
        agent = _agent(modules={"recall": {"enabled": False}})
        assert "recall" not in agent.mounted_modules()

    def test_the_rest_of_the_chain_is_untouched(self, project: Path) -> None:
        agent = _agent(modules={"recall": {"enabled": False}})
        assert len(agent.mounted_modules()) == 10

    def test_nothing_is_disabled_by_default(self, project: Path) -> None:
        assert len(_agent().mounted_modules()) == 11

    def test_an_enabled_true_entry_changes_nothing(self, project: Path) -> None:
        agent = _agent(modules={"recall": {"enabled": True}})
        assert "recall" in agent.mounted_modules()

    def test_a_disabled_agent_still_runs(self, project: Path) -> None:
        from lottie.project.discovery import load_input_model

        agent = _agent(modules={"recall": {"enabled": False}})
        model = load_input_model(Path.cwd(), "probe")
        assert agent.run(model(task="t")).answer == "ok"


class TestKnownModules:
    def test_known_modules_matches_the_real_chain(self, project: Path) -> None:
        # If a module is added without updating KNOWN_MODULES, `doctor` would start
        # reporting a legitimate config line as a typo.
        assert set(KNOWN_MODULES) == {m.name for m in build_chain(_agent())}


class TestOrderConflicts:
    def test_a_duplicate_order_is_rejected_at_composition(self, project: Path) -> None:
        """Two modules claiming one slot is ambiguous about who owns it, and a plugin
        must never be able to silently displace a security gate."""
        agent = _agent()
        chain = build_chain(agent)
        clash = type(chain[0])
        original = clash.order
        try:
            clash.order = chain[1].order
            with pytest.raises(ModuleConflictError, match="already held by"):
                build_chain(agent)
        finally:
            clash.order = original


class TestBrokenAgent:
    def test_an_agent_that_cannot_load_is_reported_not_fatal(self, project: Path) -> None:
        """One broken agent must not hide the others.

        `lottie modules` is a diagnostic; if it aborted on the first agent that fails to
        import, it would be least useful exactly when something is wrong.
        """
        broken = project / "agents" / "broken"
        broken.mkdir(parents=True)
        (broken / "__init__.py").write_text("")
        (broken / "agent.py").write_text("raise RuntimeError('cannot import me')\n")
        (broken / "schema.py").write_text(_SCHEMA)
        (broken / "config.yaml").write_text(yaml.safe_dump({"provider": "mock/sim"}))

        result = runner.invoke(app, ["modules"])
        assert result.exit_code == 0, result.output
        assert "cannot inspect" in result.output
        # …and the healthy agent is still reported.
        assert "11 module(s) mounted" in result.output
