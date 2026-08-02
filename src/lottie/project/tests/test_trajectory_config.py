"""Trajectory persistence config: opt-in, and absent from every existing project."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.project.config import AgentConfig, MemoryConfig, TrajectoryConfig
from lottie.project.discovery import instantiate_agent


def test_disabled_by_default() -> None:
    assert TrajectoryConfig().enabled is False


def test_default_size_bound() -> None:
    assert TrajectoryConfig().max_chars == 4000


def test_memory_config_carries_a_default_instance() -> None:
    assert MemoryConfig().trajectory.enabled is False


def test_an_existing_config_without_the_block_still_parses() -> None:
    # Every project on disk today omits `trajectory:` entirely.
    cfg = AgentConfig(provider="mock", memory={"enabled": True})
    assert cfg.memory.trajectory.enabled is False


def test_the_block_parses_when_present() -> None:
    cfg = AgentConfig(
        provider="mock",
        memory={"enabled": True, "trajectory": {"enabled": True, "max_chars": 100}},
    )
    assert cfg.memory.trajectory.enabled is True
    assert cfg.memory.trajectory.max_chars == 100


# --- instantiate_agent wiring: closes the config -> runtime path ---


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(answer=data.task)


def _instantiate(tmp_path: Path, cfg: AgentConfig) -> BaseAgent[_In, _Out]:
    return instantiate_agent(  # type: ignore[return-value]
        _Agent,  # type: ignore[arg-type]
        llm=MockLLMProvider(responses=["ok"]),
        root=tmp_path,
        config=cfg,
        enable_benchmarks=False,
    )


def test_memory_disabled_leaves_trajectory_off(tmp_path: Path) -> None:
    cfg = AgentConfig(
        provider="mock", memory={"enabled": False, "trajectory": {"enabled": True}}
    )
    assert _instantiate(tmp_path, cfg)._trajectory_enabled is False


def test_trajectory_enabled_turns_the_hook_on(tmp_path: Path) -> None:
    cfg = AgentConfig(
        provider="mock", memory={"enabled": True, "trajectory": {"enabled": True}}
    )
    assert _instantiate(tmp_path, cfg)._trajectory_enabled is True


def test_namespace_defaults_to_the_agent_name(tmp_path: Path) -> None:
    cfg = AgentConfig(
        provider="mock", memory={"enabled": True, "trajectory": {"enabled": True}}
    )
    agent = _instantiate(tmp_path, cfg)
    assert agent._trajectory_namespace == agent.name


def test_explicit_namespace_wins(tmp_path: Path) -> None:
    cfg = AgentConfig(
        provider="mock",
        memory={"enabled": True, "namespace": "shared", "trajectory": {"enabled": True}},
    )
    assert _instantiate(tmp_path, cfg)._trajectory_namespace == "shared"


def test_max_chars_is_threaded_through(tmp_path: Path) -> None:
    cfg = AgentConfig(
        provider="mock",
        memory={"enabled": True, "trajectory": {"enabled": True, "max_chars": 42}},
    )
    assert _instantiate(tmp_path, cfg)._trajectory_max_chars == 42


# --- harness.compaction (V2 S5a) ---------------------------------------------


def test_compaction_disabled_by_default() -> None:
    assert AgentConfig(provider="mock").harness.compaction.enabled is False


def test_compaction_defaults() -> None:
    c = AgentConfig(provider="mock").harness.compaction
    assert c.max_context_tokens == 8000 and c.keep_recent == 6


def test_a_config_without_a_harness_block_still_parses() -> None:
    # Every project on disk today omits `harness:` entirely.
    cfg = AgentConfig(provider="mock", memory={"enabled": True})
    assert cfg.harness.compaction.enabled is False


def test_compaction_is_wired_when_enabled(tmp_path: Path) -> None:
    cfg = AgentConfig(
        provider="mock",
        harness={"compaction": {"enabled": True, "max_context_tokens": 500, "keep_recent": 3}},
    )
    agent = _instantiate(tmp_path, cfg)
    assert agent._compaction_enabled is True
    assert agent._max_context_tokens == 500
    assert agent._keep_recent == 3


def test_compaction_is_independent_of_memory(tmp_path: Path) -> None:
    # A run can outgrow its window whether or not the agent learns.
    cfg = AgentConfig(
        provider="mock",
        memory={"enabled": False},
        harness={"compaction": {"enabled": True}},
    )
    assert _instantiate(tmp_path, cfg)._compaction_enabled is True
