"""Trajectory persistence config: opt-in, and absent from every existing project."""

from __future__ import annotations

from lottie.project.config import AgentConfig, MemoryConfig, TrajectoryConfig


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
