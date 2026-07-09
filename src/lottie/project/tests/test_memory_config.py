from lottie.project.config import AgentConfig, MemoryConfig


def test_agent_config_memory_defaults_off() -> None:
    cfg = AgentConfig(provider="mock")
    assert isinstance(cfg.memory, MemoryConfig)
    assert cfg.memory.enabled is False
    assert cfg.memory.backend == "sqlite"


def test_agent_config_memory_from_dict() -> None:
    cfg = AgentConfig.model_validate(
        {"provider": "mock", "memory": {"enabled": True, "backend": "mock"}}
    )
    assert cfg.memory.enabled is True
    assert cfg.memory.backend == "mock"
