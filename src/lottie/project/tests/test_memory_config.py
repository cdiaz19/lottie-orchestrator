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


def test_memory_config_recall_defaults_off() -> None:
    from lottie.project.config import AgentConfig, RecallConfig

    cfg = AgentConfig(provider="mock")
    assert isinstance(cfg.memory.recall, RecallConfig)
    assert cfg.memory.recall.enabled is False
    assert cfg.memory.recall.limit == 5
    assert cfg.memory.namespace is None


def test_memory_config_recall_from_dict() -> None:
    from lottie.project.config import AgentConfig

    cfg = AgentConfig.model_validate(
        {
            "provider": "mock",
            "memory": {
                "enabled": True,
                "namespace": "lessons",
                "recall": {"enabled": True, "limit": 3},
            },
        }
    )
    assert cfg.memory.namespace == "lessons"
    assert cfg.memory.recall.enabled is True
    assert cfg.memory.recall.limit == 3


def test_memory_config_reflect_defaults_off() -> None:
    from lottie.project.config import AgentConfig, ReflectConfig

    cfg = AgentConfig(provider="mock")
    assert isinstance(cfg.memory.reflect, ReflectConfig)
    assert cfg.memory.reflect.enabled is False


def test_memory_config_reflect_from_dict() -> None:
    from lottie.project.config import AgentConfig

    cfg = AgentConfig.model_validate(
        {"provider": "mock", "memory": {"enabled": True, "reflect": {"enabled": True}}}
    )
    assert cfg.memory.reflect.enabled is True
