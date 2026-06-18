from __future__ import annotations

from lottie.project.config import AgentConfig


def test_chat_block_parses() -> None:
    cfg = AgentConfig.model_validate(
        {"provider": "anthropic/x", "chat": {"input_field": "query", "output_field": "result"}}
    )
    assert cfg.chat is not None
    assert cfg.chat.input_field == "query"
    assert cfg.chat.output_field == "result"


def test_chat_absent_defaults_none() -> None:
    cfg = AgentConfig.model_validate({"provider": "anthropic/x"})
    assert cfg.chat is None
