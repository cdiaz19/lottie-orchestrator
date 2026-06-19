from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lottie.llm import LiteLLMProvider, LLMProvider, Message


def _fake_response(content: str, prompt: int, completion: int) -> Any:
    """Mimic the shape of a litellm ModelResponse (attribute access)."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_litellm_is_a_concrete_llm_provider() -> None:
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6")
    assert isinstance(provider, LLMProvider)


def test_model_property_reflects_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o")
    assert provider.model == "openai/gpt-4o"


def test_complete_converts_messages_and_passes_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_response("ok", 5, 7)

    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion", fake_completion)
    monkeypatch.setattr(
        "lottie.llm.litellm_provider.litellm.completion_cost", lambda **_: 0.0
    )

    provider = LiteLLMProvider(model="openai/gpt-4o")
    provider.complete(
        [Message(role="system", content="sys"), Message(role="user", content="hi")]
    )

    assert captured["model"] == "openai/gpt-4o"
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_complete_normalizes_content_usage_and_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lottie.llm.litellm_provider.litellm.completion",
        lambda **_: _fake_response("the answer", 11, 22),
    )
    monkeypatch.setattr(
        "lottie.llm.litellm_provider.litellm.completion_cost", lambda **_: 0.0042
    )

    provider = LiteLLMProvider(model="openai/gpt-4o")
    resp = provider.complete([Message(role="user", content="q")])

    assert resp.content == "the answer"
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 22
    assert resp.cost_usd == 0.0042
    assert resp.model == "openai/gpt-4o"


def test_complete_forwards_model_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _fake_response("ok", 1, 1)

    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion", fake_completion)
    monkeypatch.setattr(
        "lottie.llm.litellm_provider.litellm.completion_cost", lambda **_: 0.0
    )

    provider = LiteLLMProvider(model="openai/gpt-4o")
    provider.complete(
        [Message(role="user", content="q")],
        model_params={"temperature": 0.2, "max_tokens": 256},
    )

    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 256


def test_cost_defaults_to_zero_when_uncomputable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**_: Any) -> float:
        raise ValueError("cannot compute cost for this model")

    monkeypatch.setattr(
        "lottie.llm.litellm_provider.litellm.completion",
        lambda **_: _fake_response("ok", 1, 1),
    )
    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion_cost", boom)

    provider = LiteLLMProvider(model="local/llama")
    resp = provider.complete([Message(role="user", content="q")])
    assert resp.cost_usd == 0.0


def test_stream_yields_content_deltas_skipping_empties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _chunk(text: Any) -> Any:
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    def fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return iter([_chunk("The "), _chunk("launch "), _chunk(""), _chunk(None), _chunk("post")])

    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="openai/gpt-4o")
    deltas = list(provider.stream([Message(role="user", content="q")]))

    assert deltas == ["The ", "launch ", "post"]  # empties + None skipped
    assert captured["stream"] is True
    assert captured["model"] == "openai/gpt-4o"
    assert captured["messages"] == [{"role": "user", "content": "q"}]
