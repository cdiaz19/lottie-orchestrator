from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from lottie.llm import LLMProvider, LLMResponse, Message, TokenUsage


def test_token_usage_defaults_to_zero() -> None:
    usage = TokenUsage()
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_token_usage_accepts_values() -> None:
    usage = TokenUsage(input_tokens=12, output_tokens=34)
    assert usage.input_tokens == 12
    assert usage.output_tokens == 34


def test_message_rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        Message(role="root", content="hi")


def test_llm_response_requires_core_fields_and_defaults_cost() -> None:
    resp = LLMResponse(content="hello", usage=TokenUsage(), model="m")
    assert resp.content == "hello"
    assert resp.cost_usd == 0.0


def test_llm_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_concrete_provider_returns_response() -> None:
    class EchoProvider(LLMProvider):
        @property
        def model(self) -> str:
            return "echo-1"

        def complete(
            self,
            messages: list[Message],
            model_params: Mapping[str, object] | None = None,
        ) -> LLMResponse:
            return LLMResponse(
                content=messages[-1].content,
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                model=self.model,
            )

    provider = EchoProvider()
    resp = provider.complete([Message(role="user", content="ping")])
    assert resp.content == "ping"
    assert resp.model == "echo-1"


def test_build_provider_returns_litellm_provider() -> None:
    from lottie.llm import LiteLLMProvider, build_provider

    provider = build_provider("anthropic/claude-sonnet-4-6")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "anthropic/claude-sonnet-4-6"
