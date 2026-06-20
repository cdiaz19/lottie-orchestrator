from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from lottie.llm import LLMProvider, LLMResponse, Message, TokenUsage
from lottie.llm.base import StreamResult


def _drain(gen: object) -> tuple[list[str], object]:
    out: list[str] = []
    try:
        while True:
            out.append(next(gen))  # type: ignore[arg-type]
    except StopIteration as stop:
        return out, stop.value


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


def test_stream_default_yields_complete_content_once() -> None:
    class OneShot(LLMProvider):
        @property
        def model(self) -> str:
            return "one/shot"

        def complete(
            self,
            messages: list[Message],
            model_params: Mapping[str, object] | None = None,
        ) -> LLMResponse:
            return LLMResponse(content="hello world", model=self.model)

    deltas = list(OneShot().stream([Message(role="user", content="hi")]))
    assert deltas == ["hello world"]  # default = one delta over complete()


class _CompleteOnly(LLMProvider):
    @property
    def model(self) -> str:
        return "stub/model"

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content="hello world", usage=TokenUsage(input_tokens=4, output_tokens=2),
            model="stub/model", cost_usd=0.3,
        )


def test_stream_complete_default_one_shot_yields_content_and_returns_usage() -> None:
    deltas, result = _drain(_CompleteOnly().stream_complete([Message(role="user", content="x")]))
    assert deltas == ["hello world"]            # one delta (no incremental latency)
    assert isinstance(result, StreamResult)
    assert result.usage.input_tokens == 4 and result.usage.output_tokens == 2
    assert result.cost_usd == 0.3
