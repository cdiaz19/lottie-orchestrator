from collections.abc import Mapping

import pytest

from lottie.llm import LLMProvider, LLMResponse, Message, MockLLMProvider


def test_mock_is_a_concrete_llm_provider() -> None:
    provider = MockLLMProvider(responses=["hi"])
    assert isinstance(provider, LLMProvider)


def test_mock_returns_configured_responses_in_order() -> None:
    provider = MockLLMProvider(responses=["first", "second"])
    r1 = provider.complete([Message(role="user", content="a")])
    r2 = provider.complete([Message(role="user", content="b")])
    assert isinstance(r1, LLMResponse)
    assert r1.content == "first"
    assert r2.content == "second"


def test_mock_response_carries_model_identifier() -> None:
    provider = MockLLMProvider(responses=["x"], model="mock/test-7")
    resp = provider.complete([Message(role="user", content="a")])
    assert provider.model == "mock/test-7"
    assert resp.model == "mock/test-7"


def test_mock_model_has_default() -> None:
    provider = MockLLMProvider(responses=["x"])
    assert provider.model == "mock/mock-model"


def test_mock_records_received_messages() -> None:
    provider = MockLLMProvider(responses=["x", "y"])
    msgs1 = [Message(role="user", content="one")]
    msgs2 = [Message(role="system", content="sys"), Message(role="user", content="two")]
    provider.complete(msgs1)
    provider.complete(msgs2)
    assert provider.calls == [msgs1, msgs2]


def test_mock_raises_when_responses_exhausted() -> None:
    provider = MockLLMProvider(responses=["only"])
    provider.complete([Message(role="user", content="a")])
    with pytest.raises(RuntimeError, match="exhausted"):
        provider.complete([Message(role="user", content="b")])


def test_mock_rejects_empty_responses() -> None:
    with pytest.raises(ValueError, match="at least one response"):
        MockLLMProvider(responses=[])


def test_mock_accepts_model_params_without_error() -> None:
    provider = MockLLMProvider(responses=["x"])
    params: Mapping[str, object] = {"temperature": 0.1}
    resp = provider.complete([Message(role="user", content="a")], model_params=params)
    assert resp.content == "x"


def test_mock_stream_reconstructs_response() -> None:
    p = MockLLMProvider(["the launch post"])
    deltas = list(p.stream([Message(role="user", content="x")]))
    assert len(deltas) > 1  # multiple deltas for a multi-word response
    assert "".join(deltas) == "the launch post"  # reconstructs exactly


def test_mock_stream_single_token_one_delta() -> None:
    p = MockLLMProvider(["hi"])
    assert list(p.stream([Message(role="user", content="x")])) == ["hi"]


def test_mock_stream_advances_queue_and_records_call() -> None:
    p = MockLLMProvider(["first", "second"])
    list(p.stream([Message(role="user", content="a")]))
    assert len(p.calls) == 1
    # the queue advanced — a following complete() gets the SECOND response
    assert p.complete([Message(role="user", content="b")]).content == "second"


def test_mock_stream_raises_when_exhausted() -> None:
    p = MockLLMProvider(["only"])
    list(p.stream([Message(role="user", content="a")]))
    with pytest.raises(RuntimeError):
        list(p.stream([Message(role="user", content="b")]))
