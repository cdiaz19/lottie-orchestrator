from __future__ import annotations

from lottie.serve.openai_schema import (
    ChatCompletionRequest,
    chat_completion_dict,
    error_dict,
    last_user_message,
)


def test_request_parses_and_ignores_extra() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "echo",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.1,  # unmodeled -> ignored, not an error
        }
    )
    assert req.model == "echo"
    assert req.stream is False
    assert req.messages[0].content == "hi"


def test_last_user_message_picks_final_user() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "echo",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
        }
    )
    assert last_user_message(req) == "second"


def test_last_user_message_none_when_absent() -> None:
    req = ChatCompletionRequest.model_validate(
        {"model": "echo", "messages": [{"role": "system", "content": "x"}]}
    )
    assert last_user_message(req) is None


def test_chat_completion_dict_shape() -> None:
    body = chat_completion_dict(
        agent="echo",
        content="hello world",
        input_tokens=3,
        output_tokens=2,
        latency_ms=12.0,
        cost_usd=0.0,
        status="complete",
    )
    assert body["object"] == "chat.completion"
    assert body["model"] == "echo"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello world"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert body["lottie"] == {"latency_ms": 12.0, "cost_usd": 0.0, "status": "complete"}


def test_chat_completion_dict_content_filter_finish() -> None:
    body = chat_completion_dict(
        agent="echo",
        content="",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        cost_usd=0.0,
        status="complete",
        finish_reason="content_filter",
    )
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert body["choices"][0]["message"]["content"] == ""


def test_error_dict_shape() -> None:
    err = error_dict("bad", type_="invalid_request_error", code="model_not_found")
    assert err == {
        "error": {
            "message": "bad",
            "type": "invalid_request_error",
            "code": "model_not_found",
            "param": None,
        }
    }
