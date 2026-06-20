from __future__ import annotations

from lottie.serve.openai_schema import (
    ChatChunkEncoder,
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


def test_chat_completion_chunks_success() -> None:
    import json

    from lottie.serve.openai_schema import chat_completion_chunks

    lines = chat_completion_chunks(agent="echo", content="hi there", finish_reason="stop")
    assert len(lines) == 4  # role, content, finish, [DONE]
    assert all(line.startswith("data: ") and line.endswith("\n\n") for line in lines)
    assert lines[-1] == "data: [DONE]\n\n"

    role = json.loads(lines[0][len("data: "):])
    body = json.loads(lines[1][len("data: "):])
    finish = json.loads(lines[2][len("data: "):])
    assert role["object"] == "chat.completion.chunk"
    assert role["model"] == "echo"
    assert role["choices"][0]["delta"] == {"role": "assistant"}
    assert role["choices"][0]["finish_reason"] is None
    assert body["choices"][0]["delta"] == {"content": "hi there"}
    assert finish["choices"][0]["delta"] == {}
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert role["id"] == body["id"] == finish["id"]  # one id across chunks
    assert role["id"].startswith("chatcmpl-")


def test_chat_completion_chunks_withhold_omits_content() -> None:
    import json

    from lottie.serve.openai_schema import chat_completion_chunks

    lines = chat_completion_chunks(agent="echo", content="", finish_reason="content_filter")
    assert len(lines) == 3  # role, finish, [DONE] — NO content chunk when content is empty
    assert lines[-1] == "data: [DONE]\n\n"
    finish = json.loads(lines[1][len("data: "):])
    assert finish["choices"][0]["finish_reason"] == "content_filter"
    assert finish["choices"][0]["delta"] == {}


def test_chat_chunk_encoder_shares_id_and_shapes() -> None:
    import json

    enc = ChatChunkEncoder("echo")
    role = json.loads(enc.role()[len("data: "):])
    c1 = json.loads(enc.content("alpha\n")[len("data: "):])
    c2 = json.loads(enc.content("beta\n")[len("data: "):])
    fin = json.loads(enc.finish("stop")[len("data: "):])

    ids = {role["id"], c1["id"], c2["id"], fin["id"]}
    assert len(ids) == 1                                   # one shared id across the stream
    assert role["object"] == "chat.completion.chunk" and role["model"] == "echo"
    assert role["choices"][0]["delta"] == {"role": "assistant"}
    assert c1["choices"][0]["delta"] == {"content": "alpha\n"}
    assert c1["choices"][0]["finish_reason"] is None
    assert fin["choices"][0]["delta"] == {} and fin["choices"][0]["finish_reason"] == "stop"
    assert ChatChunkEncoder.done() == "data: [DONE]\n\n"
