"""OpenAI chat-completions request model + pure response/error builders.

Pure pydantic + stdlib — NO Starlette import — so these are unit-testable without
the [api] extra. The transport (openai_app) imports these and wraps them in HTTP.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TypedDict

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    # Tolerate the many sampling params real OpenAI clients send; we ignore them
    # (the agent owns its provider config this slice).
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def last_user_message(req: ChatCompletionRequest) -> str | None:
    """Content of the final `user`-role message, or None if there is none."""
    for message in reversed(req.messages):
        if message.role == "user":
            return message.content
    return None


# ---------------------------------------------------------------------------
# TypedDicts for the response/error shapes — lets mypy verify callers.
# ---------------------------------------------------------------------------


class _MessageDict(TypedDict):
    role: str
    content: str


class _ChoiceDict(TypedDict):
    index: int
    message: _MessageDict
    finish_reason: str


class _UsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _LottieExtDict(TypedDict):
    latency_ms: float
    cost_usd: float
    status: str


class ChatCompletionDict(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[_ChoiceDict]
    usage: _UsageDict
    lottie: _LottieExtDict


class _ErrorBodyDict(TypedDict):
    message: str
    type: str
    code: str | None
    param: None


class ErrorDict(TypedDict):
    error: _ErrorBodyDict


def chat_completion_dict(
    *,
    agent: str,
    content: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    cost_usd: float,
    status: str,
    finish_reason: str = "stop",
) -> ChatCompletionDict:
    """Build an OpenAI chat.completion response body (plus a `lottie` extension)."""
    return ChatCompletionDict(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=agent,
        choices=[
            _ChoiceDict(
                index=0,
                message=_MessageDict(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=_UsageDict(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        lottie=_LottieExtDict(
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            status=status,
        ),
    )


def error_dict(
    message: str, *, type_: str, code: str | None = None
) -> ErrorDict:
    """Build an OpenAI error envelope. `message` must never echo a payload."""
    return ErrorDict(
        error=_ErrorBodyDict(message=message, type=type_, code=code, param=None)
    )


class ChatChunkEncoder:
    """Builds OpenAI chat.completion.chunk SSE events.

    A single encoder instance shares one id/created timestamp across all events,
    which is required for real-token streaming.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())

    def _event(self, delta: dict[str, object], finish: str | None) -> str:
        body: dict[str, object] = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n"

    def role(self) -> str:
        return self._event({"role": "assistant"}, None)

    def content(self, text: str) -> str:
        return self._event({"content": text}, None)

    def finish(self, reason: str) -> str:
        return self._event({}, reason)

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"


def chat_completion_chunks(
    *, agent: str, content: str, finish_reason: str
) -> list[str]:
    """OpenAI SSE lines for a COMPLETED chat response (format-level).

    Emits: role delta, a content delta (only when `content` is non-empty — a withheld
    output has none), the finish delta, then [DONE].
    """
    enc = ChatChunkEncoder(agent)
    lines = [enc.role()]
    if content:
        lines.append(enc.content(content))
    lines.append(enc.finish(finish_reason))
    lines.append(enc.done())
    return lines
