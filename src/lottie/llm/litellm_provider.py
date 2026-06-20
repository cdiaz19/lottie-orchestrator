"""litellm-backed `LLMProvider`.

litellm is Lottie's universal proxy: one interface to Claude, GPT-4o, local
models, etc. This adapter is the ONLY place litellm is imported — agent and
skill code always goes through the `LLMProvider` abstraction (CLAUDE.md rule 1).
"""

from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping

import litellm

from lottie.llm.base import LLMProvider, LLMResponse, Message, StreamResult, TokenUsage


class LiteLLMProvider(LLMProvider):
    """Routes completions through litellm to any supported backend."""

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        params = dict(model_params or {})
        payload = [{"role": m.role, "content": m.content} for m in messages]
        response = litellm.completion(model=self._model, messages=payload, **params)
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            model=self._model,
            cost_usd=self._cost(response),
        )

    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        params = dict(model_params or {})
        params.pop("stream", None)  # the streaming path owns the flag; ignore a caller-set one
        payload = [{"role": m.role, "content": m.content} for m in messages]
        for chunk in litellm.completion(
            model=self._model, messages=payload, stream=True, **params
        ):
            if not chunk.choices:  # usage-only / sentinel chunk -> no delta
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def stream_complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Generator[str, None, StreamResult]:
        params = dict(model_params or {})
        params.pop("stream", None)          # streaming path owns these flags
        params.pop("stream_options", None)
        payload = [{"role": m.role, "content": m.content} for m in messages]
        usage = TokenUsage()
        cost = 0.0
        for chunk in litellm.completion(
            model=self._model, messages=payload, stream=True,
            stream_options={"include_usage": True}, **params,
        ):
            chunk_usage = getattr(chunk, "usage", None)     # usage rides a final, choice-less chunk
            if chunk_usage is not None:
                usage = TokenUsage(
                    input_tokens=chunk_usage.prompt_tokens or 0,
                    output_tokens=chunk_usage.completion_tokens or 0,
                )
                cost = self._cost(chunk) or cost            # same cost path as complete()
            if not chunk.choices:                           # usage-only chunk -> no delta
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return StreamResult(usage=usage, cost_usd=cost)

    @staticmethod
    def _cost(response: object) -> float:
        """litellm can't price every model (e.g. local) — fall back to 0."""
        try:
            return float(litellm.completion_cost(completion_response=response))
        except Exception:
            return 0.0
