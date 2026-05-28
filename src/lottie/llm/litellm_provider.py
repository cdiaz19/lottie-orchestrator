"""litellm-backed `LLMProvider`.

litellm is Lottie's universal proxy: one interface to Claude, GPT-4o, local
models, etc. This adapter is the ONLY place litellm is imported — agent and
skill code always goes through the `LLMProvider` abstraction (CLAUDE.md rule 1).
"""

from __future__ import annotations

from collections.abc import Mapping

import litellm

from lottie.llm.base import LLMProvider, LLMResponse, Message, TokenUsage


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

    @staticmethod
    def _cost(response: object) -> float:
        """litellm can't price every model (e.g. local) — fall back to 0."""
        try:
            return float(litellm.completion_cost(completion_response=response))
        except Exception:
            return 0.0
