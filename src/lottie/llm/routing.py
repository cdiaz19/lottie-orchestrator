"""Provider routing — honour `providers.fallback`, visibly (E5).

`lottie.yaml` has declared `providers.fallback` since Phase 0 and nothing read it. A user
who set one got nothing, which is worse than an absent feature: the config claimed a
resilience property the runtime did not have, and the failure mode was discovering that
during an outage.

What a fallback must NOT do
---------------------------
Falling back on a content-policy refusal would launder a provider's safety decision —
the framework would quietly shop a refused request to a second model. Lottie is a governed
framework with fail-closed gates, so that is the one retry it must never make.

A bad request or a bad API key fails identically on the fallback, so retrying those only
doubles the spend before failing anyway. Only TRANSIENT failures — the ones a different
provider might genuinely survive — advance the chain.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Mapping, Sequence

from lottie.llm.base import LLMProvider, LLMResponse, Message, StreamResult

#: Exception class names that mean "this provider is momentarily unavailable".
#: Matched by name so the module never imports litellm — that stays confined to
#: `litellm_provider.py` (rule 1), and a provider adapter for any other backend can raise
#: its own equivalents without this module knowing about it.
_TRANSIENT_NAMES = frozenset(
    {
        "RateLimitError",
        "Timeout",
        "APITimeoutError",
        "APIConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIError",
    }
)

#: Names that look transient but must NEVER trigger a fallback. Listed explicitly rather
#: than relying on absence, so the intent survives someone widening `_TRANSIENT_NAMES`.
_NEVER_RETRY_NAMES = frozenset(
    {
        "ContentPolicyViolationError",  # laundering a safety refusal
        "AuthenticationError",  # misconfiguration; fails the same everywhere
        "PermissionDeniedError",
        "BadRequestError",  # deterministic; the fallback fails identically
        "NotFoundError",
    }
)


def is_transient(exc: BaseException) -> bool:
    """True when `exc` is worth retrying on a different provider.

    Defaults to **False**. An unrecognised exception is treated as not-retryable, so a new
    error type introduced by a provider SDK cannot silently widen the fallback surface —
    the safe direction when the cost of being wrong is spending twice or evading a refusal.
    """
    name = type(exc).__name__
    if name in _NEVER_RETRY_NAMES:
        return False
    return name in _TRANSIENT_NAMES


class RoutedProvider(LLMProvider):
    """An ordered chain of providers, advancing only on transient failures.

    `model` reports whichever provider served the LAST call, so `agent.provider` — which
    feeds the audit record — names what actually ran rather than what was intended.
    """

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        on_fallback: Callable[[str, str, BaseException], None] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("RoutedProvider needs at least one provider")
        self._providers = list(providers)
        self._on_fallback = on_fallback
        self._active = self._providers[0]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def chain(self) -> list[str]:
        """Model ids in routing order — the read model behind `lottie doctor`."""
        return [p.model for p in self._providers]

    def _notify(self, failed: LLMProvider, nxt: LLMProvider, exc: BaseException) -> None:
        if self._on_fallback is None:
            return
        # Observability must never be the thing that breaks a fallback.
        with contextlib.suppress(Exception):
            self._on_fallback(failed.model, nxt.model, exc)

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        last: BaseException | None = None
        for index, provider in enumerate(self._providers):
            try:
                response = provider.complete(messages, model_params)
            except Exception as exc:
                if not is_transient(exc) or index == len(self._providers) - 1:
                    raise
                last = exc
                self._notify(provider, self._providers[index + 1], exc)
                continue
            self._active = provider
            return response
        # Unreachable by construction: the chain is non-empty (constructor guards it) and
        # every iteration either returns or re-raises on the last index.
        raise last if last is not None else RuntimeError(  # pragma: no cover
            "no provider produced a response"
        )

    def stream_complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Generator[str, None, StreamResult]:
        """Stream, falling back ONLY before the first delta.

        Once bytes have reached the caller, switching providers would splice a response
        from two different models together — a silently corrupt answer, which is worse
        than a clean failure. After the first delta the error propagates, which the
        transport already knows how to handle.

        The generator is driven explicitly rather than with `yield from` because the
        fallback decision depends on whether a delta has escaped, and `yield from` hides
        exactly that.
        """
        for index, provider in enumerate(self._providers):
            started = False
            try:
                gen = provider.stream_complete(messages, model_params)
                while True:
                    try:
                        piece = next(gen)
                    except StopIteration as stop:
                        result: StreamResult = stop.value
                        break
                    started = True
                    yield piece
            except Exception as exc:
                if started or not is_transient(exc) or index == len(self._providers) - 1:
                    raise
                self._notify(provider, self._providers[index + 1], exc)
                continue
            self._active = provider
            return result
        # Unreachable by construction — see `complete`.
        raise RuntimeError("no provider produced a stream")  # pragma: no cover
