from collections.abc import Callable

from lottie.llm.base import LLMProvider, LLMResponse, Message, Role, StreamResult, TokenUsage
from lottie.llm.litellm_provider import LiteLLMProvider
from lottie.llm.mock import MockLLMProvider
from lottie.llm.routing import RoutedProvider, is_transient


def build_provider(
    model: str,
    *,
    fallback: str | None = None,
    on_fallback: Callable[[str, str, BaseException], None] | None = None,
) -> LLMProvider:
    """Construct the LLM provider for a model id, with an optional fallback (E5).

    Single construction point used by the CLI; tests monkeypatch
    ``litellm.completion`` beneath the returned provider.

    With no `fallback` this returns a plain `LiteLLMProvider`, exactly as before — a
    project that has not configured one pays nothing for the feature. When a fallback IS
    configured, the chain advances only on transient failures; see `llm.routing`.
    """
    primary = LiteLLMProvider(model)
    if fallback is None or fallback == model:
        return primary
    return RoutedProvider([primary, LiteLLMProvider(fallback)], on_fallback=on_fallback)


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LiteLLMProvider",
    "Message",
    "MockLLMProvider",
    "Role",
    "StreamResult",
    "TokenUsage",
    "RoutedProvider",
    "build_provider",
    "is_transient",
]
