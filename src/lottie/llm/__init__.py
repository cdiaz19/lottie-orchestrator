from lottie.llm.base import LLMProvider, LLMResponse, Message, Role, TokenUsage
from lottie.llm.litellm_provider import LiteLLMProvider
from lottie.llm.mock import MockLLMProvider


def build_provider(model: str) -> LLMProvider:
    """Construct the default LLM provider for a model id.

    Single construction point used by the CLI; tests monkeypatch
    ``litellm.completion`` beneath the returned provider.
    """
    return LiteLLMProvider(model)


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LiteLLMProvider",
    "Message",
    "MockLLMProvider",
    "Role",
    "TokenUsage",
    "build_provider",
]
