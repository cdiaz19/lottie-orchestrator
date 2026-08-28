"""Provider routing: what triggers a fallback, and what must never (E5)."""

from __future__ import annotations

from collections.abc import Generator, Mapping

import pytest

from lottie.llm import build_provider
from lottie.llm.base import LLMProvider, LLMResponse, Message, StreamResult, TokenUsage
from lottie.llm.routing import RoutedProvider, is_transient


# Exceptions named after litellm's taxonomy. Classified BY NAME so this module never
# imports litellm — that stays confined to the adapter (rule 1).
class RateLimitError(Exception): ...
class APIConnectionError(Exception): ...
class ServiceUnavailableError(Exception): ...
class ContentPolicyViolationError(Exception): ...
class AuthenticationError(Exception): ...
class BadRequestError(Exception): ...
class SomethingBrandNew(Exception): ...


class _Fake(LLMProvider):
    """A provider that either answers or raises a given exception."""

    def __init__(self, name: str, *, raises: Exception | None = None,
                 deltas: list[str] | None = None) -> None:
        self._name = name
        self._raises = raises
        self._deltas = deltas if deltas is not None else [f"{name}-delta"]
        self.calls = 0

    @property
    def model(self) -> str:
        return self._name

    def complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> LLMResponse:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return LLMResponse(content=f"from {self._name}", usage=TokenUsage(), model=self._name)

    def stream_complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> Generator[str, None, StreamResult]:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        yield from self._deltas
        return StreamResult(usage=TokenUsage(), cost_usd=0.0)


class _PartialThenFails(LLMProvider):
    """Yields one delta, then raises — the mid-stream failure case."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def model(self) -> str:
        return self._name

    def complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> LLMResponse:
        raise NotImplementedError

    def stream_complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> Generator[str, None, StreamResult]:
        yield "partial"
        raise RateLimitError("died mid-stream")


def _msgs() -> list[Message]:
    return [Message(role="user", content="hi")]


class TestIsTransient:
    @pytest.mark.parametrize(
        "exc", [RateLimitError(), APIConnectionError(), ServiceUnavailableError()]
    )
    def test_availability_failures_are_transient(self, exc: Exception) -> None:
        assert is_transient(exc) is True

    def test_a_content_policy_refusal_is_NEVER_transient(self) -> None:
        """The one retry this framework must never make.

        Falling back here would shop a refused request to a second model — laundering a
        provider's safety decision through a framework that advertises fail-closed gates.
        """
        assert is_transient(ContentPolicyViolationError()) is False

    @pytest.mark.parametrize("exc", [AuthenticationError(), BadRequestError()])
    def test_deterministic_failures_are_not_transient(self, exc: Exception) -> None:
        # These fail identically on the fallback, so retrying only doubles the spend.
        assert is_transient(exc) is False

    def test_an_unknown_exception_defaults_to_not_transient(self) -> None:
        # A new provider error type must not silently widen the fallback surface.
        assert is_transient(SomethingBrandNew()) is False

    def test_a_bare_exception_is_not_transient(self) -> None:
        assert is_transient(Exception("mystery")) is False


class TestComplete:
    def test_the_primary_serves_when_healthy(self) -> None:
        primary, secondary = _Fake("a"), _Fake("b")
        out = RoutedProvider([primary, secondary]).complete(_msgs())
        assert out.content == "from a" and secondary.calls == 0

    def test_a_transient_failure_advances_to_the_fallback(self) -> None:
        primary = _Fake("a", raises=RateLimitError("429"))
        out = RoutedProvider([primary, _Fake("b")]).complete(_msgs())
        assert out.content == "from b"

    def test_a_content_policy_refusal_does_NOT_advance(self) -> None:
        primary = _Fake("a", raises=ContentPolicyViolationError("refused"))
        secondary = _Fake("b")
        with pytest.raises(ContentPolicyViolationError):
            RoutedProvider([primary, secondary]).complete(_msgs())
        assert secondary.calls == 0  # the refusal was NOT shopped elsewhere

    def test_a_bad_request_does_not_advance(self) -> None:
        secondary = _Fake("b")
        with pytest.raises(BadRequestError):
            RoutedProvider([_Fake("a", raises=BadRequestError("400")), secondary]).complete(
                _msgs()
            )
        assert secondary.calls == 0

    def test_the_last_provider_failing_transiently_still_raises(self) -> None:
        with pytest.raises(RateLimitError):
            RoutedProvider(
                [_Fake("a", raises=RateLimitError()), _Fake("b", raises=RateLimitError())]
            ).complete(_msgs())

    def test_model_reports_who_actually_served(self) -> None:
        # `agent.provider` feeds the audit record; it must name what ran, not what was
        # intended.
        routed = RoutedProvider([_Fake("a", raises=RateLimitError()), _Fake("b")])
        routed.complete(_msgs())
        assert routed.model == "b"

    def test_an_empty_chain_is_refused(self) -> None:
        with pytest.raises(ValueError):
            RoutedProvider([])


class TestObservability:
    def test_a_fallback_notifies(self) -> None:
        seen: list[tuple[str, str]] = []
        routed = RoutedProvider(
            [_Fake("a", raises=RateLimitError()), _Fake("b")],
            on_fallback=lambda frm, to, exc: seen.append((frm, to)),
        )
        routed.complete(_msgs())
        assert seen == [("a", "b")]

    def test_no_fallback_means_no_notification(self) -> None:
        seen: list[tuple[str, str]] = []
        RoutedProvider(
            [_Fake("a")], on_fallback=lambda frm, to, exc: seen.append((frm, to))
        ).complete(_msgs())
        assert seen == []

    def test_a_broken_observer_cannot_break_the_fallback(self) -> None:
        def _boom(frm: str, to: str, exc: BaseException) -> None:
            raise RuntimeError("observer down")

        routed = RoutedProvider(
            [_Fake("a", raises=RateLimitError()), _Fake("b")], on_fallback=_boom
        )
        assert routed.complete(_msgs()).content == "from b"


class TestStreaming:
    def test_a_healthy_stream_yields_from_the_primary(self) -> None:
        routed = RoutedProvider([_Fake("a", deltas=["x", "y"]), _Fake("b")])
        assert list(routed.stream_complete(_msgs())) == ["x", "y"]

    def test_a_transient_failure_before_any_delta_falls_back(self) -> None:
        routed = RoutedProvider(
            [_Fake("a", raises=RateLimitError()), _Fake("b", deltas=["z"])]
        )
        assert list(routed.stream_complete(_msgs())) == ["z"]

    def test_a_failure_MID_STREAM_never_falls_back(self) -> None:
        """Switching providers after bytes have shipped would splice two models'
        answers together — a silently corrupt response, worse than a clean failure."""
        secondary = _Fake("b", deltas=["should-not-appear"])
        routed = RoutedProvider([_PartialThenFails("a"), secondary])
        seen: list[str] = []
        with pytest.raises(RateLimitError):
            for piece in routed.stream_complete(_msgs()):
                seen.append(piece)
        assert seen == ["partial"] and secondary.calls == 0


class TestBuildProvider:
    def test_no_fallback_returns_a_plain_provider(self) -> None:
        # A project without a fallback pays nothing for the feature.
        assert not isinstance(build_provider("anthropic/x"), RoutedProvider)

    def test_a_fallback_returns_a_routed_provider(self) -> None:
        routed = build_provider("anthropic/x", fallback="openai/y")
        assert isinstance(routed, RoutedProvider)
        assert routed.chain == ["anthropic/x", "openai/y"]

    def test_a_fallback_equal_to_the_primary_is_not_wrapped(self) -> None:
        # Routing to the same model twice buys nothing but doubles the failure latency.
        assert not isinstance(build_provider("a/b", fallback="a/b"), RoutedProvider)
