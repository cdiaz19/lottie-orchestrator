"""Unit tests for the Event Runtime.

Two properties are structural, not conventional, and are the reason this module exists:
a subscriber can never break a run, and no event ever carries raw content.
"""

from __future__ import annotations

import types
import typing
import warnings

import pytest

from lottie.runtime.events import (
    EventBus,
    RunBlocked,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
)

ALL_EVENTS: list[type[RunEvent]] = [RunStarted, RunCompleted, RunFailed, RunBlocked]

SCALARS = {str, int, float, bool, type(None)}

# Names that would signal a raw payload rode along on the bus.
FORBIDDEN_FIELD_NAMES = {
    "input",
    "output",
    "content",
    "payload",
    "prompt",
    "messages",
    "response",
    "text",
    "data",
}


class _Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.seen.append(event)


class _Exploder:
    name = "exploder"

    def on_event(self, event: RunEvent) -> None:
        raise RuntimeError("subscriber blew up")


def _started() -> RunStarted:
    return RunStarted(run_id="r1", runnable="Demo", kind="agent", input_sha256="a" * 64)


class TestEventBus:
    def test_emit_reaches_every_subscriber(self) -> None:
        bus = EventBus()
        first, second = _Recorder(), _Recorder()
        bus.subscribe(first)
        bus.subscribe(second)
        bus.emit(_started())
        assert len(first.seen) == 1
        assert len(second.seen) == 1

    def test_emit_dispatches_in_registration_order(self) -> None:
        order: list[str] = []

        class _Named:
            def __init__(self, label: str) -> None:
                self.name = label

            def on_event(self, event: RunEvent) -> None:
                order.append(self.name)

        bus = EventBus()
        bus.subscribe(_Named("first"))
        bus.subscribe(_Named("second"))
        bus.emit(_started())
        assert order == ["first", "second"]

    def test_a_failing_subscriber_never_propagates(self) -> None:
        bus = EventBus()
        bus.subscribe(_Exploder())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bus.emit(_started())  # must not raise

    def test_a_failing_subscriber_warns_and_names_itself(self) -> None:
        bus = EventBus()
        bus.subscribe(_Exploder())
        with pytest.warns(UserWarning, match="exploder"):
            bus.emit(_started())

    def test_a_failing_subscriber_does_not_starve_the_next_one(self) -> None:
        bus = EventBus()
        survivor = _Recorder()
        bus.subscribe(_Exploder())
        bus.subscribe(survivor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bus.emit(_started())
        assert len(survivor.seen) == 1

    def test_emit_with_no_subscribers_is_a_no_op(self) -> None:
        EventBus().emit(_started())


class TestEventsAreFrozen:
    @pytest.mark.parametrize("model", ALL_EVENTS)
    def test_events_cannot_be_mutated_by_a_subscriber(self, model: type[RunEvent]) -> None:
        assert model.model_config.get("frozen") is True


class TestHashOnlyContract:
    """V3 spec D6: the bus is an observation surface the Plugin SDK opens to third
    parties, so raw payloads on it would be an exfiltration channel."""

    @pytest.mark.parametrize("model", ALL_EVENTS)
    def test_every_field_is_a_scalar(self, model: type[RunEvent]) -> None:
        offenders: list[str] = []
        for field_name, field in model.model_fields.items():
            annotation = field.annotation
            parts: set[object]
            if typing.get_origin(annotation) is typing.Literal:
                parts = {type(arg) for arg in typing.get_args(annotation)}
            elif typing.get_origin(annotation) in (typing.Union, types.UnionType):
                parts = set(typing.get_args(annotation))
            else:
                parts = {annotation}
            if not parts <= SCALARS:
                offenders.append(f"{model.__name__}.{field_name}: {annotation}")
        assert offenders == []

    @pytest.mark.parametrize("model", ALL_EVENTS)
    def test_no_field_is_named_like_a_raw_payload(self, model: type[RunEvent]) -> None:
        leaked = set(model.model_fields) & FORBIDDEN_FIELD_NAMES
        assert leaked == set()
