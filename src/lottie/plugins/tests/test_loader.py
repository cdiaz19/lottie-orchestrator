"""Plugin loading (E7): what is accepted, and what is refused at the door."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lottie.plugins import PluginLoadError, load_plugin, load_plugins
from lottie.runtime.context import ExecutionContext
from lottie.runtime.events import RunEvent


class GoodSubscriber:
    name = "good"

    def on_event(self, event: RunEvent) -> None:
        return None


class NotASubscriber:
    """No `on_event` — the shape a plugin must have."""

    name = "impostor"


class AMiddleware:
    """Deliberately unsupported: middleware can abort a run and sees raw content."""

    name = "interceptor"
    order = 50

    def __call__(self, ctx: ExecutionContext, nxt: object) -> BaseModel:
        raise NotImplementedError


class ExplodesOnConstruction:
    name = "boom"

    def __init__(self) -> None:
        raise RuntimeError("cannot construct me")

    def on_event(self, event: RunEvent) -> None:
        return None


_HERE = "lottie.plugins.tests.test_loader"


class TestAccepts:
    def test_loads_a_subscriber_by_explicit_path(self) -> None:
        assert load_plugin(f"{_HERE}:GoodSubscriber").name == "good"

    def test_loads_several_in_declared_order(self) -> None:
        loaded = load_plugins([f"{_HERE}:GoodSubscriber", f"{_HERE}:GoodSubscriber"])
        assert len(loaded) == 2

    def test_an_empty_list_loads_nothing(self) -> None:
        # A project with no plugins runs no loading code at all.
        assert load_plugins([]) == []


class TestRefusesMiddleware:
    """The bound that is structural rather than a matter of trust."""

    def test_a_middleware_is_rejected(self) -> None:
        with pytest.raises(PluginLoadError):
            load_plugin(f"{_HERE}:AMiddleware")

    def test_the_error_explains_why_middleware_is_unsupported(self) -> None:
        # An author who tries this deserves the reason, not just a refusal.
        with pytest.raises(PluginLoadError) as exc:
            load_plugin(f"{_HERE}:AMiddleware")
        assert "abort a run" in str(exc.value) and "raw input" in str(exc.value)


class TestRefusesMalformed:
    def test_a_non_subscriber_is_rejected_at_load(self) -> None:
        # Rejected by SHAPE at load time, not discovered at the first event.
        with pytest.raises(PluginLoadError, match="not a Subscriber"):
            load_plugin(f"{_HERE}:NotASubscriber")

    @pytest.mark.parametrize("bad", ["no-colon", "too:many:colons", ":NoModule", "mod:"])
    def test_a_malformed_spec_is_rejected(self, bad: str) -> None:
        with pytest.raises(PluginLoadError, match="invalid plugin spec"):
            load_plugin(bad)

    def test_an_unimportable_module_is_rejected(self) -> None:
        with pytest.raises(PluginLoadError, match="cannot import"):
            load_plugin("no.such.module:Thing")

    def test_a_missing_attribute_is_rejected(self) -> None:
        with pytest.raises(PluginLoadError, match="no attribute"):
            load_plugin(f"{_HERE}:DoesNotExist")

    def test_a_constructor_failure_is_reported(self) -> None:
        with pytest.raises(PluginLoadError, match="failed to construct"):
            load_plugin(f"{_HERE}:ExplodesOnConstruction")


class TestFailuresAreFatal:
    def test_one_bad_plugin_fails_the_whole_load(self) -> None:
        """Never warned-and-skipped.

        A plugin that silently fails to load leaves an operator believing their audit
        exporter is running — a refused startup is better than false observability.
        """
        with pytest.raises(PluginLoadError):
            load_plugins([f"{_HERE}:GoodSubscriber", "no.such.module:Thing"])
