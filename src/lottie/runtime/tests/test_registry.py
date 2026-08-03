"""Module Orchestrator support.

Conflicts are rejected at REGISTRATION, not at run time: a plugin must not be able to
silently take a security middleware's slot, and the failure should surface at startup
where an operator sees it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.events import EventBus, RunEvent
from lottie.runtime.middleware import Next, Order
from lottie.runtime.registry import (
    Deps,
    ModuleConflictError,
    ModuleRegistry,
    Mountable,
)


class _Config(BaseModel):
    audit_enabled: bool = True
    policy_enabled: bool = True


class _PolicyMiddleware:
    name = "policy"
    order = Order.POLICY

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        return nxt(ctx)


class _AuditSubscriber:
    name = "audit"

    def on_event(self, event: RunEvent) -> None:
        return None


class _PolicyFactory:
    name = "policy"
    order = Order.POLICY

    def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
        return _PolicyMiddleware() if cfg.policy_enabled else None


class _AuditFactory:
    name = "audit"
    order = 900

    def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
        return _AuditSubscriber() if cfg.audit_enabled else None


class _EarlyMiddleware:
    name = "early"
    order = Order.SECURITY_INPUT

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        return nxt(ctx)


class _EarlyFactory:
    name = "early"
    order = Order.SECURITY_INPUT

    def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
        return _EarlyMiddleware()


def _deps(tmp_path: Path) -> Deps:
    return Deps(bus=EventBus(), root=tmp_path)


class TestRegistration:
    def test_registers_a_factory(self) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        assert reg.names() == ["policy"]

    def test_duplicate_name_is_rejected(self) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        with pytest.raises(ModuleConflictError, match="policy"):
            reg.register(_PolicyFactory())

    def test_duplicate_order_is_rejected_at_registration(self) -> None:
        class _Impostor:
            name = "impostor"
            order = Order.POLICY

            def build(self, cfg: _Config, deps: Deps) -> Mountable | None:
                return None

        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        with pytest.raises(ModuleConflictError) as exc:
            reg.register(_Impostor())
        # The message must name both claimants so the operator can act on it.
        assert "impostor" in str(exc.value)
        assert "policy" in str(exc.value)

    def test_distinct_orders_coexist(self) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_AuditFactory())
        assert reg.names() == ["policy", "audit"]


class TestBuild:
    def test_sorts_middleware_into_a_chain(self, tmp_path: Path) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_EarlyFactory())
        middleware, _ = reg.build(_Config(), _deps(tmp_path))
        assert [m.name for m in middleware] == ["early", "policy"]

    def test_separates_subscribers_from_middleware(self, tmp_path: Path) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_AuditFactory())
        middleware, subscribers = reg.build(_Config(), _deps(tmp_path))
        assert [m.name for m in middleware] == ["policy"]
        assert [s.name for s in subscribers] == ["audit"]

    def test_a_factory_returning_none_is_not_mounted(self, tmp_path: Path) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        reg.register(_AuditFactory())
        middleware, subscribers = reg.build(
            _Config(policy_enabled=False, audit_enabled=False), _deps(tmp_path)
        )
        assert middleware == []
        assert subscribers == []

    def test_build_on_an_empty_registry_returns_empty_lists(self, tmp_path: Path) -> None:
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        assert reg.build(_Config(), _deps(tmp_path)) == ([], [])

    def test_build_is_repeatable(self, tmp_path: Path) -> None:
        # The orchestrator builds one chain per agent; registration state must not be
        # consumed by the first build.
        reg: ModuleRegistry[_Config] = ModuleRegistry()
        reg.register(_PolicyFactory())
        first, _ = reg.build(_Config(), _deps(tmp_path))
        second, _ = reg.build(_Config(), _deps(tmp_path))
        assert [m.name for m in first] == [m.name for m in second]


class TestDeps:
    def test_deps_is_frozen(self, tmp_path: Path) -> None:
        deps = _deps(tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            deps.root = tmp_path  # type: ignore[misc]
