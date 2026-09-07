"""Load third-party subscribers by explicit import path (E7).

The trust boundary, stated plainly
----------------------------------
A plugin runs **in-process with the host's privileges**. Nothing here sandboxes it, and
nothing could — the capability gate constrains *agents calling skills*, not arbitrary
Python. The only real lever is how much of the system a plugin is handed, and the runtime
kernel already draws the line that matters:

* A **Subscriber** observes. The bus wraps every dispatch, so one that raises can neither
  fail a run nor starve the next, and every event carries scalars and hashes only.
* A **Middleware** intercepts. It can abort a run and it sees raw input and output.

A plugin may be a Subscriber. It may not be a Middleware. That bound is structural rather
than a matter of trusting the author, and widening a frozen API later is far cheaper than
narrowing one.

Why import paths and not entry points
-------------------------------------
An entry-point name lives in a global namespace shared by every installed distribution:
two packages can claim the same name, and a typo can resolve to a *different* package that
happens to advertise it. Opt-in does not fix squatting — not having a namespace does. An
explicit `module:attr` has no discovery step, enumerates nothing, and a typo simply fails
to import.
"""

from __future__ import annotations

import importlib

from lottie.runtime.events import Subscriber


class PluginLoadError(RuntimeError):
    """A configured plugin could not be loaded.

    Raised rather than warned-and-skipped: a plugin that silently fails to load leaves an
    operator believing their audit exporter is running. A refused startup is better than a
    false sense of observability.
    """


def load_plugin(spec: str) -> Subscriber:
    """Import and instantiate `module:attr`, verifying it is a Subscriber.

    The shape is checked at LOAD time, not at first event: a plugin that is not a
    subscriber should be rejected before a run depends on it.
    """
    if spec.count(":") != 1:
        raise PluginLoadError(
            f"invalid plugin spec {spec!r}: expected 'module.path:AttributeName'"
        )
    module_path, attr = spec.split(":")
    if not module_path or not attr:
        raise PluginLoadError(
            f"invalid plugin spec {spec!r}: expected 'module.path:AttributeName'"
        )

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise PluginLoadError(f"cannot import plugin module {module_path!r}: {exc}") from exc

    target = getattr(module, attr, None)
    if target is None:
        raise PluginLoadError(f"plugin module {module_path!r} has no attribute {attr!r}")

    try:
        instance = target() if isinstance(target, type) else target
    except Exception as exc:
        raise PluginLoadError(f"plugin {spec!r} failed to construct: {exc}") from exc

    if not isinstance(instance, Subscriber):
        raise PluginLoadError(
            f"plugin {spec!r} is not a Subscriber — it must expose `name` and "
            "`on_event(event)`. Middleware plugins are deliberately not supported: they "
            "could abort a run and would see raw input and output."
        )
    return instance


def load_plugins(specs: list[str]) -> list[Subscriber]:
    """Load every configured plugin, in declared order (which is dispatch order).

    An empty list runs no loading code at all — a project with no plugins pays nothing.
    """
    return [load_plugin(spec) for spec in specs]
