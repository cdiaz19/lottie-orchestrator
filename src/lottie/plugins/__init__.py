"""Public extension API for Lottie plugins (E7).

A plugin is a **Subscriber**: it observes runs and cannot break one. Implement `name` and
`on_event`, then name it in `config.yaml`:

```yaml
plugins:
  - module: "mypkg.telemetry:DatadogSubscriber"
```

```python
from lottie.plugins import RunCompleted, RunEvent

class DatadogSubscriber:
    name = "datadog"

    def on_event(self, event: RunEvent) -> None:
        if isinstance(event, RunCompleted):
            statsd.timing("lottie.run", event.latency_ms)
```

What a plugin can and cannot see
--------------------------------
Events carry **scalars and hashes only** — never a task, a prompt, or an output. That is
enforced by a contract test over every event model, not by convention.

A plugin runs in-process with the host's privileges and is **not sandboxed**. It may
observe; it may not intercept. Middleware plugins are deliberately unsupported: middleware
can abort a run and sees raw content, and that is not a position to hand to unsandboxed
third-party code.
"""

from lottie.plugins.loader import PluginLoadError, load_plugin, load_plugins
from lottie.runtime.events import (
    RunBlocked,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
    Subscriber,
)

__all__ = [
    "PluginLoadError",
    "RunBlocked",
    "RunCompleted",
    "RunEvent",
    "RunFailed",
    "RunStarted",
    "Subscriber",
    "load_plugin",
    "load_plugins",
]
