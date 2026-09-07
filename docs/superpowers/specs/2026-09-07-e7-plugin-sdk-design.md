# E7 — Plugin SDK

> Epic design. Target: **v3.4.0**. Date: 2026-09-07.
> Theme: let a third party observe a run, without handing them the position that could
> break one.
> Follows `2026-07-30-v3-runtime-kernel-design.md` §8 (E7). **Last epic on purpose:** a
> public extension API is only worth freezing once the internal modules have proved the
> interface.

---

## 1. Architecture review

### The extension point does not exist yet

`ModuleRegistry` and `ModuleFactory` shipped in V3 S1 and **nothing uses them**. S6 wired
module composition as `build_chain(agent, disabled)` — a hardcoded list plus
disabling-by-name — and never went through the registry.

So E7 cannot simply "add entry points to the registry": the registry is not load-bearing.
Either E7 makes it real, or it builds on a fiction. This slice makes it real for the
subscriber path.

### The security position, restated

The V3 spec put this on record and it has not changed:

> Plugins load **in-process with full trust**. The capability gate constrains *agents
> calling skills*; it does **not** sandbox a plugin.

That is not a limitation to work around — it is the fact that shapes every decision below.
A plugin is code running with the host's privileges. The only real lever is **how much of
the system it is handed**, and the kernel already draws the line that matters.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **A plugin may be a Subscriber. Not a Middleware.** | The bus already guarantees a raising subscriber can neither fail a run nor starve the next, and events carry hashes only. The blast radius is bounded **by construction, not by trust**. Middleware can abort a run and sees raw input/output — handing that to unsandboxed third-party code gives away the strongest position in the system. Covers the real cases (telemetry, metrics, exporters), and **widening a frozen API later is far cheaper than narrowing one**. |
| D2 | **Named by explicit import path** (`mypkg.telemetry:DatadogSubscriber`), not entry points. | There is no discovery step and no enumeration of installed packages; a typo simply fails to import. Entry-point names live in a global namespace shared across every installed distribution — two packages can claim one name, and a typo can resolve to a *different* package that happens to advertise it. Opt-in does not fix squatting; not having a namespace does. |
| D3 | **Load failures are fatal at startup, not skipped.** | A plugin that silently fails to load leaves an operator believing their audit exporter is running. Better a refused startup than a false sense of observability. |
| D4 | **`lottie doctor` and `lottie modules` list loaded plugins** and state the trust boundary in words. | An operator should never have to read source to learn what third-party code is in their process. |

---

## 3. Architecture

### 3.1 `plugins:` config block

```yaml
plugins:
  - module: "mypkg.telemetry:DatadogSubscriber"
  - module: "internal.audit:S3Exporter"
```

A list, not a map: order is the subscribe order, and the bus dispatches in registration
order. Absent block = no plugins = no loading code runs at all.

### 3.2 `load_plugin(spec) -> Subscriber`

Imports `module:attr`, instantiates it, and **verifies it satisfies the `Subscriber`
protocol** before returning — a plugin that is not a subscriber is rejected by shape, not
discovered at first event.

Raises `PluginLoadError` on anything: bad path, import failure, missing attribute, wrong
shape. Per D3, the caller does not catch it.

### 3.3 What a plugin sees

Exactly what any subscriber sees: `RunStarted`, `RunCompleted`, `RunFailed`, `RunBlocked`
— all hash-only (V3 spec D6, enforced by a contract test over every event model). A plugin
cannot read a task, a prompt, or an output. That is the whole point of D1.

### 3.4 Public surface

`lottie.plugins` re-exports `Subscriber`, the event types, and `PluginLoadError` — one
import for a plugin author, and an explicit statement of what is frozen.

---

## 4. Slice plan

| Slice | Delivers | Lab |
|---|---|---|
| **S1** | `plugins:` config, `load_plugin`, subscriber mounting, `lottie plugins` listing, doctor advisory, `lottie.plugins` public surface, author docs | **R38** |
| **S2** | Release: bump 3.4.0, CHANGELOG, tag — **completes the V3 roadmap** | full regression |

---

## 5. Invariants

- **No auto-discovery.** Nothing loads that the config did not name.
- **Subscribers only.** A non-subscriber is rejected at load, by shape.
- **A load failure fails startup.** Never skipped, never warned-and-continued.
- **Events stay hash-only.** A plugin cannot see raw content.
- **No plugins configured → no loading code executes.**
- Rule 7b gate, one PR, one lab round.

---

## 6. Definition of Done (v3.4.0)

- A third-party subscriber can be loaded by explicit path and receives real events.
- A non-subscriber, a bad path, and a broken import each fail startup loudly.
- `lottie plugins` and `doctor` show what is loaded and state the trust boundary.
- R38 green, full regression green, `v3.4.0` tagged — **V3 roadmap complete**.
