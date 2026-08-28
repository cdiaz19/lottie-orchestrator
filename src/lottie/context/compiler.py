"""Context Compiler — an ordering authority, a budget, and provenance (E4 S1).

Before this, context was assembled in four places and compaction saw a flat list with no
idea what came from where. It could only summarise by *recency*, so it had no way to
prefer dropping stale knowledge over recent turns — usually the right trade.

The compiler gives assembly three things it lacked:

* **Ordering** — sources emit in declared order, not in whatever order the agent wrote.
* **A cross-source budget** — the ceiling is applied across everything, and the drop
  policy chooses *which source* to give up rather than which position.
* **Provenance** — `CompileResult.contributions` answers "which source filled the
  window?", which is exactly the question when a prompt gets expensive.

Pinning moves from ROLE to SOURCE, which is the point. A knowledge block and the recall
block are both system messages; only the recall block is load-bearing (S2a's
anti-poisoning contract). Keying on role could not tell them apart.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel

from lottie.context.compaction import compact, estimate_tokens
from lottie.llm import Message


class ContextSource(Protocol):
    """One contributor to a prompt.

    `order` is the assembly authority: low emits first. `pinned` means the drop policy
    must never touch it.
    """

    name: str
    order: int
    pinned: bool

    def emit(self) -> list[Message]: ...


class SourceContribution(BaseModel):
    """What one source cost, and what the budget did to it."""

    name: str
    tokens: int
    messages: int
    pinned: bool
    dropped: bool = False
    summarised: bool = False


class CompileResult(BaseModel):
    """The assembled prompt plus a record of how it got that way."""

    messages: list[Message]
    contributions: list[SourceContribution]

    @property
    def dropped(self) -> list[str]:
        """Names of the sources the budget gave up, in drop order."""
        return [c.name for c in self.contributions if c.dropped]

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.contributions if not c.dropped)


class StaticSource:
    """A source over a fixed message list. The shape every standard source takes."""

    def __init__(self, name: str, order: int, messages: list[Message], *, pinned: bool) -> None:
        self.name = name
        self.order = order
        self.pinned = pinned
        self._messages = messages

    def emit(self) -> list[Message]:
        return self._messages


def compile_context(
    sources: Sequence[ContextSource],
    *,
    max_tokens: int | None = None,
    summarize: Callable[[list[Message]], str] | None = None,
    keep_recent: int = 6,
) -> CompileResult:
    """Assemble `sources` in order, applying the budget when one is set.

    Drop policy, in order of preference:

    1. Under the ceiling → return everything untouched, and **call nothing**. The cheap
       estimate runs first so a short run never pays for summarisation it did not need.
    2. Over the ceiling → drop DROPPABLE sources lowest-order-first. Lowest order means
       furthest from the task, which is the least contextually relevant thing to lose.
    3. Still over, and a summariser is available → summarise the remaining droppable
       content rather than dropping it outright.

    4. Still over with only PINNED sources left → fall back to compacting their older
       messages, keeping the most recent `keep_recent`. This is the case that matters for
       a long conversation: the turns are all pinned, so the only thing left to give up is
       history, and losing the oldest turns beats overflowing the window.

    A pinned source is never DROPPED. Step 4 may summarise its older messages, which is a
    different thing — the source still contributes, in condensed form. When even that is
    not enough the result is returned as-is: the provider's own error is a better failure
    than a silently truncated prompt.
    """
    ordered = sorted(sources, key=lambda s: s.order)
    emitted: list[tuple[ContextSource, list[Message]]] = [(s, s.emit()) for s in ordered]
    contributions = [
        SourceContribution(
            name=source.name,
            tokens=estimate_tokens(messages),
            messages=len(messages),
            pinned=source.pinned,
        )
        for source, messages in emitted
    ]

    def _assemble(dropped: set[str]) -> list[Message]:
        return [m for source, msgs in emitted if source.name not in dropped for m in msgs]

    if max_tokens is None:
        return CompileResult(messages=_assemble(set()), contributions=contributions)

    dropped: set[str] = set()
    if estimate_tokens(_assemble(dropped)) <= max_tokens:
        return CompileResult(messages=_assemble(dropped), contributions=contributions)

    by_name = {c.name: c for c in contributions}
    for source, messages in emitted:
        if source.pinned or not messages:
            continue
        if summarize is not None:
            summary = summarize(messages)
            emitted[emitted.index((source, messages))] = (
                source,
                [Message(role="system", content=f"[compacted {source.name}] {summary}")],
            )
            by_name[source.name].summarised = True
        else:
            dropped.add(source.name)
            by_name[source.name].dropped = True
        if estimate_tokens(_assemble(dropped)) <= max_tokens:
            break

    assembled = _assemble(dropped)
    if summarize is not None and estimate_tokens(assembled) > max_tokens:
        # Only pinned sources are left, so source-pinning can no longer discriminate —
        # everything here is pinned. Within a surviving source, ROLE is the right signal:
        # system messages are structural (the system prompt, the recall-as-data block)
        # while user/assistant turns are conversational history that compacts by recency.
        #
        # Two different questions, answered at two different levels: which SOURCES
        # survive, then which MESSAGES within them.
        assembled = compact(
            assembled,
            max_tokens=max_tokens,
            keep_recent=keep_recent,
            pinned=lambda m: m.role == "system",
            summarize=summarize,
        )

    return CompileResult(messages=assembled, contributions=contributions)
