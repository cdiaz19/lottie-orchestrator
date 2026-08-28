"""Context compaction — keep a long run inside its context window.

A pure function with a single call site in `BaseAgent.complete()`. The shape is fixed by
the V3 spec (§1.1) so that V3's E4 Context Compiler can absorb compaction by moving the
call site rather than rewriting it: "pure function over messages" stays correct regardless
of what the compiler turns out to look like.

Three properties this module owes its caller
--------------------------------------------
1. **Nothing load-bearing is dropped.** The caller supplies a `pinned` predicate; pinned
   messages survive compaction untouched. `BaseAgent` pins system messages (which carry
   the recall-as-data block — a security contract, not a nicety) and the task itself.
2. **`summarize` is injected, never called on `self.complete`.** Summarising through the
   agent's own `complete()` would re-enter compaction unboundedly. Injection also makes
   every branch here unit-testable with a stub and zero LLM.
3. **Compaction is best-effort in the caller, total here.** This function either returns a
   valid message list or raises; it never returns something half-compacted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from lottie.llm import Message

#: Characters per token. A deliberate heuristic: a real tokenizer would mean a provider
#: dependency in the hot path of every completion, and compaction only needs to know
#: "roughly, are we near the limit". Errs toward over-estimating, so it compacts slightly
#: early rather than slightly late — the safe direction when the cost of being wrong is a
#: hard context-window error from the provider.
_CHARS_PER_TOKEN = 4

SUMMARY_PREFIX = "[compacted history]"


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Approximate the token cost of `messages`. See `_CHARS_PER_TOKEN`."""
    return sum(len(m.content) for m in messages) // _CHARS_PER_TOKEN


def compact(
    messages: list[Message],
    *,
    max_tokens: int,
    keep_recent: int,
    pinned: Callable[[Message], bool],
    summarize: Callable[[list[Message]], str],
) -> list[Message]:
    """Summarise older turns when `messages` exceeds `max_tokens`.

    Returns `messages` unchanged when it already fits, or when there is nothing droppable
    (everything is pinned or recent). Otherwise the droppable span is replaced by ONE
    summary message, inserted where that span began so ordering is preserved.

    `summarize` is only called when there is something to summarise, so a caller that
    spends tokens per call spends none on a run that never grows.
    """
    if estimate_tokens(messages) <= max_tokens:
        return messages

    total = len(messages)
    recent_from = max(0, total - keep_recent) if keep_recent > 0 else total
    droppable = [
        i for i, m in enumerate(messages) if i < recent_from and not pinned(m)
    ]
    if not droppable:
        # Everything is pinned or recent. Returning unchanged is correct: silently
        # dropping a pinned message to hit a budget would break the caller's contract,
        # and the provider's own error is a better failure than a corrupted prompt.
        return messages

    summary = summarize([messages[i] for i in droppable])
    dropped = set(droppable)
    out: list[Message] = []
    for i, message in enumerate(messages):
        if i == droppable[0]:
            out.append(Message(role="system", content=f"{SUMMARY_PREFIX} {summary}"))
        if i not in dropped:
            out.append(message)
    return out
