"""Context compaction: what survives, what is summarised, and what is never touched."""

from __future__ import annotations

import pytest

from lottie.context.compaction import SUMMARY_PREFIX, compact, estimate_tokens
from lottie.llm import Message, Role


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def _summarize(messages: list[Message]) -> str:
    return f"summary of {len(messages)}"


def _never_pinned(m: Message) -> bool:
    return False


def _system_pinned(m: Message) -> bool:
    return m.role == "system"


def _big(n: int, tag: str = "u") -> list[Message]:
    # 400 chars each ≈ 100 tokens under the heuristic.
    return [_msg("user", f"{tag}{i}" + "x" * 400) for i in range(n)]


class TestEstimateTokens:
    def test_empty_is_zero(self) -> None:
        assert estimate_tokens([]) == 0

    def test_scales_with_content_length(self) -> None:
        assert estimate_tokens([_msg("user", "x" * 400)]) == 100

    def test_sums_across_messages(self) -> None:
        assert estimate_tokens([_msg("user", "x" * 400)] * 3) == 300


class TestNoOp:
    def test_returns_unchanged_when_it_already_fits(self) -> None:
        messages = _big(2)
        assert (
            compact(
                messages,
                max_tokens=10_000,
                keep_recent=1,
                pinned=_never_pinned,
                summarize=_summarize,
            )
            is messages
        )

    def test_summarize_is_not_called_when_it_fits(self) -> None:
        calls: list[int] = []

        def _counting(messages: list[Message]) -> str:
            calls.append(len(messages))
            return "s"

        compact(
            _big(2), max_tokens=10_000, keep_recent=1, pinned=_never_pinned, summarize=_counting
        )
        assert calls == []

    def test_returns_unchanged_when_everything_is_pinned(self) -> None:
        # Dropping a pinned message to hit a budget would break the caller's contract;
        # the provider's own error is a better failure than a corrupted prompt.
        messages = [_msg("system", "s" + "x" * 400) for _ in range(5)]
        assert (
            compact(
                messages,
                max_tokens=10,
                keep_recent=0,
                pinned=_system_pinned,
                summarize=_summarize,
            )
            is messages
        )

    def test_returns_unchanged_when_everything_is_recent(self) -> None:
        messages = _big(3)
        assert (
            compact(
                messages,
                max_tokens=10,
                keep_recent=99,
                pinned=_never_pinned,
                summarize=_summarize,
            )
            is messages
        )


class TestCompaction:
    def test_shrinks_the_message_list(self) -> None:
        out = compact(
            _big(10), max_tokens=100, keep_recent=2, pinned=_never_pinned, summarize=_summarize
        )
        assert len(out) < 10

    def test_keeps_the_requested_recent_messages(self) -> None:
        messages = _big(10)
        out = compact(
            messages, max_tokens=100, keep_recent=2, pinned=_never_pinned, summarize=_summarize
        )
        assert out[-2:] == messages[-2:]

    def test_inserts_exactly_one_summary(self) -> None:
        out = compact(
            _big(10), max_tokens=100, keep_recent=2, pinned=_never_pinned, summarize=_summarize
        )
        assert sum(1 for m in out if m.content.startswith(SUMMARY_PREFIX)) == 1

    def test_the_summary_covers_the_dropped_span(self) -> None:
        out = compact(
            _big(10), max_tokens=100, keep_recent=2, pinned=_never_pinned, summarize=_summarize
        )
        summary = next(m for m in out if m.content.startswith(SUMMARY_PREFIX))
        assert "summary of 8" in summary.content

    def test_the_summary_is_marked_as_compacted_history(self) -> None:
        # A reader (human or model) must be able to tell this is not a real turn.
        out = compact(
            _big(6), max_tokens=100, keep_recent=1, pinned=_never_pinned, summarize=_summarize
        )
        assert any(m.content.startswith(SUMMARY_PREFIX) for m in out)


class TestPinning:
    def test_pinned_messages_survive(self) -> None:
        messages = [_msg("system", "PINNED" + "x" * 400), *_big(10)]
        out = compact(
            messages, max_tokens=100, keep_recent=1, pinned=_system_pinned, summarize=_summarize
        )
        assert any("PINNED" in m.content for m in out)

    def test_pinned_messages_are_not_summarised(self) -> None:
        seen: list[list[Message]] = []

        def _capture(messages: list[Message]) -> str:
            seen.append(messages)
            return "s"

        messages = [_msg("system", "PINNED" + "x" * 400), *_big(10)]
        compact(
            messages, max_tokens=100, keep_recent=1, pinned=_system_pinned, summarize=_capture
        )
        assert all("PINNED" not in m.content for m in seen[0])

    def test_a_pinned_message_keeps_its_position(self) -> None:
        # Ordering matters: a system message that migrated after the task would change
        # how the model reads the prompt.
        messages = [_msg("system", "PINNED" + "x" * 400), *_big(10)]
        out = compact(
            messages, max_tokens=100, keep_recent=1, pinned=_system_pinned, summarize=_summarize
        )
        assert out[0].content.startswith("PINNED")

    def test_the_final_message_can_be_pinned_by_recency(self) -> None:
        messages = _big(10, tag="task")
        out = compact(
            messages, max_tokens=100, keep_recent=1, pinned=_never_pinned, summarize=_summarize
        )
        assert out[-1] is messages[-1]


class TestOrdering:
    def test_the_summary_sits_where_the_dropped_span_began(self) -> None:
        messages = [_msg("system", "S" + "x" * 400), *_big(8)]
        out = compact(
            messages, max_tokens=100, keep_recent=2, pinned=_system_pinned, summarize=_summarize
        )
        assert out[0].content.startswith("S")
        assert out[1].content.startswith(SUMMARY_PREFIX)

    def test_relative_order_of_survivors_is_preserved(self) -> None:
        messages = _big(10)
        out = compact(
            messages, max_tokens=100, keep_recent=3, pinned=_never_pinned, summarize=_summarize
        )
        survivors = [m for m in out if not m.content.startswith(SUMMARY_PREFIX)]
        assert survivors == messages[-3:]


class TestEdgeCases:
    def test_empty_list(self) -> None:
        assert compact(
            [], max_tokens=0, keep_recent=1, pinned=_never_pinned, summarize=_summarize
        ) == []

    def test_keep_recent_zero_allows_dropping_everything_unpinned(self) -> None:
        out = compact(
            _big(4), max_tokens=1, keep_recent=0, pinned=_never_pinned, summarize=_summarize
        )
        assert len(out) == 1 and out[0].content.startswith(SUMMARY_PREFIX)

    def test_a_raising_summarizer_propagates(self) -> None:
        # This function is total: it returns a valid list or raises. The CALLER decides
        # whether a failed summary is fatal.
        def _boom(messages: list[Message]) -> str:
            raise RuntimeError("summariser down")

        with pytest.raises(RuntimeError, match="summariser down"):
            compact(
                _big(6), max_tokens=10, keep_recent=1, pinned=_never_pinned, summarize=_boom
            )
