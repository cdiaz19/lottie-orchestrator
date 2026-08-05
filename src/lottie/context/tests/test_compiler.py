"""Context Compiler — ordering, the budget, pinning, and provenance (E4 S1)."""

from __future__ import annotations

from lottie.context.compiler import CompileResult, StaticSource, compile_context
from lottie.llm import Message


def _msgs(n: int, tag: str) -> list[Message]:
    """Exactly 400 chars each == exactly 100 tokens under the chars/4 heuristic.

    The tag is padded INTO the 400 rather than added to it, so the budget arithmetic in
    these tests is exact rather than approximately right.
    """
    return [
        Message(role="system", content=f"{tag}{i}".ljust(400, "x")) for i in range(n)
    ]


def _src(name: str, order: int, n: int, *, pinned: bool = False) -> StaticSource:
    return StaticSource(name, order, _msgs(n, name), pinned=pinned)


class TestOrdering:
    def test_sources_emit_in_declared_order(self) -> None:
        result = compile_context([_src("b", 20, 1), _src("a", 10, 1)])
        assert result.messages[0].content.startswith("a")

    def test_registration_order_does_not_matter(self) -> None:
        first = compile_context([_src("a", 10, 1), _src("b", 20, 1)]).messages
        second = compile_context([_src("b", 20, 1), _src("a", 10, 1)]).messages
        assert first == second

    def test_every_message_is_present(self) -> None:
        result = compile_context([_src("a", 10, 2), _src("b", 20, 3)])
        assert len(result.messages) == 5


class TestNoBudget:
    def test_no_ceiling_returns_everything(self) -> None:
        result = compile_context([_src("a", 10, 50)])
        assert len(result.messages) == 50

    def test_provenance_is_reported_even_without_a_budget(self) -> None:
        result = compile_context([_src("a", 10, 2), _src("b", 20, 3)])
        assert [(c.name, c.messages) for c in result.contributions] == [("a", 2), ("b", 3)]


class TestUnderCeiling:
    def test_returns_everything_untouched(self) -> None:
        result = compile_context([_src("a", 10, 2)], max_tokens=100_000)
        assert len(result.messages) == 2 and result.dropped == []

    def test_the_summariser_is_not_called(self) -> None:
        calls: list[int] = []

        def _counting(messages: list[Message]) -> str:
            calls.append(len(messages))
            return "s"

        compile_context([_src("a", 10, 2)], max_tokens=100_000, summarize=_counting)
        assert calls == []


class TestDropPolicy:
    def test_lowest_order_droppable_goes_first(self) -> None:
        # Lowest order == furthest from the task == least contextually relevant.
        result = compile_context(
            [_src("knowledge", 10, 20), _src("task", 90, 1, pinned=True)], max_tokens=150
        )
        assert result.dropped == ["knowledge"]

    def test_a_pinned_source_is_never_dropped(self) -> None:
        result = compile_context([_src("recall", 20, 20, pinned=True)], max_tokens=10)
        assert result.dropped == []

    def test_only_pinned_over_budget_returns_as_is(self) -> None:
        # Silently discarding a pinned source to hit a number would break the contract;
        # the provider's own error is a better failure than a corrupted prompt.
        result = compile_context(
            [_src("recall", 20, 5, pinned=True), _src("task", 90, 5, pinned=True)],
            max_tokens=10,
        )
        assert len(result.messages) == 10

    def test_dropping_stops_once_under_budget(self) -> None:
        # The decision compaction could not make before: give up ONE source, not all.
        result = compile_context(
            [_src("a", 10, 10), _src("b", 20, 1), _src("task", 90, 1, pinned=True)],
            max_tokens=250,
        )
        assert result.dropped == ["a"]


class TestSummarisation:
    def test_a_droppable_source_is_summarised_when_a_summariser_exists(self) -> None:
        result = compile_context(
            [_src("knowledge", 10, 20), _src("task", 90, 1, pinned=True)],
            max_tokens=150,
            summarize=lambda ms: f"summary of {len(ms)}",
        )
        assert any("[compacted knowledge]" in m.content for m in result.messages)

    def test_summarising_beats_dropping(self) -> None:
        result = compile_context(
            [_src("knowledge", 10, 20), _src("task", 90, 1, pinned=True)],
            max_tokens=150,
            summarize=lambda ms: "s",
        )
        assert result.dropped == []
        assert [c.summarised for c in result.contributions if c.name == "knowledge"] == [True]

    def test_a_pinned_source_is_never_summarised(self) -> None:
        result = compile_context(
            [_src("recall", 20, 20, pinned=True)],
            max_tokens=10,
            summarize=lambda ms: "s",
        )
        assert all(not c.summarised for c in result.contributions)


class TestProvenance:
    def test_contributions_name_every_source_and_its_cost(self) -> None:
        result = compile_context([_src("a", 10, 2), _src("b", 20, 3)])
        by_name = {c.name: c for c in result.contributions}
        assert by_name["a"].tokens == 200 and by_name["b"].tokens == 300

    def test_total_tokens_excludes_dropped_sources(self) -> None:
        result = compile_context(
            [_src("a", 10, 10), _src("task", 90, 1, pinned=True)], max_tokens=150
        )
        assert result.total_tokens == 100  # only the pinned task survives

    def test_pinning_is_recorded(self) -> None:
        result = compile_context([_src("recall", 20, 1, pinned=True)])
        assert result.contributions[0].pinned is True


class TestEdgeCases:
    def test_no_sources(self) -> None:
        result = compile_context([])
        assert result.messages == [] and result.contributions == []

    def test_an_empty_source_is_reported_but_contributes_nothing(self) -> None:
        result = compile_context([StaticSource("empty", 10, [], pinned=False)])
        assert result.messages == [] and result.contributions[0].messages == 0

    def test_the_result_is_a_model(self) -> None:
        assert isinstance(compile_context([]), CompileResult)
