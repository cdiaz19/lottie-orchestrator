from lottie.memory.recall import RecalledMemory, render_as_data
from lottie.memory.schema import (
    MemoryHit,
    MemoryOrigin,
    MemoryRecord,
    RecallResult,
)


def _result(*contents: str) -> RecallResult:
    return RecallResult(
        hits=[
            MemoryHit(
                record=MemoryRecord(
                    content=c, namespace="ns", origin=MemoryOrigin.REFLECTION, source_agent="Digest"
                ),
                score=1.0,
            )
            for c in contents
        ]
    )


def test_from_result_collects_records() -> None:
    recalled = RecalledMemory.from_result(_result("a", "b"))
    assert [r.content for r in recalled.records] == ["a", "b"]


def test_render_marks_data_not_instructions() -> None:
    text = render_as_data(RecalledMemory.from_result(_result("use backoff")))
    assert "use backoff" in text
    lower = text.lower()
    assert "data" in lower and "not instructions" in lower
    assert "Digest" in text  # provenance surfaced


def test_render_empty_is_empty_string() -> None:
    assert render_as_data(RecalledMemory()) == ""
