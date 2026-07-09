import pytest

from lottie.distill.generate import (
    build_distill_prompt,
    extract_slots,
    fill_template,
)


def test_build_prompt_has_system_and_notes() -> None:
    msgs = build_distill_prompt(["use backoff", "cache config"])
    assert [m.role for m in msgs] == ["system", "user"]
    assert "use backoff" in msgs[1].content
    assert "cache config" in msgs[1].content


def test_extract_slots_sorted_unique() -> None:
    assert extract_slots("Summarize {topic} in {n} words about {topic}.") == ["n", "topic"]
    assert extract_slots("no slots here") == []


def test_fill_template_replaces_and_leaves_other_braces() -> None:
    out = fill_template("Summarize {topic}", {"topic": "cats"})
    assert out == "Summarize cats"


def test_fill_template_missing_slot_raises() -> None:
    with pytest.raises(KeyError):
        fill_template("Summarize {topic}", {})
