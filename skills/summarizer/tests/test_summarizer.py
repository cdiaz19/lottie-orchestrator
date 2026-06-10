"""Unit tests for SummarizerSkill — MockLLMProvider only (no real LLM)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lottie.llm import MockLLMProvider
from skills.summarizer.schema import SummarizerInput, SummarizerOutput
from skills.summarizer.skill import SummarizerSkill

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LOTTIE_DOC = (
    "Lottie is a multi-agent orchestration framework designed for provider-agnostic "
    "AI workflows with built-in governance, security gates, and typed knowledge layers."
)

MULTI_BULLET_RESPONSE = (
    "Lottie is a multi-agent framework.\n"
    "- Typed schemas\n"
    "- Provider-agnostic\n"
    "- Security gate"
)

SIX_BULLET_RESPONSE = (
    "Here is a summary of the document.\n"
    "- Point one\n"
    "- Point two\n"
    "- Point three\n"
    "- Point four\n"
    "- Point five\n"
    "- Point six"
)

NO_BULLET_RESPONSE = "Lottie is a multi-agent framework with typed schemas and security gates."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_summarizer_extracts_prose_and_bullets() -> None:
    """LLM response with prose + bullets → summary + matching points list."""
    mock = MockLLMProvider([MULTI_BULLET_RESPONSE])
    skill = SummarizerSkill(mock)
    result = skill.run(SummarizerInput(text=LOTTIE_DOC, max_points=5))

    assert isinstance(result, SummarizerOutput)
    assert result.summary  # non-empty prose
    assert result.points == ["Typed schemas", "Provider-agnostic", "Security gate"]


def test_summarizer_caps_points_at_max_points() -> None:
    """When the response has more bullets than max_points, points are capped."""
    mock = MockLLMProvider([SIX_BULLET_RESPONSE])
    skill = SummarizerSkill(mock)
    result = skill.run(SummarizerInput(text=LOTTIE_DOC, max_points=3))

    assert isinstance(result, SummarizerOutput)
    assert len(result.points) == 3


def test_summarizer_no_bullets_returns_empty_points() -> None:
    """When the LLM returns plain prose (no bullets) points is empty."""
    mock = MockLLMProvider([NO_BULLET_RESPONSE])
    skill = SummarizerSkill(mock)
    result = skill.run(SummarizerInput(text=LOTTIE_DOC, max_points=5))

    assert isinstance(result, SummarizerOutput)
    assert result.points == []
    assert result.summary == NO_BULLET_RESPONSE.strip()


def test_summarizer_last_metrics_populated_after_run() -> None:
    """last_metrics is set after a successful run (skill is benchmarkable)."""
    mock = MockLLMProvider([MULTI_BULLET_RESPONSE])
    skill = SummarizerSkill(mock)
    skill.run(SummarizerInput(text=LOTTIE_DOC, max_points=5))

    assert skill.last_metrics is not None


def test_summarizer_prompt_contains_input_text() -> None:
    """The LLM is called with a prompt that contains the input text."""
    mock = MockLLMProvider([MULTI_BULLET_RESPONSE])
    skill = SummarizerSkill(mock)
    skill.run(SummarizerInput(text=LOTTIE_DOC, max_points=5))

    assert mock.calls, "MockLLMProvider should have recorded at least one call"
    all_content = " ".join(m.content for m in mock.calls[0])
    assert LOTTIE_DOC in all_content


def test_summarizer_schema_rejects_wrong_type() -> None:
    """Pydantic validates input types at construction time."""
    with pytest.raises(ValidationError):
        SummarizerInput.model_validate({"text": 123})


def test_summarizer_default_max_points_is_five() -> None:
    """Default max_points is 5."""
    inp = SummarizerInput(text="hello")
    assert inp.max_points == 5


def test_summarizer_di_constructor_forwarding() -> None:
    """Keyword-only trio is forwarded to super().__init__ correctly."""
    from pathlib import Path

    mock = MockLLMProvider([NO_BULLET_RESPONSE])
    skill = SummarizerSkill(
        mock,
        name="custom-name",
        enable_benchmarks=False,
        benchmarks_root=Path("/tmp"),
    )
    assert skill.name == "custom-name"
    assert skill._enable_benchmarks is False


# ---------------------------------------------------------------------------
# Fix 1: max_points lower bound
# ---------------------------------------------------------------------------


def test_max_points_zero_raises_validation_error() -> None:
    """max_points=0 must be rejected by Pydantic (ge=1)."""
    with pytest.raises(ValidationError):
        SummarizerInput(text="x", max_points=0)


def test_max_points_negative_raises_validation_error() -> None:
    """max_points=-1 must be rejected by Pydantic (ge=1)."""
    with pytest.raises(ValidationError):
        SummarizerInput(text="x", max_points=-1)


def test_max_points_one_is_valid() -> None:
    """max_points=1 is the minimum allowed value."""
    inp = SummarizerInput(text="x", max_points=1)
    assert inp.max_points == 1


# ---------------------------------------------------------------------------
# Fix 2: order-independent parsing
# ---------------------------------------------------------------------------

BULLETS_BEFORE_PROSE = "- A\n- B\nSome prose."
TRAILING_PROSE = "Intro.\n- A\n- B\nConclusion."
ALL_BULLETS = "- A\n- B"


def test_parse_bullets_before_prose_summary_nonempty() -> None:
    """bullets-before-prose: summary contains the prose line, points == ['A','B']."""
    mock = MockLLMProvider([BULLETS_BEFORE_PROSE])
    skill = SummarizerSkill(mock)
    result = skill.run(SummarizerInput(text="x", max_points=5))

    assert "Some prose." in result.summary
    assert result.summary != ""
    assert result.points == ["A", "B"]


def test_parse_trailing_prose_joined_into_summary() -> None:
    """Leading and trailing prose both appear in summary, joined."""
    mock = MockLLMProvider([TRAILING_PROSE])
    skill = SummarizerSkill(mock)
    result = skill.run(SummarizerInput(text="x", max_points=5))

    assert result.summary == "Intro. Conclusion."
    assert result.points == ["A", "B"]


def test_parse_all_bullets_no_prose_summary_nonempty() -> None:
    """All-bullets response: summary falls back to full stripped text (non-empty)."""
    mock = MockLLMProvider([ALL_BULLETS])
    skill = SummarizerSkill(mock)
    result = skill.run(SummarizerInput(text="x", max_points=5))

    assert result.summary == ALL_BULLETS.strip()
    assert result.points == ["A", "B"]
