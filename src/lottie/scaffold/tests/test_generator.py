from __future__ import annotations

from typing import Literal

import pytest
import typer

from lottie.scaffold.generator import _class_name, _validate_name


@pytest.mark.parametrize(
    ("name", "kind", "expected"),
    [
        ("web_search", "skill", "WebSearchSkill"),
        ("researcher", "agent", "ResearcherAgent"),
        ("a_b_c", "skill", "ABCSkill"),
    ],
)
def test_class_name_derivation(
    name: str, kind: Literal["agent", "skill"], expected: str
) -> None:
    assert _class_name(name, kind) == expected


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "a/b", "../x", "Web", "web-search", "1foo", "class", "_private"]
)
def test_validate_name_rejects(bad: str) -> None:
    with pytest.raises(typer.BadParameter):
        _validate_name(bad)


def test_validate_name_accepts_snake() -> None:
    _validate_name("web_search")  # must not raise
