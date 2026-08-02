"""Authoring prompt + reply parsing. Pure functions — no LLM, no mocking."""

from __future__ import annotations

import json

import pytest

from lottie.distill.author import (
    DistillParseError,
    build_distill_prompt,
    parse_distilled,
)
from lottie.memory.reflection import RunTrajectory


def _traj(task: str, outcome: str) -> RunTrajectory:
    return RunTrajectory(task=task, outcome=outcome, success=True)


def _reply(**kw: object) -> str:
    payload: dict[str, object] = {
        "description": "summarise a document",
        "system_prompt": "You summarise.",
        "user_template": "Summarise {doc}.",
        "slots": [{"name": "doc", "description": "the document", "required": True}],
    }
    payload.update(kw)
    return json.dumps(payload)


class TestPrompt:
    def test_includes_every_trajectory(self) -> None:
        msgs = build_distill_prompt("digest", [_traj("a", "A"), _traj("b", "B")])
        body = msgs[1].content
        assert "task: a" in body and "task: b" in body

    def test_states_the_agent_and_count(self) -> None:
        msgs = build_distill_prompt("digest", [_traj("a", "A")])
        assert "digest" in msgs[1].content and "1 successful run" in msgs[1].content

    def test_system_message_forbids_identity_changes(self) -> None:
        msgs = build_distill_prompt("digest", [_traj("a", "A")])
        assert "identity" in msgs[0].content


class TestParse:
    def test_parses_a_clean_reply(self) -> None:
        skill = parse_distilled(_reply(), name="summarise")
        assert skill.name == "summarise"
        assert skill.slot_names() == {"doc"}

    def test_tolerates_prose_around_the_json(self) -> None:
        skill = parse_distilled(f"Here you go:\n{_reply()}\nHope that helps!", name="s")
        assert skill.description == "summarise a document"

    def test_version_is_applied(self) -> None:
        assert parse_distilled(_reply(), name="s", version="0.3.0").version == "0.3.0"

    def test_no_json_raises(self) -> None:
        with pytest.raises(DistillParseError, match="no JSON"):
            parse_distilled("sorry, I cannot help with that", name="s")

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(DistillParseError, match="not valid JSON"):
            parse_distilled('{"description": "x", oops}', name="s")

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(DistillParseError, match="validation"):
            parse_distilled(json.dumps({"description": "x"}), name="s")

    def test_undeclared_placeholder_is_rejected(self) -> None:
        # Fail-closed: a template with a hole nothing fills would render `{secret}`
        # literally into every future prompt.
        reply = _reply(user_template="Summarise {doc} using {secret}.")
        with pytest.raises(DistillParseError, match="undeclared slot"):
            parse_distilled(reply, name="s")

    def test_declared_but_unused_slot_is_allowed(self) -> None:
        # The reverse is harmless — an optional slot the template ignores.
        reply = _reply(
            slots=[
                {"name": "doc", "description": "d", "required": True},
                {"name": "extra", "description": "e", "required": False},
            ]
        )
        assert parse_distilled(reply, name="s").slot_names() == {"doc", "extra"}

    def test_bad_slot_name_is_rejected(self) -> None:
        reply = _reply(
            user_template="Summarise {doc}.",
            slots=[{"name": "Doc-Name!", "description": "d", "required": True}],
        )
        with pytest.raises(DistillParseError, match="validation"):
            parse_distilled(reply, name="s")


def test_a_bare_json_array_reply_is_rejected() -> None:
    # A model can legitimately reply with a JSON array; that is not a skill. The
    # brace-scoped search rejects it before parsing, so the message is "no JSON object".
    with pytest.raises(DistillParseError, match="no JSON object"):
        parse_distilled('["a", "b"]', name="s")
