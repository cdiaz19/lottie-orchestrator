from __future__ import annotations

from lottie.serve.security import SecurityGate


def test_check_input_is_identity() -> None:
    gate = SecurityGate()
    assert gate.check_input('{"a": 1}') == '{"a": 1}'


def test_check_output_is_identity() -> None:
    gate = SecurityGate()
    assert gate.check_output('{"result": "ok"}') == '{"result": "ok"}'
