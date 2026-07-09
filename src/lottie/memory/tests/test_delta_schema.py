from lottie.memory.schema import ApplyResult, DeltaOp, MemoryDelta


def test_delta_defaults() -> None:
    d = MemoryDelta(op=DeltaOp.ADD, content="note")
    assert d.op is DeltaOp.ADD
    assert d.content == "note"
    assert d.tags == []
    assert d.target_id is None


def test_delta_update_carries_target() -> None:
    d = MemoryDelta(op=DeltaOp.UPDATE, content="new", target_id="m1", tags=["t"])
    assert d.op is DeltaOp.UPDATE
    assert d.target_id == "m1"
    assert d.tags == ["t"]


def test_apply_result_defaults_empty() -> None:
    r = ApplyResult()
    assert r.applied_ids == []
    assert r.rejected == []
