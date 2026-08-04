"""V3 S2b — `run_stream` shares `run()`'s middleware chain (the scoped subset).

The point of `ScopedMiddleware`: under the plain `Middleware` contract a `finally` fires
when the generator is CREATED, so a cost reservation would settle before the first delta.
These tests pin that scopes actually span consumption.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent, _depth
from lottie.governance.capability import CapabilityGate, active_capability_gate
from lottie.governance.policy import PolicyDenied, PolicyGate
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


class _Streamer(BaseAgent[_In, _Out]):
    def __init__(self, log: list[str] | None = None, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.log = log if log is not None else []

    def _execute(self, data: _In) -> _Out:
        return _Out(answer="unused")

    def _stream(self, data: _In) -> Iterator[str]:
        for piece in ("a", "b", "c"):
            self.log.append(f"yield:{piece}")
            yield piece

    def _write_audit(self, data: _In, output: _Out | None, is_root: bool) -> None:
        self.log.append(f"audit(output={output!r},root={is_root})")


def _agent(log: list[str] | None = None) -> _Streamer:
    return _Streamer(log, llm=MockLLMProvider(responses=["x"]), enable_benchmarks=False)


class TestSharedChain:
    def test_the_streaming_chain_is_the_scoped_subset(self) -> None:
        names = _agent()._build_pipeline().scoped_names()
        assert names == ["policy", "cost", "audit", "depth", "capability"]

    def test_output_shaping_middleware_are_absent(self) -> None:
        # verify / check_output / reflect need an output value, which a stream has not
        # got — matching run_stream's behaviour before the swap-in.
        names = _agent()._build_pipeline().scoped_names()
        assert not {"verify", "security_output", "reflect"} & set(names)


class TestStreamingWorks:
    def test_deltas_reach_the_caller(self) -> None:
        assert list(_agent().run_stream(_In(task="t"))) == ["a", "b", "c"]

    def test_nothing_runs_if_the_generator_is_never_iterated(self) -> None:
        log: list[str] = []
        _agent(log).run_stream(_In(task="t"))  # not consumed
        assert log == []


class TestScopesSpanConsumption:
    """The reason ScopedMiddleware exists."""

    def test_the_capability_gate_is_active_during_streaming(self) -> None:
        seen: list[object] = []

        class _Probe(_Streamer):
            def _stream(self, data: _In) -> Iterator[str]:
                seen.append(active_capability_gate())
                yield "a"

        agent = _Probe(llm=MockLLMProvider(responses=["x"]), enable_benchmarks=False)
        gate = CapabilityGate(["retrieval"])
        agent.set_capability_gate(gate)
        list(agent.run_stream(_In(task="t")))
        assert seen == [gate]

    def test_audit_fires_only_after_the_last_delta(self) -> None:
        log: list[str] = []
        stream = _agent(log).run_stream(_In(task="t"))
        assert next(stream) == "a"
        assert not any(entry.startswith("audit") for entry in log)  # mid-stream
        list(stream)  # drain
        assert any(entry.startswith("audit") for entry in log)

    def test_audit_records_no_output_for_a_stream(self) -> None:
        # A stream has no single typed Output — output=None, as before the swap-in.
        log: list[str] = []
        list(_agent(log).run_stream(_In(task="t")))
        assert "audit(output=None,root=True)" in log

    def test_depth_is_restored_after_the_stream(self) -> None:
        list(_agent().run_stream(_In(task="t")))
        assert _depth() == 0


class TestGates:
    def test_a_denied_stream_raises_before_the_first_delta(self) -> None:
        log: list[str] = []
        agent = _agent(log)
        agent.set_policy(PolicyGate(["banned"], allow=set(), deny={"banned"}, escalate=set()))
        with pytest.raises(PolicyDenied):
            next(agent.run_stream(_In(task="t")))
        assert not any(entry.startswith("yield") for entry in log)

    def test_closing_early_still_unwinds_the_scopes(self) -> None:
        # The transport closes the generator to cancel; the reservation must still settle.
        log: list[str] = []
        stream = _agent(log).run_stream(_In(task="t"))
        assert next(stream) == "a"
        stream.close()
        assert any(entry.startswith("audit") for entry in log)
        assert _depth() == 0
