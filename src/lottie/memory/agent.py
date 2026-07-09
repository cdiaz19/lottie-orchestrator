"""MemoryAgent — LLM-driven consolidation of episodic memory into semantic notes.

`MemoryAgent` reads recent episodic records via `self.memory`, asks the injected
LLM to consolidate them, and writes the resulting notes back as SEMANTIC
records. `MockMemoryAgent` prewires it with mock dependencies for tests.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from lottie.core import BaseAgent
from lottie.governance.audit import AuditLogger
from lottie.governance.schema import AuditRecord
from lottie.llm import LLMProvider, Message, MockLLMProvider
from lottie.memory.base import MemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import (
    ApplyResult,
    DeltaOp,
    MemoryDelta,
    MemoryOrigin,
    MemoryPatch,
    MemoryQuery,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    ReflectionInput,
    ReflectionResult,
)
from lottie.security.memory_gate import MemoryContentGate, MemoryContentRejected

REFLECT_SYSTEM_PROMPT = (
    "You consolidate an agent's recent episodic memory into durable notes. "
    "Read the log below and produce concise, standalone semantic notes — one "
    "per line, no numbering or bullets."
)


class MemoryAgent(BaseAgent[ReflectionInput, ReflectionResult]):
    """Consolidates recent episodic memory into semantic notes via the LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        content_gate: MemoryContentGate | None = None,
        name: str | None = None,
        memory: MemoryClient | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        super().__init__(
            llm,
            name=name,
            memory=memory,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
            audit=audit,
        )
        self._content_gate = content_gate or MemoryContentGate()
        # audit sink is BaseAgent's self._audit (built from `audit` or build_audit_logger).

    def _execute(self, data: ReflectionInput) -> ReflectionResult:
        recalled = self.memory.recall(
            MemoryQuery(
                text="",
                namespace=data.namespace,
                tier=MemoryTier.EPISODIC,
                limit=data.limit,
            )
        )
        episodic = [hit.record.content for hit in recalled.hits]
        response = self.complete(
            [
                Message(role="system", content=REFLECT_SYSTEM_PROMPT),
                Message(role="user", content="\n".join(episodic)),
            ]
        )
        notes = [line.strip() for line in response.content.splitlines() if line.strip()]
        written = [
            self.memory.remember(
                MemoryRecord(
                    content=note,
                    tier=MemoryTier.SEMANTIC,
                    namespace=data.namespace,
                    tags=["reflection"],
                )
            )
            for note in notes
        ]
        return ReflectionResult(
            notes=notes,
            consolidated_count=len(episodic),
            written_ids=written,
        )

    def apply(
        self,
        deltas: list[MemoryDelta],
        *,
        namespace: str,
        source_agent: str,
        origin: MemoryOrigin = MemoryOrigin.MANUAL,
        run_id: str | None = None,
    ) -> ApplyResult:
        """Gate, dedup, provenance-stamp, and audit each delta. Fail-closed per delta."""
        result = ApplyResult()
        for delta in deltas:
            if delta.op in (DeltaOp.ADD, DeltaOp.UPDATE):
                try:
                    self._content_gate.check(delta.content)
                except MemoryContentRejected as exc:
                    self._write_apply_audit(
                        source_agent, delta.content, "memory_rejected", str(exc)
                    )
                    result.rejected.append(str(exc))
                    continue
            if delta.op is DeltaOp.ADD:
                mid = self._apply_add(delta, namespace, source_agent, origin, run_id)
                self._write_apply_audit(source_agent, delta.content, "memory_write", None)
                result.applied_ids.append(mid)
            elif delta.op is DeltaOp.UPDATE:
                if delta.target_id is None:
                    result.rejected.append("update rejected: missing target_id")
                    continue
                rec = self.memory.update(
                    delta.target_id,
                    MemoryPatch(content=delta.content or None, tags=delta.tags or None),
                )
                self._write_apply_audit(source_agent, delta.content, "memory_write", None)
                result.applied_ids.append(rec.memory_id or delta.target_id)
            else:  # DEPRECATE
                if delta.target_id is None:
                    result.rejected.append("deprecate rejected: missing target_id")
                    continue
                rec = self.memory.update(
                    delta.target_id, MemoryPatch(status=MemoryStatus.DEPRECATED)
                )
                self._write_apply_audit(source_agent, "", "memory_deprecate", None)
                result.applied_ids.append(rec.memory_id or delta.target_id)
        return result

    def _apply_add(
        self,
        delta: MemoryDelta,
        namespace: str,
        source_agent: str,
        origin: MemoryOrigin,
        run_id: str | None,
    ) -> str:
        existing = self._find_by_content(namespace, delta.content)
        if existing is not None and existing.memory_id is not None:
            merged = sorted(set(existing.tags) | set(delta.tags))
            updated = self.memory.update(existing.memory_id, MemoryPatch(tags=merged))
            return updated.memory_id or existing.memory_id
        return self.memory.remember(
            MemoryRecord(
                content=delta.content,
                tier=MemoryTier.SEMANTIC,
                namespace=namespace,
                tags=delta.tags,
                origin=origin,
                source_agent=source_agent,
                run_id=run_id,
            )
        )

    def _find_by_content(self, namespace: str, content: str) -> MemoryRecord | None:
        hits = self.memory.recall(MemoryQuery(text="", namespace=namespace, limit=1000)).hits
        for hit in hits:
            if hit.record.content == content and hit.record.status is MemoryStatus.ACTIVE:
                return hit.record
        return None

    def _write_apply_audit(
        self, agent: str, content: str, status: str, error: str | None
    ) -> None:
        digest = hashlib.sha256(content.encode()).hexdigest()
        self._audit.log(
            AuditRecord(
                ts=datetime.now(UTC).isoformat(),
                agent=agent,
                provider=None,
                status=status,
                root=True,
                input_sha256=digest,
                output_sha256=None,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0.0,
                error=error,
            )
        )


class MockMemoryAgent(MemoryAgent):
    """MemoryAgent prewired with a mock LLM + mock client for tests."""

    def __init__(
        self,
        responses: list[str] | None = None,
        memory: MemoryClient | None = None,
    ) -> None:
        super().__init__(
            llm=MockLLMProvider(responses or ["note one\nnote two"]),
            memory=memory or MockMemoryClient(),
        )
