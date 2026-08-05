# E4 — Context Compiler

> Epic design. Target: **v3.1.0**. Date: 2026-08-05.
> Theme: give message assembly an **ordering authority, a cross-source budget, and
> provenance** — so compaction can make an informed decision instead of a positional one.
> Follows `2026-07-30-v3-runtime-kernel-design.md` §8 (E4).

---

## 1. Architecture analysis

Context is assembled in **four places, three of them inside agent code**:

| Where | What it adds |
|---|---|
| the agent's `_execute` | system prompt + task, hand-built `list[Message]` |
| the agent's `from_project` | knowledge, for knowledge-backed agents |
| `BaseAgent.complete` (`:536`) | the recall prefix, always prepended first |
| `_maybe_compact` (`:538`) | compaction, over the final flat list |

### Gaps

1. **No ordering authority.** Recall is hardcoded first; everything else is whatever the
   agent happened to write.
2. **No cross-source budget.** Compaction receives a flat list with no idea what came from
   where, so it summarises by *recency alone*. It cannot prefer dropping stale knowledge
   over recent turns, which is usually the right trade.
3. **No provenance.** Nothing can answer *"which source filled this window?"* — precisely
   the question when a prompt gets expensive.
4. **`pinned` is a proxy.** S5a pins `role == "system"` because role is the only signal
   available. Pinning belongs to the **source**, not the role: a knowledge block and the
   recall block are both system messages, and only one of them is load-bearing.

---

## 2. Decision (settled in review)

**The compiler wraps what the agent produced.** The agent keeps building its own task
messages and passing them to `complete()`; the compiler owns everything *around* them and
treats the agent's list as one pinned source ordered last.

| Rejected alternative | Why |
|---|---|
| Compiler owns assembly end to end | The fullest V3 vision, but it changes how every agent works, breaks `complete(messages)` as the seam, and needs a migration for `agents/` and every lab round. Far larger than E4 was scoped as. |
| Compiler opt-in beside the current path | Lowest risk, but recreates exactly the duplicated assembly path V3 spent six slices removing — and compaction would then live in two places. |

Two properties this preserves, both hard-won:

- **`complete(messages)` keeps its signature.** Zero change to any existing agent.
- **S5a's single call site survives.** Compaction is *absorbed* as a drop policy rather
  than rewritten — the contract pinned in the V3 spec §1.1 before S5 was written.

---

## 3. Architecture

### 3.1 `ContextSource` (Protocol)

```python
class ContextSource(Protocol):
    name: str
    order: int      # low emits first; the assembly authority
    pinned: bool    # survives the drop policy
    def emit(self) -> list[Message]: ...
```

Pinning moves from role to **source**, which is the point. The recall block is pinned
because recall-as-data is the S2a anti-poisoning contract; a knowledge block is not.

### 3.2 `ContextCompiler`

```python
def compile(sources: Sequence[ContextSource], *, max_tokens: int,
            summarize: Callable[[list[Message]], str] | None) -> CompileResult
```

1. Emit every source in `order`.
2. Estimate tokens (reusing `memory.compaction.estimate_tokens`).
3. Under ceiling → return unchanged, and **call nothing**.
4. Over ceiling → apply the drop policy: **droppable sources are dropped
   lowest-order-first**; only if that is not enough is the remaining droppable content
   summarised. Pinned sources are never touched.

### 3.3 `CompileResult`

Carries `messages`, plus `contributions: list[SourceContribution]` (name, tokens,
dropped, summarised). That is the provenance gap closed: an operator can see which source
filled the window.

### 3.4 Standard sources (S1)

| Source | Order | Pinned | Notes |
|---|---|---|---|
| `KnowledgeSource` | 10 | no | Present only when an agent supplies knowledge |
| `RecallSource` | 20 | **yes** | S2a contract — never dropped, never summarised |
| `AgentMessages` | 90 | **yes** | What the agent passed to `complete()` |

`AgentMessages` being pinned means today's behaviour is preserved exactly when only it and
recall are present — which is every agent that does not use knowledge.

### 3.5 Compaction becomes a drop policy

`memory.compaction.compact` is *called by* the compiler rather than by `complete()`. The
pure function is unchanged — this is the absorption the V3 spec §1.1 designed for, and
the reason S5a was built as a pure function with an injected `summarize`.

---

## 4. Slice plan

| Slice | Delivers | Lab |
|---|---|---|
| **S1** | `ContextSource`, `ContextCompiler`, `CompileResult`, the three standard sources, wired into `complete()` replacing the prefix+compact lines | **R34** |
| **S2** | Reflection as a real module — the compiler removes the `self.complete` re-entry that blocked it in V3 S5 — and the `base_agent` import reversal that unblocks | **R35** |
| **S3** | Release: bump 3.1.0, CHANGELOG, tag | full regression |

---

## 5. Invariants

- **`complete(messages)` signature unchanged.** Any agent that does not opt into knowledge
  sees byte-identical prompts.
- **Recall stays pinned.** The S2a anti-poisoning contract is a source property now, and a
  test asserts recall is never dropped or summarised.
- **No LLM call when under ceiling.** The cheap estimate runs first, exactly as S5a's guard
  does — compaction must not double the cost of every short run.
- **Best-effort at the caller.** A compiler failure sends the un-compiled prompt and warns;
  a `TokenCapExceeded` propagates. Same split S5a established.
- **Rule 7b gate per slice**, one PR each, one lab round each.

---

## 6. Definition of Done (v3.1.0)

- Provenance queryable: `CompileResult.contributions` names every source and its cost.
- Compaction drops **stale knowledge before recent turns** — the decision it could not
  make before.
- Reflection is a module (S2).
- `core/base_agent.py` subsystem imports reduced from **5**; the V3 epic metric that
  E4 was always going to close.
- R34–R35 green, full regression green, `v3.1.0` tagged.
