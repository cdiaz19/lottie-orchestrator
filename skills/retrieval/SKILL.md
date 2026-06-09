# RetrievalSkill

## What it does

Embeds the query text via an injected `EmbeddingProvider`, retrieves the top-k
scored hits from an injected `VectorStore`, and applies optional layer/tag
filters declared in the `RetrievalQuery`.  When a `GraphStore` is also injected
and `expand_graph=True`, the skill additionally appends discounted chunks from
documents that the top hits *depend on* (their `depends_on` neighbours).

No LLM is involved.  Results are fully deterministic given a fixed store,
embedder, and graph.

## Dependency injection

`RetrievalSkill(embedder, store)` or `RetrievalSkill(embedder, store, graph)` —
the skill never constructs its own embedder, store, or graph.  This keeps it
testable (swap `MockEmbeddingProvider` + `InMemoryVectorStore` + `GraphStore`
in tests) and enforces the Golden Rule: agents reach the vector store only
through this skill.

## Input

| Field | Type | Required | Description |
|---|---|---|---|
| query | RetrievalQuery | yes | Text to embed plus retrieval parameters |

### RetrievalQuery fields used

| Field | Type | Default | Description |
|---|---|---|---|
| text | str | — | Query text to embed |
| k | int | 5 | Maximum number of base hits; expansion may add more |
| layers | list[KnowledgeLayer] | [] | Layer allowlist; empty = no filter |
| tags | list[str] | [] | Tag intersection filter; empty = no filter |
| expand_graph | bool | False | When True and a GraphStore was injected, adds dependency chunks |

## Output

| Field | Type | Description |
|---|---|---|
| result | RetrievalResult | Ordered list of `RetrievalHit` (chunk + cosine score), sorted by score DESC then chunk.id ASC |

## Graph expansion contract

When `expand_graph=True` and a `GraphStore` is available:

1. **Base hits** — the normal top-`k` vector hits are computed first.
2. **Neighbour collection** — for each base-hit doc, `GraphStore.neighbors(doc_id)`
   returns its direct `depends_on` dependencies (sorted).  Neighbours already
   represented in the base hits are excluded.
3. **Wide re-query (v1)** — a second query with `k = max(q.k, store.count())`
   retrieves the full corpus rank.  For each remaining neighbour doc id (sorted
   for determinism), the first chunk matching that `doc_id` is selected.
   This avoids adding a `get_by_doc_id` API to `VectorStore` and keeps both
   layers independently replaceable.  The trade-off — a second O(n) scan — is
   acceptable for `InMemoryVectorStore` corpora (< ~200 chunks).
4. **Discount** — each expansion hit's score is multiplied by
   `GRAPH_EXPANSION_DISCOUNT = 0.5`, ensuring base hits always rank above
   expansion hits at the same raw similarity.
5. **Additive** — expansion chunks are appended; the result may exceed `q.k`.
6. **De-duplication** — chunk ids already in base hits are never repeated.

If `expand_graph=False`, or no `GraphStore` was injected, the behaviour is
identical to the pre-expansion implementation (no performance overhead).

## Side effects

None.  Read-only access to the store and graph.

## Security

This skill must be declared in the agent's `capabilities` list.  Agents never
import `VectorStore` directly (`CapabilityEnforcerSkill` enforces this at
runtime).  All inputs should pass through `InputSanitizerSkill` before
reaching this skill.

## Examples

### Basic retrieval

```python
skill = RetrievalSkill(embedder=MockEmbeddingProvider(), store=InMemoryVectorStore())
out = skill.run(RetrievalSkillInput(query=RetrievalQuery(text="apple pie", k=2)))
# out.result.hits[0].chunk.text == "apple pie"
```

### Layer-filtered retrieval

```python
out = skill.run(
    RetrievalSkillInput(
        query=RetrievalQuery(text="...", k=5, layers=[KnowledgeLayer.GLOBAL])
    )
)
# Only GLOBAL-layer chunks are returned.
```

### Graph-expanded hybrid retrieval

```python
manifest = KnowledgeManifest(documents=[doc_a, doc_b])  # b depends_on a
graph = GraphStore(manifest)
skill = RetrievalSkill(embedder, store, graph)

out = skill.run(
    RetrievalSkillInput(
        query=RetrievalQuery(text="beta banana", k=1, expand_graph=True)
    )
)
# out.result.hits[0] — "b" chunk at full score (base hit)
# out.result.hits[1] — "a" chunk at score * 0.5 (graph expansion, b depends_on a)
```
