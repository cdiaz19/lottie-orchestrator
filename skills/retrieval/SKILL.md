# RetrievalSkill

## What it does

Embeds the query text via an injected `EmbeddingProvider`, retrieves the top-k
scored hits from an injected `VectorStore`, and applies optional layer/tag
filters declared in the `RetrievalQuery`.  No LLM is involved.  Results are
fully deterministic given a fixed store and embedder.

## Dependency injection

`RetrievalSkill(embedder, store)` — the skill never constructs its own embedder
or store.  This keeps it testable (swap `MockEmbeddingProvider` +
`InMemoryVectorStore` in tests) and enforces the Golden Rule: agents reach the
vector store only through this skill.

## Input

| Field | Type | Required | Description |
|---|---|---|---|
| query | RetrievalQuery | yes | Text to embed plus retrieval parameters |

### RetrievalQuery fields used

| Field | Type | Default | Description |
|---|---|---|---|
| text | str | — | Query text to embed |
| k | int | 5 | Maximum number of hits to return |
| layers | list[KnowledgeLayer] | [] | Layer allowlist; empty = no filter |
| tags | list[str] | [] | Tag intersection filter; empty = no filter |
| expand_graph | bool | False | Reserved for graph expansion (Task 13 hybrid retrieval); ignored here |

## Output

| Field | Type | Description |
|---|---|---|
| result | RetrievalResult | Ordered list of `RetrievalHit` (chunk + cosine score) |

## Side effects

None.  Read-only access to the store.

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
