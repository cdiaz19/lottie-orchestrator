# S4 Design — HTTP hardening (auth + rate limit + pagination)

- **Date:** 2026-07-08
- **Epic:** v1.0.0 hardening — slice S4. Lab round R18.
- **Closes:** auth / rate-limit / pagination (context.md deferral) — makes the HTTP transport
  "minimum viable production."

## 1. Goal

The `lottie serve --port` HTTP app (OpenAI-compat + REST over one AgentService) has no auth,
no rate limit, and unbounded list endpoints. Add all three as **middleware + query params on
the shared app**, opt-in and back-compatible (defaults off → existing tests + lab drivers
unchanged).

## 2. Design

### 2.1 API-key auth (`serve/auth.py`, `ApiKeyAuthMiddleware`)
- Valid keys from env **`LOTTIE_API_KEYS`** (comma-separated), read per-request (testable via
  monkeypatch; supports rotation without restart).
- **Unset/empty → open** (no auth; dev + back-compat).
- Set → every request must present a valid key via **`Authorization: Bearer <k>`** OR
  **`X-API-Key: <k>`**. **Constant-time** compare (`hmac.compare_digest` against each key).
  Missing/invalid → `401` OpenAI-shaped JSON (`type=invalid_request_error`, code=`unauthorized`);
  message never echoes the presented token.

### 2.2 Rate limit (`serve/ratelimit.py`, `RateLimitMiddleware`)
- Per-minute cap from env **`LOTTIE_RATE_LIMIT_PER_MIN`** (int). **Unset/0 → no limit.**
- In-memory **token bucket per identity** (the presented API key if any, else `client.host`):
  capacity = rate, refill = rate/60 per second (monotonic clock). Over → `429` JSON
  (`type=rate_limit_error`, code=`rate_limited`). Documented as **per-process** (not distributed).
- **Outermost** middleware (protects the auth check from floods).

### 2.3 Pagination (`serve/pagination.py` + the two list routes)
- `GET /v1/agents` and `GET /v1/models` accept `limit` + `offset` query params.
- `page_bounds(query) -> (limit, offset)`: parse ints; `offset >= 0`; `limit` clamped to
  `1..MAX_LIMIT` (default `MAX_LIMIT = 100`); non-integers → the default (`limit=100, offset=0`).
- Response shape **unchanged** (same keys), the list is just sliced. Default limit ≥ typical
  agent count, so un-paginated callers still see everything (existing assertions hold).

### 2.4 Wiring (`http_app.build_http_app`)
```python
Starlette(routes=[...], middleware=[
    Middleware(RateLimitMiddleware),   # outermost
    Middleware(ApiKeyAuthMiddleware),
])
```
Both middleware live only under `serve/` (Starlette import stays lazy — never from
`serve/__init__`). `MAX_LIMIT`, env var names are module constants.

## 3. Tests (grow from 922)
- Auth: unset → open (200); set + valid Bearer → 200; set + valid X-API-Key → 200; set + missing
  → 401; set + wrong → 401; message has no token; constant-time path exercised.
- Rate limit: unset → unlimited; set to N → N pass then 429; separate identities have separate
  buckets; refill over time admits again (monotonic advanced via a seam or a tiny sleep).
- Pagination: limit/offset slice `/v1/agents` + `/v1/models`; defaults return all (≤100);
  garbage params → defaults; offset past end → empty.
- Regression: full serve/HTTP suite green with envs unset (no behaviour change).

## 4. Files
- New: `serve/auth.py`, `serve/ratelimit.py`, `serve/pagination.py` (+ tests).
- Edit: `serve/http_app.py` (middleware), `serve/rest_app.py` (paginate list_agents),
  `serve/openai_app.py` (paginate list_models). Doc note in README (deferred to S7 release).

## 5. Risks / decisions
- **Per-process** rate limit + in-memory buckets — honest "minimum viable production," not
  distributed. Documented.
- Auth **open-when-unset** (chosen) preserves the ~922 tests + lab TestClient rounds; production
  sets `LOTTIE_API_KEYS`. `lottie doctor` (S7) will warn when serving without keys.
- Rate-limit bucket map grows with distinct identities; bounded in practice, swept-free for V1
  (documented; a TTL/LRU is a follow-up).
