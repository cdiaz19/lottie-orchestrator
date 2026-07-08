# S4 Plan — HTTP hardening

Design: 2026-07-08-http-hardening-design.md. TDD, mypy --strict + ruff clean per task. Baseline 922.

- T1 `serve/pagination.py`: `page_bounds(query) -> (limit, offset)` (clamp, garbage→defaults, MAX_LIMIT=100). Wire into list_agents + list_models. Tests.
- T2 `serve/auth.py` ApiKeyAuthMiddleware: env LOTTIE_API_KEYS (per-request), open-when-unset, Bearer + X-API-Key, hmac.compare_digest, 401 no-echo. Tests.
- T3 `serve/ratelimit.py` RateLimitMiddleware: env LOTTIE_RATE_LIMIT_PER_MIN, token bucket per key/host, 429, monotonic. Tests.
- T4 `http_app.build_http_app`: middleware=[RateLimit(outer), Auth]. Regression: full serve suite green envs-unset.
- T5 Full gate + whole-diff review.

## Lab R18
Downstream: keys unset→open; set→Bearer/X-API-Key 200, missing/wrong 401; rate limit N then 429; pagination limit/offset on /v1/agents + /v1/models.
