# ADR 007: In-Memory Rate Limiting

## Status
Accepted (with documented limitations)

## Context
Need rate limiting on zero-cost infrastructure. Redis costs $15+/mo minimum.

## Decision
Use in-memory sliding window per IP (30 req/min). Resets on container restart.

## Limitations
- State resets on Render cold start (~15 min sleep cycle)
- Not distributed across multiple workers
- Determined bot can theoretically wait for cold start

## Mitigations
- 30 req/min is conservative for a portfolio demo
- Groq API has independent rate limiting (1M tokens/day)
- No PII or financial data at risk
- API key gate on POST endpoints adds a second independent layer
- Upgrade path: Redis + sliding window counter (or Upstash Redis free tier) for production
