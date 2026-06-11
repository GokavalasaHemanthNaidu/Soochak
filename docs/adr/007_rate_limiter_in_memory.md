# ADR 007: In-Memory Rate Limiting

## Status
Accepted (with documented limitations)

## Context
Need rate limiting on zero-cost infrastructure. Redis costs $15+/mo.

## Decision
Use in-memory sliding window per IP. Resets on container restart.

## Limitations
- Resets on Render cold start (15 min sleep)
- Not distributed across multiple workers
- Bot can theoretically wait for cold start

## Mitigation
- 30 req/min is conservative
- Groq API has its own rate limiting (1M tokens/day)
- No PII or financial data at risk
- API key gate on POST endpoints adds second layer
- Upgrade: Redis + sliding window counter for production
