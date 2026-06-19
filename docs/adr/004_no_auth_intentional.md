# ADR 004: No Auth (Intentional for Demo)

## Status
Accepted

## Context
Portfolio demo API; no PII stored; stateless public endpoints; target audience is recruiters.

## Decision
No user authentication in v1. API key gate on POST endpoints for basic bot protection.

## Consequences
- Positive: Easier for recruiters to test without sign-up friction, no auth complexity
- Negative: No user isolation, no RBAC, shared rate limit pool
- Upgrade path: JWT + OAuth2 via Clerk or Supabase in v2 when multi-user needed
