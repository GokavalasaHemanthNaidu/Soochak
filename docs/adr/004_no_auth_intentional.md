# ADR 004: No Auth (Intentional for Demo)

## Status
Accepted

## Context
Portfolio demo API; no PII stored; stateless public endpoints.

## Decision
No user authentication in v1. API key gate on POST endpoints for bot protection.

## Consequences
- Positive: Easier for recruiters to test, no auth complexity
- Negative: No user isolation, no RBAC
- Upgrade path: JWT + OAuth2 (Clerk/Supabase) in v2
