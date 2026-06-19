# ADR 001: SQLite over PostgreSQL

## Status
Accepted

## Context
Need a database for <10K predictions/day, single writer, demo portfolio project on zero-cost infra.

## Decision
Use SQLite with WAL mode.

## Consequences
- Positive: Zero infrastructure, single file, no Docker networking needed
- Negative: No concurrent writes, no built-in replication, ephemeral on Render free tier
- Migration threshold: >50 concurrent writers or persistent data requirement → PostgreSQL (Neon/Supabase free tier)
