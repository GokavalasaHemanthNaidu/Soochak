# ADR 003: asyncio.Queue over Celery

## Status
Accepted

## Context
Need background batch processing without Redis broker overhead on free tier.

## Decision
Use asyncio.Queue + FastAPI lifespan background consumer.

## Consequences
- Positive: No Redis needed, in-process, demonstrates same producer-consumer pattern, zero infra cost
- Negative: Tasks lost on container restart, no distributed workers, no retry logic
- Upgrade path: Celery + Redis when >1000 batch jobs/day or durability required
