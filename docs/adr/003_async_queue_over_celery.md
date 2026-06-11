# ADR 003: Async Queue over Celery

## Status
Accepted

## Context
Need background batch processing without Redis broker overhead.

## Decision
Use asyncio.Queue + FastAPI lifespan consumer.

## Consequences
- Positive: No Redis needed, in-process, demonstrates same producer-consumer pattern
- Negative: Tasks lost on restart, no distributed workers
- Upgrade: Celery + Redis when >1000 batch jobs/day
