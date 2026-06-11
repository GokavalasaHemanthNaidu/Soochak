# ADR 006: Geographic CV over Random Split

## Status
Accepted

## Context
Road infrastructure varies massively between Indian states.

## Decision
Train on 5 states, test on Delhi holdout.

## Consequences
- Positive: Proves generalization across geographies
- Negative: Harder to implement, requires state-aware data pipeline
