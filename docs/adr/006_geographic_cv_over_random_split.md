# ADR 006: Geographic CV over Random Split

## Status
Accepted

## Context
Road infrastructure, speed limits, and driving behavior vary massively between Indian states.
A random split would leak state-specific patterns into the test set (data leakage).

## Decision
Train on 5 states, test on Delhi holdout (unseen geography).

## Consequences
- Positive: Proves generalization across geographies; no state-level data leakage
- Negative: Harder to implement; requires state-aware stratified sampling
- Interview answer: "Random split would let the model memorize Delhi road patterns during training,
  inflating test F1. Geographic CV gives honest out-of-distribution performance."
