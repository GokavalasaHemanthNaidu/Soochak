# ADR 005: Indian Dataset over UK Stats19

## Status
Accepted

## Context
Choice between clean UK Stats19 data and messy Indian FIR data.

## Decision
Use messy Indian dataset (15-30% missing values).

## Consequences
- Positive: Demonstrates real-world data cleaning, imputation, geographic CV
- Negative: Lower F1 score, more preprocessing code needed
- Portfolio value: Higher — shows you can handle messy data
