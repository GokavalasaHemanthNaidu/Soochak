# ADR 005: Indian Dataset over UK Stats19

## Status
Accepted

## Context
Choice between clean UK Stats19 data and messy Indian FIR data with 15–30% missing values.

## Decision
Use messy Indian dataset.

## Consequences
- Positive: Demonstrates real-world data cleaning, imputation, geographic CV — higher portfolio signal
- Negative: Lower F1 (~0.47 vs ~0.60 on Stats19), more preprocessing code needed
- Portfolio value: Higher — shows you can handle production-quality messy data, not just Kaggle notebooks

## Known Limitation: Fog/Target Encoding

The model assigns `0.0000` fatal risk to "Fog or mist" weather because there were zero fatal fog accidents
in the training states. This is a target encoding failure on rare categories, not a model bug.

**How to fix (if asked in interview):**
1. **Smoothed target encoding** (add-k smoothing): blend category mean with global mean weighted by sample count.
   Rare categories pull toward the global mean (15.4%) instead of collapsing to 0.
2. **Leave-one-out encoding**: removes target leakage and handles rare categories more robustly.
3. **Simple fix**: replace target encoding for weather with frequency encoding or ordinal risk encoding
   based on domain knowledge (fog = high risk regardless of sample count).

Current mitigation: threshold of 0.18 is calibrated against reliable categories, so fog predictions
will default to "non-fatal" — conservative and safer than false fatals.
