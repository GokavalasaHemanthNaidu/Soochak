# ADR 002: ONNX over XGBoost Direct

## Status
Accepted

## Context
Need to decouple training framework from inference runtime.

## Decision
Export XGBoost to ONNX; use ONNX Runtime for inference. Keep XGBoost model for SHAP.

## Consequences
- Positive: ~2x faster CPU inference, language-agnostic, smaller dependency footprint
- Negative: Some advanced XGBoost features not supported in ONNX
- Mitigation: Save original XGBoost model alongside ONNX for TreeExplainer SHAP
