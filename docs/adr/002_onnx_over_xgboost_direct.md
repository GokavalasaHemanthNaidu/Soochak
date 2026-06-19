# ADR 002: ONNX over XGBoost Direct

## Status
Accepted

## Context
Need to decouple training framework from inference runtime for faster CPU serving.

## Decision
Export XGBoost to ONNX; use ONNX Runtime for inference. Keep original XGBoost .pkl for SHAP TreeExplainer.

## Consequences
- Positive: ~2x faster CPU inference, language-agnostic runtime, smaller dependency footprint
- Negative: Some advanced XGBoost features not supported in ONNX opset
- Mitigation: Save original XGBoost model alongside ONNX for TreeExplainer; validate parity at export (tolerance 1e-5)
