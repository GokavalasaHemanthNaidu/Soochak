# ADR 008: Model Artifact Versioning

## Status
Accepted

## Context
ONNX and PKL files are binary; they bloat git history and break diffs.

## Decision
- Only the current production model is committed to `ml_artifacts/`
- Old versions stored in GitHub Releases
- `.gitattributes` marks `*.onnx` and `*.pkl` as binary (no diff, no merge conflicts)
- XGBoost raw model excluded if >2MB (store in Releases)

## Consequences
- Positive: Repo stays <10MB, CI runs fast, no binary diffs
- Negative: Historical model rollbacks require manual download from Releases
- Upgrade: DVC + S3/GCS for full experiment tracking when >10 model versions
