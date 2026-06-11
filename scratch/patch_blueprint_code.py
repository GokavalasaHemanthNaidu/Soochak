import re
from pathlib import Path

blueprint_path = Path(r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md")
workspace_dir = Path(r"c:\Users\Hemanth\SOOCHAK")

with open(blueprint_path, "r", encoding="utf-8") as f:
    bp_content = f.read()

def read_workspace_file(rel_path: str) -> str:
    path = workspace_dir / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Read actual files from workspace
sql_initial = read_workspace_file("src/db/migrations/001_initial.sql")
req_models = read_workspace_file("src/models/request.py")
resp_models = read_workspace_file("src/models/response.py")
health_router = read_workspace_file("src/api/routers/health.py")
main_api = read_workspace_file("src/api/main.py")
ml_service = read_workspace_file("src/services/ml_service.py")
drift_logger = read_workspace_file("src/services/drift_logger.py")
predict_router = read_workspace_file("src/api/routers/predict.py")

# Helper to replace block in markdown
def replace_file_block(content: str, heading_prefix: str, filepath: str, new_code: str) -> str:
    escaped_fp = re.escape(filepath)
    # We want to match: ## Step X.Y: ... \n\n```bash\ncat > filepath << 'EOF'\n(any contents)\nEOF\n```
    pattern = r"(##\s*" + re.escape(heading_prefix) + r"[^\n]*\n+```bash\ncat\s*>\s*" + escaped_fp + r"\s*<<\s*'EOF'\n).*?(\nEOF\n```)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"WARNING: Could not find block for heading prefix '{heading_prefix}' and file '{filepath}'")
        return content
    replacement = r"\g<1>" + new_code + r"\g<2>"
    return re.sub(pattern, replacement, content, flags=re.DOTALL)

# Phase 2: SQL initial schema
bp_content = replace_file_block(bp_content, "Step 2.1:", "src/db/migrations/001_initial.sql", sql_initial)

# Phase 3: Pydantic request models
bp_content = replace_file_block(bp_content, "Step 3.3:", "src/models/request.py", req_models)

# Phase 3: Pydantic response models
bp_content = replace_file_block(bp_content, "Step 3.4:", "src/models/response.py", resp_models)

# Phase 3: Health router
bp_content = replace_file_block(bp_content, "Step 3.6:", "src/api/routers/health.py", health_router)

# Phase 3: Main API app
bp_content = replace_file_block(bp_content, "Step 3.9:", "src/api/main.py", main_api)

# Phase 5: inference.py and shap_service.py consolidation into ml_service.py
bp_content = bp_content.replace(
    "## Step 5.1: Create Feature Encoder Utility (With Shape Validation)\n\n```bash\ncat > src/utils/feature_encoder.py",
    "## Step 5.1: Create Eager Singleton ML Service (ONNX + Calibrator + SHAP)\n\n```bash\ncat > src/services/ml_service.py"
)
bp_content = replace_file_block(bp_content, "Step 5.1:", "src/services/ml_service.py", ml_service)

bp_content = bp_content.replace(
    "## Step 5.2: Create Inference Service (Lazy Load + Shape Check)\n\n```bash\ncat > src/services/inference.py",
    "## Step 5.2: Create Drift Logger Service (Drift Event Tracking)\n\n```bash\ncat > src/services/drift_logger.py"
)
bp_content = replace_file_block(bp_content, "Step 5.2:", "src/services/drift_logger.py", drift_logger)

# Let's replace Step 5.3 with a note that shap_service.py is consolidated
bp_content = re.sub(
    r"## Step 5\.3: Create Real SHAP Service \(TreeExplainer\)\n\n```bash\ncat > src/services/shap_service\.py << 'EOF'\n.*?\nEOF\n```",
    "## Step 5.3: Consolidate Services\n\nSHAP computation and model inference are now consolidated in `MLService` under `ml_service.py` to optimize latency and facilitate eager initialization on server startup.",
    bp_content,
    flags=re.DOTALL
)

# Phase 5: predict router (normalized schema + eager ml service)
bp_content = bp_content.replace(
    "## Step 5.6: Update Predict Router with Real Inference + prediction_id\n\n```bash\ncat > src/api/routers/predict.py",
    "## Step 5.6: Update Predict Router with Eager ML Service & Two-Table Normalized Insert\n\n```bash\ncat > src/api/routers/predict.py"
)
bp_content = replace_file_block(bp_content, "Step 5.6:", "src/api/routers/predict.py", predict_router)

# 3. Add Colab Step 4.10.5 (F1-optimal threshold calculation)
step_4_10_5 = """## Step 4.10.5: Optimal Calibrated Threshold Calculation (F1-Score Optimization)

```python
# Cell 10.5: Find threshold that maximizes F1 score on calibrated validation set
import numpy as np

# We evaluate threshold range on the calibrated probabilities
thresholds = np.linspace(0.01, 0.99, 100)
best_threshold = 0.5
best_f1 = 0.0

for t in thresholds:
    y_pred_t = (prob_calibrated >= t).astype(int)
    f1 = f1_score(y_cal, y_pred_t)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Optimal Calibrated Threshold: {best_threshold:.4f} (Best F1: {best_f1:.4f})")

# Save this to the feature_metadata settings
# In Step 4.14 we will include this threshold under "inference_settings": {"optimal_threshold_calibrated": float(best_threshold)}
```

"""

if "## Step 4.10.5:" not in bp_content:
    bp_content = bp_content.replace("## Step 4.11: Isotonic Calibration", step_4_10_5 + "## Step 4.11: Isotonic Calibration")

with open(blueprint_path, "w", encoding="utf-8") as f:
    f.write(bp_content)

print("Blueprint code blocks successfully updated and patched from workspace!")
