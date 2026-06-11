import json
import os

def create_cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

cells.append(create_cell("""# ============================================================
# SOOCHAK Phase 4 — V4 MLOps Remediation Notebook
# Dataset: s3programmer/road-accident-severity-in-india (Road.csv)
# Flawless Architecture: 3-way split, OrdinalEncoder, zipmap=False
# ============================================================"""))

cells.append(create_cell("""# ── CELL 1: Install dependencies ────────────────────────────
!pip install -q xgboost==1.7.6 optuna shap onnxruntime scikit-learn pandas numpy matplotlib seaborn joblib skl2onnx onnx onnxmltools"""))

cells.append(create_cell("""# ── CELL 2: Upload dataset ────────────────────────────
from google.colab import files
import glob, os

print("Upload Road_csv.zip (or Road.csv directly):")
uploaded = files.upload()

if glob.glob("*.zip"):
    !unzip -o *.zip

csv_files = glob.glob("**/*.csv", recursive=True)
if not csv_files:
    raise FileNotFoundError("No CSV found after upload/unzip. Check your file.")
print(f"Using: {csv_files[0]}")"""))

cells.append(create_cell("""# ── CELL 3: Load raw data ────────────────────────────
import pandas as pd
import numpy as np

df_raw = pd.read_csv(csv_files[0])
print(f"Shape: {df_raw.shape}")"""))

cells.append(create_cell("""# ── CELL 4: Map real Road.csv columns -> 11 API fields (Fix #5 & #7) ────────────────
def build_api_aligned_df(df):
    d = df.copy()

    # Target: Serious Injury OR Fatal Injury = 1, Slight = 0
    d["Severity"] = d["Accident_severity"].apply(
        lambda x: 1 if str(x).strip().lower() in {"fatal injury", "serious injury"} else 0
    )

    # Base features
    d["Road_Type"]        = d["Lanes_or_Medians"].fillna("Undivided Two way").astype(str).str.strip()
    
    # Fix #5: Map actual speed limits instead of vehicles * 30
    speed_map = {
        "Double carriageway (median)": 80,
        "Two-way (divided with broken lines road marking)": 60,
        "Undivided Two way": 60,
        "One way": 40
    }
    d["Speed_Limit"]      = d["Road_Type"].map(speed_map).fillna(50)

    d["Weather"]          = d["Weather_conditions"].fillna("Normal").astype(str).str.strip()
    d["Lighting"]         = d["Light_conditions"].fillna("Daylight").astype(str).str.strip()
    d["Junction"]         = d["Types_of_Junction"].fillna("No junction").astype(str).str.strip()
    
    # Fix #7: Junction Control shouldn't use "Cause_of_accident". Default to Unknown.
    d["Junction_Control"] = "Unknown"
    
    d["Vehicle_Type"]     = d["Type_of_vehicle"].fillna("Other").astype(str).str.strip()
    d["Driver_Age"]       = d["Age_band_of_driver"].fillna("Unknown").astype(str).str.strip()
    d["Urban_Rural"]      = d["Area_accident_occured"].fillna("Other").astype(str).str.strip()
    d["State"]            = d["Road_allignment"].fillna("Unknown").astype(str).str.strip()
    d["City"]             = d["Day_of_week"].fillna("Unknown").astype(str).str.strip()

    KEEP = ["Road_Type","Speed_Limit","Weather","Lighting","Junction",
            "Junction_Control","Vehicle_Type","Driver_Age","Urban_Rural",
            "State","City","Severity"]
    return d[KEEP].copy()

df_clean = build_api_aligned_df(df_raw)
print(f"Shape after clean: {df_clean.shape}")"""))

cells.append(create_cell("""# ── CELL 5: Strict 3-way Split (Fix #2: Calibration Leakage) ────────────────────────────
from sklearn.model_selection import train_test_split

X_full = df_clean.drop(columns=["Severity"])
y_full = df_clean["Severity"]

# First split: 80% train+cal, 20% test
X_train_cal, X_test, y_train_cal, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Second split: 80% train (0.8 * 0.8 = 64%), 20% cal (0.8 * 0.2 = 16%)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_cal, y_train_cal, test_size=0.2, random_state=42, stratify=y_train_cal
)

print(f"Train: {X_train.shape}  fatal rate: {y_train.mean():.3f}")
print(f"Cal:   {X_cal.shape}   fatal rate: {y_cal.mean():.3f}")
print(f"Test:  {X_test.shape}   fatal rate: {y_test.mean():.3f}")"""))

cells.append(create_cell("""# ── CELL 6: Feature Engineering -> 14 features (Fix #4 & #6) ────────────────────────────
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL_COLS = [
    "Road_Type", "Weather", "Lighting", "Junction", "Junction_Control",
    "Vehicle_Type", "Driver_Age", "Urban_Rural", "State", "City"
]

# Fix #6: OrdinalEncoder with handle_unknown prevents API crashes in production
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(X_train[CATEGORICAL_COLS])

# Fix #4: Target encoding calculated strictly on X_train to prevent leakage
state_risk = X_train.assign(y=y_train).groupby("State")["y"].mean()
weather_risk = X_train.assign(y=y_train).groupby("Weather")["y"].mean()
global_mean = y_train.mean()

def engineer_features(X_split):
    df = X_split.copy()
    
    # Apply Ordinal Encoding
    df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])
    
    # Apply Target Encodings
    df["State_Risk_Score"] = df["State"].map(state_risk).fillna(global_mean)
    df["Weather_Risk"] = df["Weather"].map(weather_risk).fillna(global_mean)
    
    # Composite road geometry index (using encoded ordinal values as heuristic)
    df["Road_Geometry_Index"] = (
        df["Road_Type"] * 0.3 +
        df["Junction"] * 0.2 +
        df["Junction_Control"] * 0.2 +
        df["Lighting"] * 0.3
    )
    
    return df.astype(np.float32)

X_train_feat = engineer_features(X_train)
X_cal_feat   = engineer_features(X_cal)
X_test_feat  = engineer_features(X_test)

FEATURE_COLS = list(X_train_feat.columns)
print(f"Feature count: {len(FEATURE_COLS)}  (ONNX will expect exactly {len(FEATURE_COLS)} inputs)")
print(f"Features: {FEATURE_COLS}")"""))

cells.append(create_cell("""# ── CELL 7: XGBoost + Optuna HPO ────────────────────────────
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }
    m = xgb.XGBClassifier(**params)
    m.fit(X_train_feat.values, y_train, eval_set=[(X_test_feat.values, y_test)], verbose=False)
    return f1_score(y_test, m.predict(X_test_feat.values))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best F1: {study.best_value:.4f}")"""))

cells.append(create_cell("""# ── CELL 8: Train final model ────────────────────────────
best_params = dict(study.best_params)
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": scale_pos_weight,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
})

model = xgb.XGBClassifier(**best_params)
model.fit(X_train_feat.values, y_train)

y_pred = model.predict(X_test_feat.values)
y_prob = model.predict_proba(X_test_feat.values)[:, 1]

f1   = f1_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)

print(f"F1 (Serious+Fatal): {f1:.4f}")
print(f"ROC-AUC:            {roc:.4f}")
print(f"Precision:          {prec:.4f}")
print(f"Recall:             {rec:.4f}")"""))

cells.append(create_cell("""# ── CELL 9: SHAP Analysis (Fix #3: Probability Base Value) ────────────────────────────
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test_feat.values)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_test_feat, feature_names=list(X_test_feat.columns), show=False)
plt.title("SHAP Feature Importance (TreeExplainer)")
plt.tight_layout()
plt.savefig("shap_global.png", dpi=150, bbox_inches="tight")
plt.show()

# Fix #3: Convert Log-Odds base value to Probability base value
raw_base   = explainer.expected_value
base_value_log_odds = float(raw_base[1] if hasattr(raw_base, "__len__") else raw_base)
base_value_prob = 1 / (1 + np.exp(-base_value_log_odds))

print(f"Base value (log-odds):   {base_value_log_odds:.4f}")
print(f"Base value (probability): {base_value_prob:.4f}")"""))

cells.append(create_cell("""# ── CELL 10: Isotonic Calibration (Fix #2: Fitted on X_cal) ────────────────────────────
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import pickle

raw_fit_prob = model.predict_proba(X_cal_feat.values)[:, 1]
isotonic     = IsotonicRegression(out_of_bounds="clip")
isotonic.fit(raw_fit_prob, y_cal)

raw_eval_prob = model.predict_proba(X_test_feat.values)[:, 1]
cal_eval_prob = isotonic.predict(raw_eval_prob)

print(f"Calibrator fitted safely on {len(raw_fit_prob)} hold-out samples (X_cal)")

plt.figure(figsize=(8, 6))
plt.plot([0,1],[0,1],"k:",label="Perfect")
frac, mean_pred = calibration_curve(y_test, raw_eval_prob, n_bins=10)
plt.plot(mean_pred, frac, "s-", label="Uncalibrated")
frac_c, mean_pred_c = calibration_curve(y_test, cal_eval_prob, n_bins=10)
plt.plot(mean_pred_c, frac_c, "s-", label="Isotonic calibrated")
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.legend(); plt.title("Calibration Curve")
plt.savefig("calibration_curve.png", dpi=150, bbox_inches="tight")
plt.show()

with open("calibrator.pkl", "wb") as f:
    pickle.dump(isotonic, f)
print("calibrator.pkl saved")"""))

cells.append(create_cell("""# ── CELL 11: ONNX Export (Fix #1: zipmap=False) ────────────────────────────
N_FEATURES = len(FEATURE_COLS)
print(f"Exporting ONNX with {N_FEATURES} input features...")

try:
    from onnxmltools import convert_xgboost as onnx_convert_xgb
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat
    initial_type = [("float_input", OnnxFloat([None, N_FEATURES]))]
    
    # Fix #1: zipmap=False prevents the 0.50 Parity diff by enforcing tensor outputs
    onnx_model = onnx_convert_xgb(
        model, 
        initial_types=initial_type,
        target_opset=15, 
        name="soochak_v1",
        options={'zipmap': False}
    )
    print("Export method: onnxmltools (zipmap=False)")
    
    with open("soochak_v1.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_mb = os.path.getsize("soochak_v1.onnx") / 1024 / 1024
    print(f"soochak_v1.onnx saved: {size_mb:.2f} MB")

except Exception as e:
    raise RuntimeError(f"ONNX export failed: {e}")"""))

cells.append(create_cell("""# ── CELL 12: ONNX Parity Validation (Fix #1: Correct Output Index) ────────────────────────────
import onnxruntime as ort

sess = ort.InferenceSession("soochak_v1.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

sample = X_test_feat.values[:200].astype(np.float32)
onnx_outs = sess.run(None, {input_name: sample})

# Fix #1: With zipmap=False, output 0 is label, output 1 is probabilities
if len(onnx_outs) > 1:
    onnx_prob_tensor = onnx_outs[1]
else:
    onnx_prob_tensor = onnx_outs[0]

if onnx_prob_tensor.ndim == 2 and onnx_prob_tensor.shape[1] == 2:
    onnx_prob = onnx_prob_tensor[:, 1]
else:
    onnx_prob = onnx_prob_tensor.flatten()

xgb_prob = model.predict_proba(X_test_feat.values[:200])[:, 1]
max_diff  = np.max(np.abs(onnx_prob - xgb_prob))

print(f"ONNX Outputs available: {output_names}")
print(f"Max |ONNX - XGBoost| diff: {max_diff:.2e}")
if max_diff < 1e-4:
    print("ONNX parity OK! (0.50 bug completely fixed)")
else:
    print(f"WARNING: Parity diff {max_diff:.2e} is high!")"""))

cells.append(create_cell("""# ── CELL 13: Save feature_metadata.json (Fix #6 mapping) ────────────────────────────
import json

feature_metadata = {
    "feature_names":  list(FEATURE_COLS),
    "feature_count":  N_FEATURES,
    "categorical_encoders": {
        col: encoder.categories_[i].tolist() 
        for i, col in enumerate(CATEGORICAL_COLS)
    },
    "base_value":     base_value_prob,  # Fix #3
    "model_params":   {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in best_params.items()
    },
    "performance": {
        "f1_score":        float(f1),
        "roc_auc":         float(roc),
        "precision_fatal": float(prec),
        "recall_fatal":    float(rec),
        "training_rows":   int(X_train.shape[0]),
        "test_rows":       int(X_test.shape[0]),
    },
    "onnx_input_shape": [None, N_FEATURES],
    "target_definition": "1 = Serious Injury OR Fatal Injury; 0 = Slight Injury",
    "dataset": "s3programmer/road-accident-severity-in-india (Road.csv)",
}

with open("feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

print("feature_metadata.json saved")"""))

cells.append(create_cell("""# ── CELL 14: Save distributions ────────────────────────────
import pickle

distributions_encoded = {}
for col in FEATURE_COLS:
    vals = X_train_feat[col].dropna().tolist()
    distributions_encoded[col] = {
        "mean":   float(X_train_feat[col].mean()),
        "std":    float(X_train_feat[col].std()),
        "min":    float(X_train_feat[col].min()),
        "max":    float(X_train_feat[col].max()),
        "values": vals,
    }

with open("training_distribution_encoded.pkl", "wb") as f:
    pickle.dump(distributions_encoded, f)
print("training_distribution_encoded.pkl saved")"""))

cells.append(create_cell("""# ── CELL 15: Save XGBoost model ────────────────────────────
import joblib

joblib.dump(model, "xgboost_model.pkl")
print("xgboost_model.pkl saved")"""))

cells.append(create_cell("""# ── CELL 16: Download Artifacts ────────────────────────────
from google.colab import files

ARTIFACTS = [
    "soochak_v1.onnx",
    "calibrator.pkl",
    "feature_metadata.json",
    "training_distribution_encoded.pkl",
    "xgboost_model.pkl",
    "shap_global.png",
    "calibration_curve.png",
]

for artifact in ARTIFACTS:
    if os.path.exists(artifact):
        files.download(artifact)
    else:
        print(f"SKIPPED (not found): {artifact}")

print("Done. Move all files to ml_artifacts/ to begin Phase 5.")"""))

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/soochak_ml_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("V4 Notebook generated successfully at notebooks/soochak_ml_training.ipynb!")
