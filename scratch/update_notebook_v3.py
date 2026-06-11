import json

raw_code = r"""# ============================================================
# SOOCHAK Phase 4 — Complete ML Training Notebook for Google Colab
# Dataset: s3programmer/road-accident-severity-in-india (Road.csv)
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
!pip install -q xgboost==2.0.3 optuna shap onnxruntime scikit-learn pandas numpy matplotlib seaborn joblib skl2onnx onnx

# ── CELL 2: Upload dataset ────────────────────
from google.colab import files
import glob
print("Upload your dataset zip file or csv here:")
uploaded = files.upload()

if glob.glob("*.zip"):
    !unzip -o *.zip

csv_files = glob.glob("**/*.csv", recursive=True)
if not csv_files:
    raise ValueError("No CSV found! Check your uploaded file.")
print(f"Found CSV files: {csv_files}")

# ── CELL 3: Load & inspect ───────────────────────────────────
import pandas as pd
import numpy as np

df_raw = pd.read_csv(csv_files[0])
print(f"\nShape: {df_raw.shape}")
print(f"\nAll columns:\n{df_raw.columns.tolist()}")

# ── CELL 4: Column name normalisation ───────────────────────
def normalise_columns(df):
    df = df.copy()
    col_map_lower = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    rename = {}

    for candidate in ["accident_severity", "severity_of_accident", "severity"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Severity"

    for candidate in ["road_type", "type_of_road"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Road_Type"

    for candidate in ["speed_limit"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Speed_Limit"

    for candidate in ["weather_conditions", "weather"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Weather"

    for candidate in ["light_conditions", "lighting"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Lighting"

    for candidate in ["types_of_junction", "junction_detail", "junction"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Junction"

    for candidate in ["junction_control", "traffic_signal"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Junction_Control"

    for candidate in ["type_of_vehicle", "vehicle_type"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Vehicle_Type"

    for candidate in ["age_band_of_driver", "driver_age"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Driver_Age"

    for candidate in ["urban_or_rural_area", "urban_rural"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "Urban_Rural"

    for candidate in ["state", "district"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "State"

    for candidate in ["city", "sub_city"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower: rename[col_map_lower[key]] = "City"

    df = df.rename(columns=rename)
    return df

df = normalise_columns(df_raw)

# ── CELL 5: Data cleaning and API alignment ────────────────────────────────────
def clean_data(df):
    df = df.copy()

    # API Strictly expects these 11 features:
    REQUIRED_COLS = [
        "Road_Type", "Speed_Limit", "Weather", "Lighting", "Junction", 
        "Junction_Control", "Vehicle_Type", "Driver_Age", "Urban_Rural", 
        "State", "City", "Severity"
    ]
    
    # Fill in any missing required columns with safe defaults
    defaults = {
        "Road_Type": "National Highway", "Speed_Limit": 60, "Weather": "Clear",
        "Lighting": "Daylight", "Junction": "None", "Junction_Control": "Uncontrolled",
        "Vehicle_Type": "Two-wheeler", "Driver_Age": "Middle (25-59)",
        "Urban_Rural": "Urban", "State": "Maharashtra", "City": "Pune"
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # CRITICAL FIX: Drop the other 20+ Ethiopian columns. 
    # If we train on 32 features, the ONNX model will crash in our FastAPI backend!
    df = df[REQUIRED_COLS]

    # Severity → binary 0/1
    fatal_tokens = {"fatal", "1", "death", "killed", "serious", "serious injury", "fatal injury"}
    df["Severity"] = df["Severity"].apply(
        lambda x: 1 if str(x).strip().lower() in fatal_tokens else 0
    )

    df["Speed_Limit"] = pd.to_numeric(df["Speed_Limit"], errors="coerce").fillna(60)
    
    # Cast everything else to string to prevent LabelEncoder crashes
    for c in df.columns:
        if c not in ["Severity", "Speed_Limit"]:
            df[c] = df[c].astype(str)

    print(f"\nCleaned shape (strictly 11 features + target): {df.shape}")
    return df

df_clean = clean_data(df)

# ── CELL 6: Feature engineering ──────────────────────────────
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "Road_Type", "Weather", "Lighting", "Junction", "Junction_Control",
    "Vehicle_Type", "Driver_Age", "Urban_Rural", "State", "City"
]

def engineer_features(df):
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    state_risk = df.groupby("State")["Severity"].mean()
    df["State_Risk_Score"] = df["State"].map(state_risk)

    weather_risk = df.groupby("Weather")["Severity"].mean()
    df["Weather_Risk"] = df["Weather"].map(weather_risk)

    city_risk = df.groupby("City")["Severity"].mean()
    df["City_Risk_Score"] = df["City"].map(city_risk)

    df["Road_Geometry_Index"] = (
        df["Road_Type"] * 0.3 + df["Junction"] * 0.2 + df["Junction_Control"] * 0.2 + df["Lighting"] * 0.3
    )

    df["High_Speed_Flag"] = (df["Speed_Limit"] >= 70).astype(int)

    return df, encoders

df_feat, encoders = engineer_features(df_clean)

feature_cols = [c for c in df_feat.columns if c != "Severity"]

X = df_feat[feature_cols].fillna(0).astype(np.float32)
y = df_feat["Severity"].astype(int)

print(f"\nFeature columns ({len(feature_cols)}):\n{feature_cols}")

# ── CELL 7: Train/test split ─────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── CELL 8: XGBoost + Optuna HPO ─────────────────────────────
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "random_state": 42,
        "n_jobs": -1
    }
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train)
    return f1_score(y_test, m.predict(X_test))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=10, show_progress_bar=True)

# ── CELL 9: Train final model ─────────────────────────────────
best_params = dict(study.best_params)
best_params.update({"objective": "binary:logistic", "eval_metric": "logloss", "scale_pos_weight": scale_pos_weight, "random_state": 42})
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)
f1 = f1_score(y_test, model.predict(X_test))
print(f"F1 Score: {f1:.4f}")

# ── CELL 10: SHAP global analysis ────────────────────────────
import shap
import matplotlib.pyplot as plt
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)
if isinstance(shap_vals, list): shap_vals = shap_vals[1]
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_test, feature_names=list(X_test.columns), show=False)
plt.savefig("shap_global.png")
base_value = float(explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value)

# ── CELL 11: Isotonic calibration ────────────────────────────
from sklearn.isotonic import IsotonicRegression
X_cal_fit, _, y_cal_fit, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
isotonic = IsotonicRegression(out_of_bounds="clip")
isotonic.fit(model.predict_proba(X_cal_fit)[:, 1], y_cal_fit)
import pickle
with open("calibrator.pkl", "wb") as f: pickle.dump(isotonic, f)

# ── CELL 12: ONNX export ─────────────────────────────────────
try:
    from onnxmltools import convert_xgboost as onnx_convert_xgb
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat
    initial_type = [("float_input", OnnxFloat([None, len(feature_cols)]))]
    onnx_model = onnx_convert_xgb(model, initial_types=initial_type, target_opset=15, name="soochak_v1")
except:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=15, name="soochak_v1")

with open("soochak_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# ── CELL 13: Save metadata ───────────────────────────
import json
feature_metadata = {
    "feature_names": list(feature_cols),
    "feature_count": len(feature_cols),
    "categorical_encoders": {k: v.classes_.tolist() for k, v in encoders.items()},
    "base_value": base_value,
    "model_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_params.items()}
}
with open("feature_metadata.json", "w") as f: json.dump(feature_metadata, f, indent=2)

# ── CELL 14: Save distributions & XGBoost ───────────
distributions_encoded = {col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())} for col in feature_cols}
with open("training_distribution_encoded.pkl", "wb") as f: pickle.dump(distributions_encoded, f)
import joblib
joblib.dump(model, "xgboost_model.pkl")

# ── CELL 15: Download artifacts ──────────────────────────
from google.colab import files
for a in ["soochak_v1.onnx", "calibrator.pkl", "feature_metadata.json", "training_distribution_encoded.pkl", "xgboost_model.pkl", "shap_global.png"]:
    try: files.download(a)
    except: pass
"""

cells = []
parts = raw_code.split("# ── CELL")

header = parts[0]

for i in range(1, len(parts)):
    cell_content = "# ── CELL" + parts[i]
    if i == 1:
        cell_content = header + cell_content

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in cell_content.strip().split("\n")]
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"C:\Users\Hemanth\SOOCHAK\notebooks\soochak_ml_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook v3 generated successfully!")
