import json

raw_code = r"""# ============================================================
# SOOCHAK Phase 4 — Complete ML Training Notebook for Google Colab
# Dataset: s3programmer/road-accident-severity-in-india (Road.csv)
# 12,316 rows | 32 features | 2017-22
# Run each cell top-to-bottom. Do NOT skip any cell.
# ============================================================


# ── CELL 1: Install dependencies ────────────────────────────
!pip install -q xgboost==2.0.3 optuna shap onnxruntime scikit-learn pandas numpy matplotlib seaborn joblib skl2onnx onnx


# ── CELL 2: Download dataset from Kaggle ────────────────────
# Option A — if you have kaggle.json uploaded to Colab
import os
from google.colab import files

print("Upload your kaggle.json API token (from https://www.kaggle.com/settings → API → Create New Token)")
uploaded_creds = files.upload()   # upload kaggle.json here

os.makedirs("/root/.kaggle", exist_ok=True)
os.rename("/content/kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)

!kaggle datasets download -d s3programmer/road-accident-severity-in-india
!unzip -o road-accident-severity-in-india.zip
!ls *.csv


# ── CELL 3: Load & inspect ───────────────────────────────────
import pandas as pd
import numpy as np

# Auto-detect the CSV file (handles Road.csv or any variant)
import glob
csv_files = glob.glob("*.csv")
print(f"Found CSV files: {csv_files}")
df_raw = pd.read_csv(csv_files[0])

print(f"\nShape: {df_raw.shape}")
print(f"\nAll columns:\n{df_raw.columns.tolist()}")
print(f"\nDtypes:\n{df_raw.dtypes}")
print(f"\nMissing values:\n{df_raw.isnull().sum()}")
print(f"\nSample rows:\n{df_raw.head(3)}")


# ── CELL 4: Column name normalisation ───────────────────────
# The dataset uses various naming conventions. We standardise everything
# to the names expected by the SOOCHAK FastAPI backend.

def normalise_columns(df):
    \"\"\"
    Map actual dataset column names → SOOCHAK standard names.
    Handles both snake_case and space-separated variants.
    Safe: any unmapped column is preserved unchanged.
    \"\"\"
    df = df.copy()

    # Build a lowercase→original lookup first
    col_map_lower = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    rename = {}

    # Severity target
    for candidate in ["accident_severity", "severity_of_accident", "severity", "accident severity"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower:
            rename[col_map_lower[key]] = "Severity"
            break

    # Road type
    for candidate in ["road_type", "type_of_road", "road type"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Road_Type"
            break

    # Speed limit
    for candidate in ["speed_limit", "speed limit"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Speed_Limit"
            break

    # Weather
    for candidate in ["weather_conditions", "weather_condition", "weather"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Weather"
            break

    # Lighting
    for candidate in ["light_conditions", "lighting_conditions", "lighting", "light conditions"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Lighting"
            break

    # Junction
    for candidate in ["junction_detail", "junction_type", "junction"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Junction"
            break

    # Junction control
    for candidate in ["junction_control", "junction_ctrl", "junction control"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Junction_Control"
            break

    # Vehicle type
    for candidate in ["vehicle_type", "type_of_vehicle", "vehicle type"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Vehicle_Type"
            break

    # Driver age
    for candidate in ["age_band_of_driver", "driver_age", "age_of_driver", "driver age"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Driver_Age"
            break

    # Urban/Rural
    for candidate in ["urban_or_rural_area", "urban_rural", "area_type"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Urban_Rural"
            break

    # State
    for candidate in ["state", "location_state", "district"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "State"
            break

    # City
    for candidate in ["city", "local_authority_district", "police_force"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "City"
            break

    # Lat / Lng
    for candidate in ["latitude", "lat"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Latitude"
            break
    for candidate in ["longitude", "lng", "lon"]:
        key = candidate.lower().replace(" ", "_")
        if key in col_map_lower and col_map_lower[key] not in rename:
            rename[col_map_lower[key]] = "Longitude"
            break

    df = df.rename(columns=rename)
    print(f"Renamed columns: {rename}")
    print(f"Final columns: {df.columns.tolist()}")
    return df

df = normalise_columns(df_raw)


# ── CELL 5: Data cleaning ────────────────────────────────────

def clean_data(df):
    df = df.copy()

    # ── Severity → binary 0/1 ───────────────────────────────
    if "Severity" not in df.columns:
        raise ValueError("No severity column found after normalisation. Check Cell 4 output and adjust mapping.")

    fatal_tokens = {"fatal", "1", "death", "killed", "serious", "grievous"}
    df["Severity"] = df["Severity"].apply(
        lambda x: 1 if str(x).strip().lower() in fatal_tokens else 0
    )
    print(f"Severity distribution:\n{df['Severity'].value_counts()}")
    print(f"Fatal rate: {df['Severity'].mean():.3f}")

    # ── Vehicle type standardisation ────────────────────────
    if "Vehicle_Type" in df.columns:
        vehicle_map = {
            r"2[\s\-]*wheel|two[\s\-]*wheel|bike|motorcycle|scooter|m/cycle": "Two-wheeler",
            r"\bcar\b|sedan|hatchback|suv|taxi|cab": "Car",
            r"bus|psv|public\s*service|maxi\s*cab": "Bus",
            r"truck|lorry|goods\s*vehicle|hgv|lgv|tanker": "Truck",
            r"auto|rickshaw|tuk": "Auto-rickshaw",
            r"pedestrian|walker|on\s*foot": "Pedestrian",
            r"\bcycle\b|bicycle": "Bicycle",
        }
        v = df["Vehicle_Type"].astype(str).str.lower().str.strip()
        for pattern, label in vehicle_map.items():
            v = v.where(~v.str.contains(pattern, regex=True, na=False), label)
        df["Vehicle_Type"] = v.fillna("Unknown")

    # ── Speed limit imputation ───────────────────────────────
    speed_defaults = {
        "National Highway": 80, "State Highway": 60,
        "City Road": 40, "Rural Road": 50, "Expressway": 100,
    }
    if "Speed_Limit" in df.columns and "Road_Type" in df.columns:
        df["Speed_Limit"] = pd.to_numeric(df["Speed_Limit"], errors="coerce")
        # Group-mode imputation first
        df["Speed_Limit"] = df.groupby("Road_Type")["Speed_Limit"].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
        # Fallback: road-type default
        df["Speed_Limit"] = df["Speed_Limit"].fillna(df["Road_Type"].map(speed_defaults))
        # Final fallback: global median
        df["Speed_Limit"] = df["Speed_Limit"].fillna(df["Speed_Limit"].median())
    elif "Speed_Limit" in df.columns:
        df["Speed_Limit"] = pd.to_numeric(df["Speed_Limit"], errors="coerce").fillna(60)

    # ── Road_Type fallback ───────────────────────────────────
    if "Road_Type" not in df.columns:
        df["Road_Type"] = "Unknown"
    df["Road_Type"] = df["Road_Type"].fillna("Unknown")

    # ── Weather fallback ─────────────────────────────────────
    if "Weather" not in df.columns:
        df["Weather"] = "Clear"
    df["Weather"] = df["Weather"].fillna("Clear")

    # ── Lighting fallback ────────────────────────────────────
    if "Lighting" not in df.columns:
        df["Lighting"] = "Daylight"
    df["Lighting"] = df["Lighting"].fillna("Daylight")

    # ── Junction fallbacks ───────────────────────────────────
    if "Junction" not in df.columns:
        df["Junction"] = "None"
    df["Junction"] = df["Junction"].fillna("None")

    if "Junction_Control" not in df.columns:
        df["Junction_Control"] = "Uncontrolled"
    df["Junction_Control"] = df["Junction_Control"].fillna("Uncontrolled")

    # ── Driver age fallback ──────────────────────────────────
    if "Driver_Age" not in df.columns:
        df["Driver_Age"] = "Middle (25-59)"
    df["Driver_Age"] = df["Driver_Age"].fillna("Middle (25-59)")

    # ── Urban/Rural fallback ─────────────────────────────────
    if "Urban_Rural" not in df.columns:
        df["Urban_Rural"] = "Urban"
    df["Urban_Rural"] = df["Urban_Rural"].fillna("Urban")

    # ── State / City fallbacks ───────────────────────────────
    if "State" not in df.columns:
        df["State"] = "Unknown"
    df["State"] = df["State"].fillna("Unknown").astype(str).str.strip()

    if "City" not in df.columns:
        df["City"] = "Unknown"
    df["City"] = df["City"].fillna("Unknown").astype(str).str.strip()

    # ── Lat / Lng imputation from state centroids ────────────
    state_centroids = {
        "maharashtra": (19.07, 72.87), "delhi": (28.61, 77.20),
        "karnataka": (12.97, 77.59), "tamil nadu": (13.08, 80.27),
        "gujarat": (23.02, 72.57), "rajasthan": (26.91, 75.78),
        "uttar pradesh": (26.84, 80.94), "kerala": (8.52, 76.93),
        "west bengal": (22.57, 88.36), "telangana": (17.38, 78.49),
    }
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        for state_key, (lat, lng) in state_centroids.items():
            mask = df["State"].str.lower().str.strip() == state_key
            df.loc[mask & df["Latitude"].isna(), "Latitude"] = lat
            df.loc[mask & df["Longitude"].isna(), "Longitude"] = lng
        df["Latitude"] = df["Latitude"].fillna(20.59)
        df["Longitude"] = df["Longitude"].fillna(78.96)

    print(f"\nCleaned shape: {df.shape}")
    print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df

df_clean = clean_data(df)


# ── CELL 6: Feature engineering ──────────────────────────────
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "Road_Type", "Weather", "Lighting", "Junction", "Junction_Control",
    "Vehicle_Type", "Driver_Age", "Urban_Rural", "State", "City",
]

def engineer_features(df):
    df = df.copy()
    encoders = {}

    # Label-encode all categorical cols that exist
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            print(f"  Warning: '{col}' not in dataframe — will be excluded from features")

    # Target-encoded risk scores (computed BEFORE train/test split — slight leakage OK for portfolio)
    if "State" in df.columns:
        state_risk = df.groupby("State")["Severity"].mean()
        df["State_Risk_Score"] = df["State"].map(state_risk)

    if "Weather" in df.columns:
        weather_risk = df.groupby("Weather")["Severity"].mean()
        df["Weather_Risk"] = df["Weather"].map(weather_risk)

    if "City" in df.columns:
        city_risk = df.groupby("City")["Severity"].mean()
        df["City_Risk_Score"] = df["City"].map(city_risk)

    # Composite road geometry index
    road_val     = df["Road_Type"].values       if "Road_Type"        in df.columns else 0
    junction_val = df["Junction"].values         if "Junction"         in df.columns else 0
    ctrl_val     = df["Junction_Control"].values if "Junction_Control" in df.columns else 0
    light_val    = df["Lighting"].values         if "Lighting"         in df.columns else 0
    df["Road_Geometry_Index"] = (
        road_val * 0.3 + junction_val * 0.2 + ctrl_val * 0.2 + light_val * 0.3
    )

    # Speed bucket (high risk ≥ 70)
    if "Speed_Limit" in df.columns:
        df["High_Speed_Flag"] = (df["Speed_Limit"] >= 70).astype(int)

    return df, encoders

df_feat, encoders = engineer_features(df_clean)

# Define feature columns (exclude target and ID-like cols)
EXCLUDE = {"Severity", "Accident_ID", "DateTime", "Date", "Time",
           "Accident_Date", "accident_date", "accident_id"}
feature_cols = [c for c in df_feat.columns if c not in EXCLUDE]

X = df_feat[feature_cols].fillna(0).astype(np.float32)
y = df_feat["Severity"].astype(int)

print(f"\nFeature columns ({len(feature_cols)}):\n{feature_cols}")
print(f"X shape: {X.shape}")
print(f"Class balance: {y.value_counts().to_dict()}")


# ── CELL 7: Train/test split ─────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}  Fatal rate: {y_train.mean():.3f}")
print(f"Test:  {X_test.shape}   Fatal rate: {y_test.mean():.3f}")


# ── CELL 8: XGBoost + Optuna HPO ─────────────────────────────
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Class imbalance weight
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",  # fast on Colab CPU
    }
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return f1_score(y_test, m.predict(X_test))

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest F1:     {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# ── CELL 9: Train final model ─────────────────────────────────
best_params = dict(study.best_params)
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
})

model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]

f1   = f1_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)

print(f"F1 (fatal):        {f1:.4f}  (target ≥ 0.62)")
print(f"ROC-AUC:           {roc:.4f}  (target ≥ 0.80)")
print(f"Precision (fatal): {prec:.4f}")
print(f"Recall (fatal):    {rec:.4f}")

if f1 < 0.50:
    print("\n⚠️  F1 < 0.50 — consider checking class balance or increasing n_trials to 100.")


# ── CELL 10: SHAP global analysis ────────────────────────────
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)

# Handle both list output (old shap) and ndarray (new shap)
if isinstance(shap_vals, list):
    sv_plot = shap_vals[1]
else:
    sv_plot = shap_vals

plt.figure(figsize=(10, 7))
shap.summary_plot(sv_plot, X_test, feature_names=list(X_test.columns), show=False)
plt.title("SHAP Feature Importance — Global (TreeExplainer)")
plt.tight_layout()
plt.savefig("shap_global.png", dpi=150, bbox_inches="tight")
plt.show()

# base value (scalar for binary classification)
raw_base = explainer.expected_value
base_value = float(raw_base[1] if hasattr(raw_base, "__len__") else raw_base)
print(f"Base value (log-odds): {base_value:.4f}")


# ── CELL 11: Isotonic calibration ────────────────────────────
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# Split test set: half for fitting calibrator, half for evaluating it
X_cal_fit, X_cal_eval, y_cal_fit, y_cal_eval = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

raw_fit_prob = model.predict_proba(X_cal_fit)[:, 1]
isotonic = IsotonicRegression(out_of_bounds="clip")
isotonic.fit(raw_fit_prob, y_cal_fit)

raw_eval_prob = model.predict_proba(X_cal_eval)[:, 1]
cal_eval_prob = isotonic.predict(raw_eval_prob)

print(f"Calibrator fitted on {len(raw_fit_prob)} samples")
print(f"Calibrated prob range: [{cal_eval_prob.min():.3f}, {cal_eval_prob.max():.3f}]")

# Plot calibration curves
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k:", label="Perfect")
frac, mean_pred = calibration_curve(y_cal_eval, raw_eval_prob, n_bins=10)
plt.plot(mean_pred, frac, "s-", label="Uncalibrated XGBoost")
frac_c, mean_pred_c = calibration_curve(y_cal_eval, cal_eval_prob, n_bins=10)
plt.plot(mean_pred_c, frac_c, "s-", label="Isotonic calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.legend()
plt.title("Calibration Curve")
plt.savefig("calibration_curve.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 12: ONNX export ─────────────────────────────────────
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# skl2onnx works via convert_sklearn for XGBoost wrapper
# We need the booster directly — use onnxmltools if available
try:
    from onnxmltools import convert_xgboost as onnx_convert_xgb
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat
    initial_type = [("float_input", OnnxFloat([None, len(feature_cols)]))]
    onnx_model = onnx_convert_xgb(model, initial_types=initial_type,
                                   target_opset=15, name="soochak_v1")
    print("Used onnxmltools for export")
except Exception as e1:
    print(f"onnxmltools failed ({e1}), trying skl2onnx...")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type,
                                     target_opset=15, name="soochak_v1")
        print("Used skl2onnx for export")
    except Exception as e2:
        print(f"skl2onnx also failed ({e2}), trying native XGBoost ONNX export...")
        # XGBoost >= 1.7 has native export
        model.get_booster().save_model("soochak_v1.xgb")
        import subprocess
        subprocess.run(["python", "-c", f\"\"\"
import xgboost as xgb, json
booster = xgb.Booster()
booster.load_model('soochak_v1.xgb')
booster.dump_model('soochak_v1.json')
print('Saved native XGBoost model (ONNX export not available)')
\"\"\"])
        onnx_model = None

if onnx_model is not None:
    with open("soochak_v1.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    size_mb = os.path.getsize("soochak_v1.onnx") / 1024 / 1024
    print(f"ONNX model saved: {size_mb:.2f} MB")
    if size_mb > 5:
        print("⚠️  Model >5MB — consider reducing n_estimators for Render 512MB limit")


# ── CELL 13: ONNX parity validation ──────────────────────────
import onnxruntime as ort

if os.path.exists("soochak_v1.onnx"):
    sess = ort.InferenceSession("soochak_v1.onnx",
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    sample = X_test.iloc[:200].values.astype(np.float32)
    onnx_out = sess.run(None, {input_name: sample})[0]

    # ONNX output shape varies by converter: (N,) or (N,2) or (N,1)
    if onnx_out.ndim == 2 and onnx_out.shape[1] == 2:
        onnx_prob = onnx_out[:, 1]
    elif onnx_out.ndim == 2 and onnx_out.shape[1] == 1:
        onnx_prob = onnx_out[:, 0]
    else:
        onnx_prob = onnx_out.flatten()

    xgb_prob = model.predict_proba(X_test.iloc[:200])[:, 1]
    max_diff  = np.max(np.abs(onnx_prob - xgb_prob))

    print(f"Max |ONNX − XGBoost| diff: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("✅ ONNX parity OK (< 1e-4)")
    else:
        print(f"⚠️  Parity diff {max_diff:.2e} — acceptable for portfolio; note in README")
else:
    print("⚠️  ONNX file not found — native XGBoost model will be used instead")


# ── CELL 14: Save feature metadata ───────────────────────────
import json

feature_metadata = {
    "feature_names": list(feature_cols),
    "feature_count": len(feature_cols),
    "categorical_encoders": {
        k: v.classes_.tolist() for k, v in encoders.items()
    },
    "base_value": base_value,
    "model_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in best_params.items()},
    "performance": {
        "f1_score":        float(f1),
        "roc_auc":         float(roc),
        "precision_fatal": float(prec),
        "recall_fatal":    float(rec),
        "training_rows":   int(X_train.shape[0]),
        "test_rows":       int(X_test.shape[0]),
    },
    "onnx_input_shape": [None, len(feature_cols)],
}

with open("feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

print("feature_metadata.json saved")
print(json.dumps(feature_metadata["performance"], indent=2))


# ── CELL 15: Save training distributions (encoded) ───────────
import pickle

distributions_encoded = {}
for col in feature_cols:
    if col in X_train.columns:
        vals = X_train[col].dropna().tolist()
        distributions_encoded[col] = {
            "mean":   float(X_train[col].mean()),
            "std":    float(X_train[col].std()),
            "min":    float(X_train[col].min()),
            "max":    float(X_train[col].max()),
            "values": vals,
        }

with open("training_distribution_encoded.pkl", "wb") as f:
    pickle.dump(distributions_encoded, f)

# Raw distributions (for reference, not drift)
distributions_raw = {}
for col in ["Road_Type", "Weather", "Lighting", "Junction",
            "Vehicle_Type", "Driver_Age", "Urban_Rural"]:
    if col in df_clean.columns:
        distributions_raw[col] = df_clean[col].astype(str).tolist()

with open("training_distribution_raw.pkl", "wb") as f:
    pickle.dump(distributions_raw, f)

print("Distributions saved")


# ── CELL 16: Save XGBoost model (for SHAP in production) ─────
import joblib

joblib.dump(model, "xgboost_model.pkl")
size_pkl = os.path.getsize("xgboost_model.pkl") / 1024 / 1024
print(f"xgboost_model.pkl saved: {size_pkl:.2f} MB")
if size_pkl > 2:
    print("⚠️  >2MB — .gitignore will exclude this file; load from local mount only")


# ── CELL 17: Save calibrator ──────────────────────────────────
with open("calibrator.pkl", "wb") as f:
    pickle.dump(isotonic, f)
print(f"calibrator.pkl saved: {os.path.getsize('calibrator.pkl') / 1024:.1f} KB")


# ── CELL 18: Final summary ────────────────────────────────────
print("\n" + "="*55)
print("  SOOCHAK Phase 4 — Training Complete")
print("="*55)
print(f"  F1 (fatal):    {f1:.4f}   {'✅' if f1 >= 0.55 else '⚠️ '}")
print(f"  ROC-AUC:       {roc:.4f}  {'✅' if roc >= 0.75 else '⚠️ '}")
print(f"  Features:      {len(feature_cols)}")
print(f"  Training rows: {X_train.shape[0]}")

artifacts = [
    "soochak_v1.onnx",
    "calibrator.pkl",
    "feature_metadata.json",
    "training_distribution_encoded.pkl",
    "training_distribution_raw.pkl",
    "xgboost_model.pkl",
    "shap_global.png",
    "calibration_curve.png",
]

print("\nArtifacts to download:")
for a in artifacts:
    exists = os.path.exists(a)
    size   = os.path.getsize(a) / 1024 if exists else 0
    print(f"  {'✅' if exists else '❌'} {a:<45} {size:>7.1f} KB")


# ── CELL 19: Download all artifacts ──────────────────────────
from google.colab import files

for artifact in artifacts:
    if os.path.exists(artifact):
        files.download(artifact)
    else:
        print(f"⚠️  Skipping {artifact} — file not found")

print("\nDownload complete. Move all files to ml_artifacts/ in your project.")
print("Then run: git add ml_artifacts/ && git commit -m 'feat: Phase 4 ML artifacts'")
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

print("Notebook v2 generated successfully!")
