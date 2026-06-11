import json
import os

def create_cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "execution_count": None if cell_type == "code" else None,
        "outputs": [] if cell_type == "code" else None,
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

cells.append(create_cell("# Cell 1: Install dependencies\n!pip install -q xgboost optuna shap onnx onnxmltools scikit-learn pandas numpy matplotlib seaborn joblib"))

cells.append(create_cell("""# Cell 2: Upload your dataset
from google.colab import files
uploaded = files.upload()

# Cell 3: Load and inspect
import pandas as pd
import numpy as np

df = pd.read_csv(list(uploaded.keys())[0])
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\\n{df.isnull().sum()}")
print(f"Fatal rate: {df['Severity'].value_counts(normalize=True)}")"""))

cells.append(create_cell("""# Cell 4: Data cleaning
import re

def clean_indian_accident_data(df):
    \"\"\"Clean Indian road accident dataset.\"\"\"
    df = df.copy()

    vehicle_map = {
        r'2\\s*wheeler|two\\s*wheeler|bike|motorcycle|scooter': 'Two-wheeler',
        r'car|sedan|hatchback|suv': 'Car',
        r'bus|psv|public\\s*service': 'Bus',
        r'truck|lorry|goods\\s*vehicle|hgv': 'Truck',
        r'auto|rickshaw|tuk\\s*tuk': 'Auto-rickshaw',
        r'pedestrian|walker': 'Pedestrian',
        r'cycle|bicycle': 'Bicycle',
    }

    if 'Vehicle_Type' in df.columns:
        df['Vehicle_Type'] = df['Vehicle_Type'].astype(str).str.lower()
        for pattern, standard in vehicle_map.items():
            df.loc[df['Vehicle_Type'].str.contains(pattern, regex=True, na=False), 'Vehicle_Type'] = standard
        df['Vehicle_Type'] = df['Vehicle_Type'].fillna('Unknown')

    speed_map = {
        'National Highway': 80,
        'State Highway': 60,
        'City Road': 40,
        'Rural Road': 50,
        'Expressway': 100,
    }

    if 'Speed_Limit' in df.columns:
        df['Speed_Limit'] = df.groupby('Road_Type')['Speed_Limit'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 60)
        )
        df['Speed_Limit'] = df['Speed_Limit'].fillna(df['Road_Type'].map(speed_map))

    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        state_centroids = {
            'Maharashtra': (19.07, 72.87),
            'Delhi': (28.61, 77.20),
            'Karnataka': (12.97, 77.59),
            'Tamil Nadu': (13.08, 80.27),
            'Gujarat': (23.02, 72.57),
            'Rajasthan': (26.91, 75.78),
            'Uttar Pradesh': (26.84, 80.94),
            'Kerala': (8.52, 76.93),
        }
        for state, (lat, lng) in state_centroids.items():
            mask = (df['State'] == state) & df['Latitude'].isna()
            df.loc[mask, 'Latitude'] = lat
            df.loc[mask, 'Longitude'] = lng

    if 'Severity' in df.columns:
        df['Severity'] = df['Severity'].apply(
            lambda x: 1 if str(x).lower() in ['fatal', '1', 'death', 'killed'] else 0
        )

    return df

df_clean = clean_indian_accident_data(df)
print(f"Cleaned shape: {df_clean.shape}")
print(f"Fatal rate after cleaning: {df_clean['Severity'].mean():.3f}")"""))

cells.append(create_cell("""# Cell 5: Feature engineering
from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    \"\"\"Create engineered features for model.\"\"\"
    df = df.copy()

    categorical_cols = ['Road_Type', 'Weather', 'Lighting', 'Junction', 'Junction_Control',
                        'Vehicle_Type', 'Driver_Age', 'Urban_Rural', 'State', 'City']

    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    if 'State' in df.columns and 'Severity' in df.columns:
        state_risk = df.groupby('State')['Severity'].mean().to_dict()
        df['State_Risk_Score'] = df['State'].map(state_risk)

    if 'Weather' in df.columns:
        weather_risk = df.groupby('Weather')['Severity'].mean().to_dict()
        df['Weather_Risk'] = df['Weather'].map(weather_risk)

    df['Road_Geometry_Index'] = (
        df.get('Road_Type', 0) * 0.3 +
        df.get('Junction', 0) * 0.2 +
        df.get('Junction_Control', 0) * 0.2 +
        df.get('Lighting', 0) * 0.3
    )

    return df, encoders

df_features, encoders = engineer_features(df_clean)
print(f"Feature columns: {df_features.columns.tolist()}")"""))

cells.append(create_cell("""# Cell 6: Geographic train/test split
from sklearn.model_selection import train_test_split

feature_cols = [c for c in df_features.columns if c not in ['Severity', 'Accident_ID', 'DateTime']]
X = df_features[feature_cols].fillna(0)
y = df_features['Severity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Fatal rate: {y_train.mean():.3f}")
print(f"Test: {X_test.shape}, Fatal rate: {y_test.mean():.3f}")"""))

cells.append(create_cell("""# Cell 7: Optuna hyperparameter tuning
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")"""))

cells.append(create_cell("""# Cell 8: Train final model
best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'logloss'
best_params['random_state'] = 42
best_params['n_jobs'] = -1

model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(f"Precision (Fatal): {prec:.4f}")
print(f"Recall (Fatal): {rec:.4f}")"""))

cells.append(create_cell("""# Cell 9: SHAP global analysis
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.title("SHAP Feature Importance — Global")
plt.tight_layout()
plt.savefig("shap_global.png", dpi=150)
plt.show()

base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
print(f"Base value (expected value): {base_value}")"""))

cells.append(create_cell("""# Cell 10: Probability calibration
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

X_val, X_cal, y_val, y_cal = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
val_prob = model.predict_proba(X_val)[:, 1]

isotonic = IsotonicRegression(out_of_bounds='clip')
isotonic.fit(val_prob, y_val)

cal_prob = isotonic.predict(model.predict_proba(X_cal)[:, 1])
print(f"Calibration fitted on {len(val_prob)} samples")

prob_uncalibrated = model.predict_proba(X_cal)[:, 1]
prob_calibrated = cal_prob

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

fraction_of_positives, mean_predicted_value = calibration_curve(y_cal, prob_uncalibrated, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Uncalibrated")

fraction_of_positives, mean_predicted_value = calibration_curve(y_cal, prob_calibrated, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Isotonic calibrated")

plt.ylabel("Fraction of positives")
plt.xlabel("Mean predicted value")
plt.legend()
plt.title("Calibration Curve")
plt.savefig("calibration_curve.png", dpi=150)
plt.show()"""))

cells.append(create_cell("""# Cell 11: ONNX export
from skl2onnx import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
import os

initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]

onnx_model = convert_xgboost(
    model,
    initial_types=initial_type,
    target_opset=15,
    name="soochak_v1"
)

with open("soochak_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX model saved: {os.path.getsize('soochak_v1.onnx') / 1024 / 1024:.2f} MB")

import pickle
with open("calibrator.pkl", "wb") as f:
    pickle.dump(isotonic, f)

print(f"Calibrator saved: {os.path.getsize('calibrator.pkl') / 1024:.2f} KB")"""))

cells.append(create_cell("""# Cell 12: Validate ONNX parity
import onnxruntime as ort

session = ort.InferenceSession("soochak_v1.onnx")
input_name = session.get_inputs()[0].name

sample = X_test.iloc[:100].values.astype(np.float32)
onnx_prob = session.run(None, {input_name: sample})[0]

xgb_prob = model.predict_proba(X_test.iloc[:100])[:, 1]

max_diff = np.max(np.abs(onnx_prob.flatten() - xgb_prob))
print(f"Max difference between ONNX and XGBoost: {max_diff:.2e}")
assert max_diff < 1e-5, "ONNX parity failed!"
print("✅ ONNX parity validated!")"""))

cells.append(create_cell("""# Cell 13: Save feature metadata
import json

feature_metadata = {
    "feature_names": feature_cols,
    "feature_count": len(feature_cols),
    "categorical_encoders": {k: v.classes_.tolist() for k, v in encoders.items()},
    "base_value": base_value,
    "model_params": best_params,
    "performance": {
        "f1_score": f1,
        "roc_auc": roc,
        "precision_fatal": prec,
        "recall_fatal": rec,
    }
}

with open("feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

print("Feature metadata saved.")"""))

cells.append(create_cell("""# Cell 14: Save training distributions for drift detection
from scipy import stats
import pickle

# Encoded distributions (for drift detection against encoded live features)
distributions_encoded = {}
for col in feature_cols:
    if col in X_train.columns:
        distributions_encoded[col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "values": X_train[col].tolist()
        }

with open("training_distribution_encoded.pkl", "wb") as f:
    pickle.dump(distributions_encoded, f)

# Raw distributions for reference
distributions_raw = {}
for col in ['Road_Type', 'Weather', 'Lighting', 'Junction', 'Vehicle_Type', 'Driver_Age', 'Urban_Rural']:
    if col in df_clean.columns:
        distributions_raw[col] = df_clean[col].astype(str).tolist()

with open("training_distribution_raw.pkl", "wb") as f:
    pickle.dump(distributions_raw, f)

print("Training distributions saved for drift detection.")"""))

cells.append(create_cell("""# Cell 15: Save XGBoost model for real TreeExplainer SHAP in production
import joblib

joblib.dump(model, "xgboost_model.pkl")
print(f"XGBoost model saved: {os.path.getsize('xgboost_model.pkl') / 1024 / 1024:.2f} MB")"""))

cells.append(create_cell("""# Cell 16: Download all artifacts
from google.colab import files

files.download("soochak_v1.onnx")
files.download("calibrator.pkl")
files.download("feature_metadata.json")
files.download("training_distribution_encoded.pkl")
files.download("training_distribution_raw.pkl")
files.download("xgboost_model.pkl")
files.download("shap_global.png")
files.download("calibration_curve.png")"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"C:\Users\Hemanth\SOOCHAK\notebooks\soochak_ml_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
