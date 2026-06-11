"""Production ML inference service for SOOCHAK.

CRITICAL FIX: The optimal threshold must be applied to CALIBRATED probabilities,
not raw probabilities. The isotonic calibrator compresses the output range
from ~0.60 (raw) down to ~0.15-0.25 (calibrated). A threshold of 0.35 on
calibrated probabilities would NEVER predict fatal. The correct calibrated
threshold is ~0.18, derived from the test set metrics (F1=0.2334).

Calibration behavior:
  Raw XGBoost output:    ~0.60 (systematically overconfident)
  Isotonic calibrated:   ~0.21 (matches true ~15.4% base rate)
  Decision threshold:    0.18 (maximizes F1 on calibration set)
"""

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import onnxruntime as ort
import shap
import asyncio
from src.services.drift_logger import log_drift_event

warnings.filterwarnings("ignore", category=UserWarning)

ML_ARTIFACTS = Path(__file__).parent.parent.parent / "ml_artifacts"
ONNX_PATH = ML_ARTIFACTS / "soochak_v1.onnx"
CALIBRATOR_PATH = ML_ARTIFACTS / "calibrator.pkl"
XGB_PATH = ML_ARTIFACTS / "xgboost_model.pkl"
META_PATH = ML_ARTIFACTS / "feature_metadata.json"

FEATURE_ORDER = [
    "Road_Type",
    "Speed_Limit",
    "Weather",
    "Lighting",
    "Junction",
    "Junction_Control",
    "Vehicle_Type",
    "Driver_Age",
    "Urban_Rural",
    "State",
    "City",
    "State_Risk_Score",
    "Weather_Risk",
    "Road_Geometry_Index",
]

CATEGORICAL_FEATURES = [
    "Road_Type",
    "Weather",
    "Lighting",
    "Junction",
    "Junction_Control",
    "Vehicle_Type",
    "Driver_Age",
    "Urban_Rural",
    "State",
    "City",
]

# Fallback global mean (will be overridden by metadata)
GLOBAL_MEAN_TARGET = 0.1544220276614643

# CRITICAL: This threshold is for CALIBRATED probabilities.
# The isotonic calibrator compresses raw ~0.60 down to ~0.21.
# A 0.35 threshold on calibrated output would never predict fatal.
# Derived from test set: threshold that produces F1=0.2334, Recall=0.4105.
OPTIMAL_THRESHOLD_CALIBRATED = 0.18


class MLService:
    _instance: Optional["MLService"] = None
    _initialized: bool = False

    def __new__(cls) -> "MLService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if MLService._initialized:
            return
        self._load_artifacts()
        MLService._initialized = True

    def _load_artifacts(self) -> None:
        t0 = time.time()

        if not META_PATH.exists():
            raise RuntimeError(f"feature_metadata.json not found at {META_PATH}")
        with open(META_PATH, "r") as f:
            self.metadata: Dict = json.load(f)

        self.feature_names = self.metadata.get("feature_names", FEATURE_ORDER)
        self.n_features = self.metadata.get("feature_count", len(FEATURE_ORDER))
        self.encoders = self.metadata.get("categorical_encoders", {})
        self.target_encodings = self.metadata.get("target_encodings", {})

        te_state = self.target_encodings.get("State_Risk_Score", {})
        te_weather = self.target_encodings.get("Weather_Risk", {})
        self.global_mean = float(
            te_state.get("global_mean", te_weather.get("global_mean", GLOBAL_MEAN_TARGET))
        )
        self.state_risk_map = te_state.get("mapping", {})
        self.weather_risk_map = te_weather.get("mapping", {})

        # Load calibrated threshold from metadata or use default
        self.optimal_threshold = float(
            self.metadata.get("inference_settings", {}).get(
                "optimal_threshold_calibrated", OPTIMAL_THRESHOLD_CALIBRATED
            )
        )

        self.shap_base_value = float(self.metadata.get("base_value", 0.5308434221730036))
        self.shap_base_log_odds = float(self.metadata.get("base_value_log_odds", 0.1235))

        if not ONNX_PATH.exists():
            raise RuntimeError(f"ONNX model not found at {ONNX_PATH}")
        self.onnx_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        onnx_shape = self.onnx_session.get_inputs()[0].shape
        if onnx_shape[1] != self.n_features:
            raise ValueError(
                f"ONNX expects {onnx_shape[1]} features, metadata says {self.n_features}"
            )

        if not CALIBRATOR_PATH.exists():
            raise RuntimeError(f"Calibrator not found at {CALIBRATOR_PATH}")
        with open(CALIBRATOR_PATH, "rb") as f:
            self.calibrator = pickle.load(f)

        if not XGB_PATH.exists():
            raise RuntimeError(f"XGBoost model not found at {XGB_PATH}")
        self.xgb_model = joblib.load(XGB_PATH)
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)

        load_ms = round((time.time() - t0) * 1000, 2)
        print(
            f"[MLService] Loaded in {load_ms}ms | threshold={self.optimal_threshold} | global_mean={self.global_mean:.4f}"
        )

    def _encode_categorical(self, feature_name: str, value: str) -> int:
        classes = self.encoders.get(feature_name, [])
        try:
            return classes.index(str(value))
        except ValueError:
            # Asynchronously log the unknown value as a drift event
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(log_drift_event(feature_name, str(value)))
            except RuntimeError:
                pass  # Fallback if no async loop running (e.g. unit tests)
            return -1

    def _target_encode(self, feature_name: str, raw_value: str) -> float:
        if feature_name == "State_Risk_Score":
            return float(self.state_risk_map.get(str(raw_value), self.global_mean))
        if feature_name == "Weather_Risk":
            return float(self.weather_risk_map.get(str(raw_value), self.global_mean))
        return self.global_mean

    def _build_feature_vector(self, data: Dict) -> np.ndarray:
        encoded = {}
        for feat in CATEGORICAL_FEATURES:
            raw_value = data.get(feat, "")
            encoded[feat] = self._encode_categorical(feat, str(raw_value))

        state_raw = data.get("State", "")
        weather_raw = data.get("Weather", "")
        state_risk = self._target_encode("State_Risk_Score", state_raw)
        weather_risk = self._target_encode("Weather_Risk", weather_raw)

        road_geom_idx = (
            encoded["Road_Type"] * 0.3
            + encoded["Junction"] * 0.2
            + encoded["Junction_Control"] * 0.2
            + encoded["Lighting"] * 0.3
        )

        vector = [
            encoded["Road_Type"],
            float(data.get("Speed_Limit", 50)),
            encoded["Weather"],
            encoded["Lighting"],
            encoded["Junction"],
            encoded["Junction_Control"],
            encoded["Vehicle_Type"],
            encoded["Driver_Age"],
            encoded["Urban_Rural"],
            encoded["State"],
            encoded["City"],
            state_risk,
            weather_risk,
            road_geom_idx,
        ]

        arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != self.n_features:
            raise ValueError(
                f"Feature vector has {arr.shape[1]} elements, model expects {self.n_features}"
            )
        return arr

    def predict(self, data: Dict) -> Dict:
        t0 = time.time()

        features = self._build_feature_vector(data)
        onnx_outs = self.onnx_session.run(None, {self.onnx_input_name: features})

        if len(onnx_outs) > 1:
            prob_tensor = onnx_outs[1]
        else:
            prob_tensor = onnx_outs[0]

        if prob_tensor.ndim == 2 and prob_tensor.shape[1] == 2:
            raw_prob = float(prob_tensor[0][1])
        elif prob_tensor.ndim == 2 and prob_tensor.shape[1] == 1:
            raw_prob = float(prob_tensor[0][0])
        else:
            raw_prob = float(prob_tensor.flatten()[0])

        # Isotonic calibration: raw ~0.60 → calibrated ~0.21
        calibrated_prob = float(self.calibrator.predict(np.array([raw_prob]))[0])
        calibrated_prob = max(0.0, min(1.0, calibrated_prob))

        # CRITICAL: Use calibrated threshold (0.18), not raw threshold (0.35)
        predicted_class = 1 if calibrated_prob >= self.optimal_threshold else 0

        # SHAP explainability
        shap_values = self.shap_explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = shap_values.flatten()

        contributions = []
        for i, name in enumerate(self.feature_names):
            contrib = float(shap_values[i])
            contrib_prob = float(1 / (1 + np.exp(-(self.shap_base_log_odds + contrib))))
            contributions.append(
                {
                    "feature": name,
                    "value": float(features[0][i]),
                    "contribution": round(contrib, 4),
                    "contribution_prob": round(contrib_prob, 4),
                }
            )

        top_features = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)[:3]

        # Confidence: distance from calibrated threshold (not 0.5)
        distance = abs(calibrated_prob - self.optimal_threshold)
        if distance > 0.15:
            confidence = "high"
        elif distance > 0.05:
            confidence = "medium"
        else:
            confidence = "low"

        inference_ms = round((time.time() - t0) * 1000, 2)

        return {
            "predicted_class": predicted_class,
            "probability": round(calibrated_prob, 4),
            "raw_probability": round(raw_prob, 4),
            "inference_ms": inference_ms,
            "confidence": confidence,
            "shap_base_value": round(self.shap_base_value, 4),
            "top_features": top_features,
            "all_contributions": contributions,
            "feature_vector": features.flatten().tolist(),
            "threshold_used": self.optimal_threshold,
        }

    def health_check(self) -> Dict:
        return {
            "status": "healthy",
            "onnx_loaded": ONNX_PATH.exists(),
            "calibrator_loaded": CALIBRATOR_PATH.exists(),
            "xgb_loaded": XGB_PATH.exists(),
            "metadata_loaded": META_PATH.exists(),
            "n_features": self.n_features,
            "global_mean": self.global_mean,
            "optimal_threshold": self.optimal_threshold,
            "shap_base_value": self.shap_base_value,
        }


def get_ml_service() -> MLService:
    return MLService()
