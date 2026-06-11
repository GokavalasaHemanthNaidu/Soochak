"""Feature encoding utilities for drift detection."""

import json
from pathlib import Path
import numpy as np

ML_ARTIFACTS = Path(__file__).parent.parent.parent / "ml_artifacts"
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


def _load_metadata():
    with open(META_PATH, "r") as f:
        return json.load(f)


def encode_features(data: dict) -> np.ndarray:
    """Encode raw incident dict to feature vector (same logic as MLService)."""
    metadata = _load_metadata()
    encoders = metadata.get("categorical_encoders", {})
    global_mean = 0.1544

    encoded = {}
    for feat in [
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
    ]:
        classes = encoders.get(feat, [])
        val = str(data.get(feat, ""))
        try:
            encoded[feat] = classes.index(val)
        except ValueError:
            encoded[feat] = -1

    state_risk = global_mean
    weather_risk = global_mean
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
    return np.array(vector, dtype=np.float32)


def get_feature_dict(vec: np.ndarray) -> dict:
    """Convert feature vector back to dict for drift detection."""
    return {name: float(vec[i]) for i, name in enumerate(FEATURE_ORDER)}
