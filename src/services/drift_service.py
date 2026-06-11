"""KS-test drift detection comparing live vs training distribution."""

import pickle
from pathlib import Path
from scipy import stats
from datetime import datetime
from typing import List, Dict

DISTRIBUTION_PATH = Path(__file__).parent.parent.parent / "ml_artifacts" / "training_distribution_encoded.pkl"

with open(DISTRIBUTION_PATH, "rb") as f:
    TRAINING_DIST = pickle.load(f)


def detect_drift(live_features: List[dict]) -> Dict:
    """Run KS-test for each feature in encoded numeric space."""
    results = []
    overall_drift = False

    for feature_name, train_dist in TRAINING_DIST.items():
        if feature_name not in live_features[0]:
            continue

        live_values = [f.get(feature_name, 0) for f in live_features]
        train_values = train_dist.get("values", [])

        if not train_values or not live_values:
            continue

        ks_stat, p_value = stats.ks_2samp(train_values, live_values)
        drift_detected = p_value < 0.05

        if drift_detected:
            overall_drift = True

        results.append({
            "feature": feature_name,
            "p_value": round(p_value, 4),
            "drifted": drift_detected,
        })

    return {
        "overall_drift": overall_drift,
        "features": results,
        "checked_at": datetime.now().isoformat(),
    }
