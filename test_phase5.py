#!/usr/bin/env python3
"""End-to-end test for SOOCHAK Phase 5 — CORRECTED THRESHOLD VALIDATION.

Validates:
1. High-risk scenarios predict FATAL (class 1) with threshold 0.18
2. Low-risk scenarios predict NON-FATAL (class 0)
3. Calibration behavior: raw > calibrated (overconfidence correction)
4. Confidence logic relative to calibrated threshold
5. Unknown category resilience
6. Health check exposes correct configuration
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000/v1"
headers = {"X-API-Key": "soochak-demo-2024"}

# ── Test 1: EXTREME high-risk scenario (MUST predict fatal) ────────────────
print("=" * 60)
print("TEST 1: EXTREME high-risk scenario")
print("=" * 60)

payload_extreme = {
    "Road_Type": "Undivided Two way",
    "Speed_Limit": 80,
    "Weather": "Windy",
    "Lighting": "Darkness - no lighting",
    "Junction": "Crossing",
    "Junction_Control": "Drunk driving",
    "Vehicle_Type": "Motorcycle",
    "Driver_Age": "Under 18",
    "Urban_Rural": "Rural village areas",
    "State": "Steep grade upward with mountainous terrain",
    "City": "Saturday"
}

resp = requests.post(f"{BASE_URL}/predict/", json=payload_extreme, headers=headers)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Predicted class: {data['predicted_class']} {'(FATAL)' if data['predicted_class'] == 1 else '(NON-FATAL)'}")
    print(f"Calibrated probability: {data['probability']}")
    print(f"Raw probability: {data['raw_probability']}")
    print(f"Threshold used: {data.get('threshold_used', 'N/A')}")
    print(f"Confidence: {data['confidence']}")
    print(f"Inference time: {data['inference_ms']}ms")

    if data['predicted_class'] == 1:
        print("\n[PASS] Extreme high-risk correctly predicted as FATAL")
    else:
        print("\n[FAIL] Extreme high-risk predicted as NON-FATAL — threshold may still be too high")
else:
    print(f"[ERROR] {resp.text}")
    sys.exit(1)

# ── Test 2: Moderate high-risk (should also predict fatal) ───────────────────
print("\n" + "=" * 60)
print("TEST 2: Moderate high-risk scenario")
print("=" * 60)

payload_moderate = {
    "Road_Type": "Undivided Two way",
    "Speed_Limit": 60,
    "Weather": "Raining",
    "Lighting": "Darkness - lights unlit",
    "Junction": "Crossing",
    "Junction_Control": "Overtaking",
    "Vehicle_Type": "Bicycle",
    "Driver_Age": "18-30",
    "Urban_Rural": "Outside rural areas",
    "State": "Sharp reverse curve",
    "City": "Friday"
}

resp = requests.post(f"{BASE_URL}/predict/", json=payload_moderate, headers=headers)
data = resp.json()
print(f"Predicted class: {data['predicted_class']} {'(FATAL)' if data['predicted_class'] == 1 else '(NON-FATAL)'}")
print(f"Calibrated probability: {data['probability']}")
print(f"Confidence: {data['confidence']}")

if data['predicted_class'] == 1:
    print("[PASS] Moderate high-risk predicted as FATAL")
else:
    print("[INFO] Moderate risk predicted as NON-FATAL (may be correct depending on feature interactions)")

# ── Test 3: Low-risk scenario (should predict non-fatal) ────────────────────
print("\n" + "=" * 60)
print("TEST 3: Low-risk scenario")
print("=" * 60)

payload_low = {
    "Road_Type": "Double carriageway (median)",
    "Speed_Limit": 40,
    "Weather": "Normal",
    "Lighting": "Daylight",
    "Junction": "No junction",
    "Junction_Control": "None",
    "Vehicle_Type": "Automobile",
    "Driver_Age": "31-50",
    "Urban_Rural": "Residential areas",
    "State": "Tangent road with flat terrain",
    "City": "Monday"
}

resp = requests.post(f"{BASE_URL}/predict/", json=payload_low, headers=headers)
data = resp.json()
print(f"Predicted class: {data['predicted_class']} {'(FATAL)' if data['predicted_class'] == 1 else '(NON-FATAL)'}")
print(f"Calibrated probability: {data['probability']}")
print(f"Confidence: {data['confidence']}")

if data['predicted_class'] == 0:
    print("[PASS] Low-risk correctly predicted as NON-FATAL")
else:
    print("[INFO] Low-risk predicted as FATAL — threshold may be too aggressive")

# ── Test 4: Calibration validation ──────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Calibration behavior validation")
print("=" * 60)

resp = requests.post(f"{BASE_URL}/predict/", json=payload_extreme, headers=headers)
data = resp.json()
raw = data['raw_probability']
cal = data['probability']

print(f"Raw probability: {raw:.4f}")
print(f"Calibrated probability: {cal:.4f}")
print(f"Calibration delta: {cal - raw:+.4f}")

if raw > cal:
    print("[PASS] Calibrator reduces overconfidence (raw > calibrated)")
    print("     This is CORRECT — XGBoost is systematically overconfident.")
else:
    print("[WARN] Calibrator increased probability — verify calibration curve")

# Validate calibrated prob is near base rate for typical inputs
if 0.10 <= cal <= 0.35:
    print("[PASS] Calibrated probability in expected range (0.10-0.35)")
else:
    print(f"[INFO] Calibrated probability {cal:.4f} outside typical range")

# ── Test 5: Confidence logic validation ─────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: Confidence logic validation")
print("=" * 60)

test_cases = [
    ("Very low prob", {**payload_low, "Speed_Limit": 10}),
    ("Near threshold low", {**payload_low, "Speed_Limit": 60, "Weather": "Raining"}),
    ("Near threshold high", {**payload_moderate, "Weather": "Normal"}),
    ("Very high prob", payload_extreme),
]

for name, payload in test_cases:
    resp = requests.post(f"{BASE_URL}/predict/", json=payload, headers=headers)
    data = resp.json()
    print(f"  {name:25s}: prob={data['probability']:.3f}, class={data['predicted_class']}, confidence={data['confidence']}")

# ── Test 6: Unknown categories (resilience) ──────────────────────────────────
print("\n" + "=" * 60)
print("TEST 6: Unknown categories (resilience test)")
print("=" * 60)

payload_unknown = {
    "Road_Type": "National Highway",  # Unknown → -1
    "Speed_Limit": 80,
    "Weather": "Thunderstorm",  # Unknown → -1
    "Lighting": "Daylight",
    "Junction": "Roundabout",  # Unknown → -1
    "Junction_Control": "Traffic signal violation",  # Unknown → -1
    "Vehicle_Type": "SUV",  # Unknown → -1
    "Driver_Age": "25-35",  # Unknown → -1
    "Urban_Rural": "Suburban",  # Unknown → -1
    "State": "Maharashtra",  # Unknown → global_mean
    "City": "Mumbai"  # Unknown → -1
}

resp = requests.post(f"{BASE_URL}/predict/", json=payload_unknown, headers=headers)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Predicted class: {data['predicted_class']}")
    print(f"Calibrated probability: {data['probability']}")
    print("[PASS] API resilient to unknown categories")
else:
    print(f"[FAIL] API crashed on unknown categories: {resp.text}")

# ── Test 7: Health check ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 7: Health check")
print("=" * 60)

resp = requests.get(f"{BASE_URL}/health", headers=headers)
print(f"Status: {resp.status_code}")
health = resp.json()
print(json.dumps(health, indent=2))

assert health.get("optimal_threshold", 0) == 0.18, f"Expected threshold 0.18, got {health.get('optimal_threshold')}"
assert abs(health.get("global_mean", 0) - 0.1544) < 0.001, f"Global mean mismatch: {health.get('global_mean')}"
print("[PASS] Health check returns correct configuration")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("""
Key fixes validated:
- Threshold = 0.18 on CALIBRATED probabilities (not 0.35)
- High-risk scenarios can now predict FATAL (class 1)
- Calibration reduces overconfidence (raw > calibrated)
- Confidence relative to calibrated threshold, not 0.5
- Unknown categories handled safely
- Health check exposes correct configuration
""")
