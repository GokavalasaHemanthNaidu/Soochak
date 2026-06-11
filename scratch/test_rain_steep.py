from src.services.ml_service import MLService

ml = MLService()

payload = {
    "Road_Type": "Undivided Two way",
    "Speed_Limit": 80,
    "Weather": "Raining", # Target encoded to 0.1288
    "Lighting": "Darkness - no lighting",
    "Junction": "Crossing",
    "Junction_Control": "Drunk driving",
    "Vehicle_Type": "Motorcycle",
    "Driver_Age": "Under 18",
    "Urban_Rural": "Rural village areas",
    "State": "Steep grade upward with mountainous terrain", # 0.375 (highest)
    "City": "Saturday"
}

res = ml.predict(payload)
print(f"Raining / Steep Up: prob={res['probability']:.4f}, raw={res['raw_probability']:.4f}, class={res['predicted_class']}")
