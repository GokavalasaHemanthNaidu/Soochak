from src.services.ml_service import MLService
import itertools

ml = MLService()

# Let's inspect encoders and target encodings to build a list of options
# State_Risk_Score:
# "Steep grade upward with mountainous terrain" has risk 0.375
# Weather_Risk:
# "Unknown" has risk 0.18032, "Windy" has risk 0.16393, "Normal" has risk 0.16037

# Let's write a search script that tries combinations of different features to maximize probability.
# Since CATEGORICAL_FEATURES contains:
# Road_Type, Weather, Lighting, Junction, Junction_Control, Vehicle_Type, Driver_Age, Urban_Rural, State, City

# Let's test the user's proposed high risk combination with correct weather values
payload_user_windy = {
    "Road_Type": "Undivided Two way",
    "Speed_Limit": 80,
    "Weather": "Windy", # Use Windy (0.1639)
    "Lighting": "Darkness - no lighting",
    "Junction": "Crossing",
    "Junction_Control": "Drunk driving",
    "Vehicle_Type": "Motorcycle",
    "Driver_Age": "Under 18",
    "Urban_Rural": "Rural village areas",
    "State": "Sharp reverse curve", # 0.128 risk score
    "City": "Saturday"
}

payload_user_steep_up = {
    "Road_Type": "Undivided Two way",
    "Speed_Limit": 80,
    "Weather": "Windy", 
    "Lighting": "Darkness - no lighting",
    "Junction": "Crossing",
    "Junction_Control": "Drunk driving",
    "Vehicle_Type": "Motorcycle",
    "Driver_Age": "Under 18",
    "Urban_Rural": "Rural village areas",
    "State": "Steep grade upward with mountainous terrain", # 0.375 risk score (highest State risk)
    "City": "Saturday"
}

payload_user_normal_weather = {
    "Road_Type": "Undivided Two way",
    "Speed_Limit": 80,
    "Weather": "Normal", # 0.1603 risk score
    "Lighting": "Darkness - no lighting",
    "Junction": "Crossing",
    "Junction_Control": "Drunk driving",
    "Vehicle_Type": "Motorcycle",
    "Driver_Age": "Under 18",
    "Urban_Rural": "Rural village areas",
    "State": "Steep grade upward with mountainous terrain",
    "City": "Saturday"
}

payload_user_unknown_weather = {
    "Road_Type": "Undivided Two way",
    "Speed_Limit": 80,
    "Weather": "Unknown", # 0.1803 risk score (highest Weather risk)
    "Lighting": "Darkness - no lighting",
    "Junction": "Crossing",
    "Junction_Control": "Drunk driving",
    "Vehicle_Type": "Motorcycle",
    "Driver_Age": "Under 18",
    "Urban_Rural": "Rural village areas",
    "State": "Steep grade upward with mountainous terrain",
    "City": "Saturday"
}

for name, p in [
    ("User Windy / Sharp Curve", payload_user_windy),
    ("User Windy / Steep Up", payload_user_steep_up),
    ("User Normal / Steep Up", payload_user_normal_weather),
    ("User Unknown / Steep Up", payload_user_unknown_weather),
]:
    res = ml.predict(p)
    print(f"{name:30s}: prob={res['probability']:.4f}, raw={res['raw_probability']:.4f}, class={res['predicted_class']}")
