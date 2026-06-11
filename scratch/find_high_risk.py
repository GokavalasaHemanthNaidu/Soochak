from src.services.ml_service import MLService

ml = MLService()

# Let's test a few combinations
payloads = [
    {
        "Road_Type": "Undivided Two way",
        "Speed_Limit": 80,
        "Weather": "Normal",
        "Lighting": "Darkness - no lighting",
        "Junction": "T Shape",
        "Junction_Control": "No distancing",
        "Vehicle_Type": "Motorcycle",
        "Driver_Age": "Under 18",
        "Urban_Rural": "Rural village areas",
        "State": "Steep grade upward with mountainous terrain",
        "City": "Saturday"
    },
    {
        "Road_Type": "Undivided Two way",
        "Speed_Limit": 100,
        "Weather": "Normal",
        "Lighting": "Darkness - no lighting",
        "Junction": "Crossing",
        "Junction_Control": "Overtaking",
        "Vehicle_Type": "Motorcycle",
        "Driver_Age": "18-30",
        "Urban_Rural": "Rural village areas",
        "State": "Steep grade upward with mountainous terrain",
        "City": "Saturday"
    },
    {
        "Road_Type": "Two-way (divided with solid lines road marking)",
        "Speed_Limit": 120,
        "Weather": "Normal",
        "Lighting": "Darkness - no lighting",
        "Junction": "Crossing",
        "Junction_Control": "Overtaking",
        "Vehicle_Type": "Motorcycle",
        "Driver_Age": "18-30",
        "Urban_Rural": "Rural village areas",
        "State": "Steep grade upward with mountainous terrain",
        "City": "Saturday"
    }
]

for i, p in enumerate(payloads):
    res = ml.predict(p)
    print(f"Payload {i+1}: prob={res['probability']}, class={res['predicted_class']}, raw={res['raw_probability']}, threshold={res['threshold_used']}")
