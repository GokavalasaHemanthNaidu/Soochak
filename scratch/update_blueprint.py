import re

blueprint_path = r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md"

with open(blueprint_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Standard Case Replacements
content = content.replace("roadrisk.db", "soochak.db")
content = content.replace("roadrisk-demo-2024", "soochak-demo-2024")
content = content.replace("roadrisk.onrender.com", "soochak.onrender.com")
content = content.replace("roadrisk_v1", "soochak_v1")
content = content.replace("ROADRISK", "SOOCHAK")
content = content.replace("RoadRisk", "Soochak")
# Match roadrisk in lowercase but ignore when it is in a URL or directory that is already replaced
content = re.sub(r'\broadrisk\b', 'soochak', content)

# 2. Threshold Replacements
# Replace 0.35 with 0.18 for optimal calibrated threshold
content = content.replace("optimal decision threshold of 0.35", "optimal decision threshold of 0.18")
content = content.replace("optimal decision threshold of `0.35`", "optimal decision threshold of `0.18`")
content = content.replace("calibrated_prob >= 0.5", "calibrated_prob >= 0.18")
content = content.replace("predicted_class = 1 if calibrated_prob >= 0.5 else 0", "predicted_class = 1 if calibrated_prob >= 0.18 else 0")

# 3. Test Scenarios (Step 5.9 and Step 12.5)
# Let's locate the payload in Step 5.9 / 12.5 and replace its values with our new windy + crossing + drunk driving + steep grade mountainous terrain payload
old_test_payload = """    payload = {
        "road_type": "National Highway",
        "speed_limit": 80,
        "weather": "Clear",
        "lighting": "Darkness - no lighting",
        "junction": "None",
        "junction_ctrl": "Uncontrolled",
        "vehicle_type": "Two-wheeler",
        "driver_age": "Young (17-24)",
        "urban_rural": "Rural",
        "state": "Maharashtra",
        "city": "Pune"
    }"""

new_test_payload = """    payload = {
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
    }"""

content = content.replace(old_test_payload, new_test_payload)

# Replace the payload in Step 12.5 E2E manual test curl command
old_curl = '{"road_type":"National Highway","speed_limit":80,"weather":"Clear","lighting":"Darkness - no lighting","junction":"None","junction_ctrl":"Uncontrolled","vehicle_type":"Two-wheeler","driver_age":"Young (17-24)","urban_rural":"Rural","state":"Maharashtra","city":"Pune"}'
new_curl = '{"Road_Type":"Undivided Two way","Speed_Limit":80,"Weather":"Windy","Lighting":"Darkness - no lighting","Junction":"Crossing","Junction_Control":"Drunk driving","Vehicle_Type":"Motorcycle","Driver_Age":"Under 18","Urban_Rural":"Rural village areas","State":"Steep grade upward with mountainous terrain","City":"Saturday"}'
content = content.replace(old_curl, new_curl)

# 4. Target Encoding Pitfalls Table in the blueprint
# Let's check if there is any target encoding reference. If so, let's update it.
# Let's write the updated contents back
with open(blueprint_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Blueprint successfully updated and rewritten!")
