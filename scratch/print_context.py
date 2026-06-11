file_path = r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

def print_around(line_num, radius=10):
    start = max(0, line_num - radius - 1)
    end = min(len(lines), line_num + radius)
    print(f"\n--- Context around line {line_num} ---")
    for idx in range(start, end):
        print(f"{idx+1}: {lines[idx]}")

# Let's print around key areas
print_around(995, 15)  # Literal / PredictRequest
print_around(1127, 15)  # PredictResponse
print_around(1261, 15)  # HealthResponse
print_around(1395, 15)  # predict router / request.py / response.py
print_around(2612, 25)  # _session
print_around(2788, 25)  # drift_logger / drift_events
print_around(2300, 25)  # Calibration / F1
