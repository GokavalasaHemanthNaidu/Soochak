import re

file_path = r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Total characters: {len(content)}")
lines = content.splitlines()
print(f"Total lines: {len(lines)}")

# Let's search for some patterns
patterns = [
    "PredictResponse",
    "PredictRequest",
    "Literal",
    "_session",
    "optimal_threshold",
    "drift_events",
    "drift_logger",
    "global_mean",
    "HealthResponse",
    "Calibration",
    "F1-optimized",
    "Phase 4",
    "Phase 5",
]

for p in patterns:
    matches = [i+1 for i, line in enumerate(lines) if p.lower() in line.lower()]
    print(f"Pattern '{p}': found on {len(matches)} lines. First few: {matches[:10]}")
