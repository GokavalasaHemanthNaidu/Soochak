import re

file_path = r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

lines = content.splitlines()

# Search for any occurrences of "roadrisk" case-insensitive
roadrisk_matches = [i+1 for i, line in enumerate(lines) if "roadrisk" in line.lower()]
print(f"Occurrences of 'roadrisk' (case-insensitive): {len(roadrisk_matches)} lines: {roadrisk_matches[:30]}")

# Search for any strict Literal annotations
literal_matches = [i+1 for i, line in enumerate(lines) if "Literal[" in line]
print(f"Occurrences of 'Literal[': {len(literal_matches)} lines: {literal_matches}")

# Search for _session
session_matches = [i+1 for i, line in enumerate(lines) if "_session" in line]
print(f"Occurrences of '_session': {len(session_matches)} lines: {session_matches}")

# Search for any code blocks writing to src/api/routers/predict.py or similar
write_matches = []
for i, line in enumerate(lines):
    if "cat > " in line or "cat >> " in line:
        write_matches.append((i+1, line))
print(f"Occurrences of cat > or cat >>: {len(write_matches)}")
for num, line in write_matches:
    print(f"Line {num}: {line}")
