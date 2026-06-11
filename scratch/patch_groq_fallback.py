from pathlib import Path

bp_path = Path(r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md")
with open(bp_path, "r", encoding="utf-8") as f:
    content = f.read()

# Target lines in groq_service.py fallback
target = 'top = top_features[0]["feature"] if top_features else "unknown"'
replacement = 'top = top_features[0]["feature"].lower() if top_features else "unknown"'

if target in content:
    content = content.replace(target, replacement)
    with open(bp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("groq_service fallback successfully updated in blueprint!")
else:
    # Let's search with regex to see if it's slightly different
    import re
    pattern = r'top\s*=\s*top_features\[0\]\["feature"\]\s*if\s*top_features\s*else\s*"unknown"'
    match = re.search(pattern, content)
    if match:
        content = re.sub(pattern, replacement, content)
        with open(bp_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("groq_service fallback successfully updated via regex in blueprint!")
    else:
        print("Target not found in blueprint!")
