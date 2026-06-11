import json

with open(r"C:\Users\Hemanth\Downloads\roadrisk_phase4_fixed.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            # Apply our strict naming rules
            line = line.replace("ROADRISK", "SOOCHAK")
            line = line.replace("roadrisk_v1", "soochak_v1")
            line = line.replace("roadrisk", "soochak")
            new_source.append(line)
        cell["source"] = new_source

with open(r"C:\Users\Hemanth\SOOCHAK\notebooks\soochak_ml_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Downloaded notebook successfully migrated, corrected, and saved to SOOCHAK project!")
