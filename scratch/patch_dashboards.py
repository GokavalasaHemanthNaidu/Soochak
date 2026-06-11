import re
from pathlib import Path

blueprint_path = Path(r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md")

with open(blueprint_path, "r", encoding="utf-8") as f:
    content = f.read()

# Define the replacements mapping
replacements = {
    'name="road_type"': 'name="Road_Type"',
    'name="speed_limit"': 'name="Speed_Limit"',
    'name="weather"': 'name="Weather"',
    'name="lighting"': 'name="Lighting"',
    'name="junction"': 'name="Junction"',
    'name="junction_ctrl"': 'name="Junction_Control"',
    'name="vehicle_type"': 'name="Vehicle_Type"',
    'name="driver_age"': 'name="Driver_Age"',
    'name="urban_rural"': 'name="Urban_Rural"',
    'name="state"': 'name="State"',
    'name="city"': 'name="City"',
    
    'form.road_type.value': 'form.Road_Type.value',
    'form.speed_limit.value': 'form.Speed_Limit.value',
    'form.weather.value': 'form.Weather.value',
    'form.lighting.value': 'form.Lighting.value',
    'form.junction.value': 'form.Junction.value',
    'form.junction_ctrl.value': 'form.Junction_Control.value',
    'form.vehicle_type.value': 'form.Vehicle_Type.value',
    'form.driver_age.value': 'form.Driver_Age.value',
    'form.urban_rural.value': 'form.Urban_Rural.value',
    'form.state.value': 'form.State.value',
    'form.city.value': 'form.City.value',

    # Also update speed_limit conversion in Javascript
    'data.speed_limit = data.speed_limit ? parseInt(data.speed_limit) : null;':
    'data.Speed_Limit = data.Speed_Limit ? parseInt(data.Speed_Limit) : null;',

    'data.speed_limit = parseInt(data.speed_limit)':
    'data.Speed_Limit = parseInt(data.Speed_Limit)',

    'feature == \'speed_limit\'': 'feature == \'Speed_Limit\'',
    'feature_to_change: feature': 'feature_to_change: feature',
}

# Apply replacements
original_len = len(content)
for k, v in replacements.items():
    content = content.replace(k, v)

# Let's check some key javascript parts in the second dashboard
# E.g., const labels = topFeatures.map(f => f.feature);
# const values = topFeatures.map(f => f.shap_value);
# Since top_features has contribution instead of shap_value:
content = content.replace('f.shap_value', 'f.contribution')
content = content.replace('f.contribution_pct', 'f.contribution_prob') # we can show contribution_prob or similar

with open(blueprint_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Dashboard HTML files successfully patched! Length changed from {original_len} to {len(content)}")
