#!/usr/bin/env python3
"""
Update transplant_focused archetype with stricter criteria to avoid matching
young patients without transplant intervention.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Load adjusted archetypes
archetypes_path = ROOT / "lakoff_archetypes_adjusted.json"
with open(archetypes_path, 'r') as f:
    archetypes = json.load(f)

# Find transplant_focused archetype
tf_index = -1
for i, arch in enumerate(archetypes):
    if arch["name"] == "transplant_focused":
        tf_index = i
        break

if tf_index == -1:
    print("Error: transplant_focused archetype not found")
    exit(1)

print("Current transplant_focused archetype:")
print(json.dumps(archetypes[tf_index], indent=2))

# Update grounding criteria
# Current criteria:
# 1. damage.deletion_het_final < 0.0477
# 2. energy.atp_final > 0.764  
# 3. damage.delta_het < -0.2517

# New criteria:
# 1. damage.deletion_het_final < 0.02 (stricter)
# 2. energy.atp_final > 0.8 (slightly higher)
# 3. intervention.het_benefit_terminal > 0.1 (significant reduction)
# 4. damage.deletion_het_initial > 0.15 (moderate deletion load)

new_criteria = [
    {
        "feature": "damage.deletion_het_final",
        "predicate": "lt",
        "value": 0.02,
        "tolerance": 0.0,
        "rationale": "Transplant reduces deletion heteroplasmy below 0.02 (very strict)",
        "layer": "grounded",
        "original_value": 0.0477,
        "adjustment_note": "tightened further (was 0.0477)",
        "meet_percentage": 100.0  # placeholder
    },
    {
        "feature": "energy.atp_final",
        "predicate": "gt",
        "value": 0.8,
        "tolerance": 0.0,
        "rationale": "Transplant achieves ATP above 0.8 MU",
        "layer": "grounded",
        "original_value": 0.764,
        "adjustment_note": "increased (was 0.764)",
        "meet_percentage": 44.2  # keep same
    },
    {
        "feature": "intervention.het_benefit_terminal",
        "predicate": "gt",
        "value": 0.1,
        "tolerance": 0.0,
        "rationale": "Transplant provides significant heteroplasmy reduction (>0.1)",
        "layer": "grounded",
        "original_value": 0.1,
        "adjustment_note": "new criterion",
        "meet_percentage": 50.0  # placeholder
    },
    {
        "feature": "damage.deletion_het_initial",
        "predicate": "gt",
        "value": 0.15,
        "tolerance": 0.0,
        "rationale": "Transplant indicated for moderate deletion load (>0.15)",
        "layer": "grounded",
        "original_value": 0.15,
        "adjustment_note": "new criterion",
        "meet_percentage": 50.0  # placeholder
    }
]

# Update ICM violation conditions
# Current:
# 1. dynamics.nad_slope < -0.01
# 2. damage.deletion_het_initial < 0.1

# Add:
# 3. energy.atp_initial < 0.6 (insufficient energy for engraftment)

current_icm = archetypes[tf_index]["icm"]
new_violation_conditions = [
    {
        "feature": "dynamics.nad_slope",
        "predicate": "lt",
        "value": -0.01,
        "tolerance": 0.0,
        "rationale": "Declining NAD impairs transplant engraftment"
    },
    {
        "feature": "damage.deletion_het_initial",
        "predicate": "lt",
        "value": 0.15,  # increased from 0.1
        "tolerance": 0.0,
        "rationale": "Transplant not indicated for low deletion heteroplasmy (<0.15)"
    },
    {
        "feature": "energy.atp_initial",
        "predicate": "lt",
        "value": 0.6,
        "tolerance": 0.0,
        "rationale": "Insufficient energy reserves for transplant engraftment"
    }
]

# Update archetype
archetypes[tf_index]["grounding_criteria"] = new_criteria
archetypes[tf_index]["icm"]["violation_conditions"] = new_violation_conditions

# Also update description to reflect patient population
archetypes[tf_index]["description"] = "Protocol centered on mtDNA transplant as primary rejuvenation for moderately damaged older patients"

# Update ICM background
archetypes[tf_index]["icm"]["background"] = [
    "Deletion heteroplasmy >0.15 indicates need for transplant",
    "Transplant competes with damaged mtDNA via displacement (Cramer C8)",
    "Patient has sufficient NAD+ and ATP reserves to support engraftment",
    "Transplant is the primary rejuvenation modality for damage reversal",
    "Patient is typically >50 years old with moderate mitochondrial damage"
]

print("\nUpdated transplant_focused archetype:")
print(json.dumps(archetypes[tf_index], indent=2))

# Save updated archetypes
output_path = ROOT / "lakoff_archetypes_adjusted_v2.json"
with open(output_path, 'w') as f:
    json.dump(archetypes, f, indent=2)

print(f"\nSaved updated archetypes to {output_path}")

# Also update the main file (optional)
update_main = input("\nUpdate main file (lakoff_archetypes_adjusted.json)? (y/n): ")
if update_main.lower() == 'y':
    with open(archetypes_path, 'w') as f:
        json.dump(archetypes, f, indent=2)
    print("Updated main file")
else:
    print("Main file not updated")