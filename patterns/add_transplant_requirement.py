#!/usr/bin/env python3
"""
Add transplant_rate requirement to transplant_focused archetype.
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

# Add violation condition for low transplant rate
violation_conditions = archetypes[tf_index]["icm"]["violation_conditions"]
# Check if already exists
has_transplant_violation = any(
    vc.get("feature") == "intervention.transplant_rate" 
    for vc in violation_conditions
)

if not has_transplant_violation:
    new_violation = {
        "feature": "intervention.transplant_rate",
        "predicate": "lt",
        "value": 0.1,
        "tolerance": 0.0,
        "rationale": "Transplant-focused protocol requires transplant_rate > 0.1"
    }
    violation_conditions.append(new_violation)
    print("Added transplant_rate violation condition")
else:
    print("Transplant_rate violation condition already exists")

# Also add a linking feature criterion (optional)
grounding_criteria = archetypes[tf_index]["grounding_criteria"]
has_transplant_criterion = any(
    gc.get("feature") == "intervention.transplant_rate"
    for gc in grounding_criteria
)

if not has_transplant_criterion:
    new_criterion = {
        "feature": "intervention.transplant_rate",
        "predicate": "gt",
        "value": 0.1,
        "tolerance": 0.0,
        "rationale": "Transplant-focused protocol includes transplant intervention",
        "layer": "linking",  # This is an intervention parameter, not biological outcome
        "original_value": 0.1,
        "adjustment_note": "new linking criterion",
        "meet_percentage": 50.0  # placeholder
    }
    grounding_criteria.append(new_criterion)
    print("Added transplant_rate linking criterion")
else:
    print("Transplant_rate criterion already exists")

# Save updated archetypes
with open(archetypes_path, 'w') as f:
    json.dump(archetypes, f, indent=2)

print(f"Updated archetypes saved to {archetypes_path}")

# Quick test: load and verify
with open(archetypes_path, 'r') as f:
    updated = json.load(f)
tf = updated[tf_index]
print(f"\nUpdated transplant_focused archetype:")
print(f"  Grounding criteria: {len(tf['grounding_criteria'])}")
print(f"  Violation conditions: {len(tf['icm']['violation_conditions'])}")
print("  Violation conditions features:", [vc['feature'] for vc in tf['icm']['violation_conditions']])