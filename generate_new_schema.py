#!/usr/bin/env python3
"""
Generate updated BIN_SCHEMA with adjusted thresholds and centers.
"""
import json
from pathlib import Path

# Load edge validation stats
edge_dir = Path("artifacts/ca_ode_validation_fixed_edge_20260222_213009")
summary_path = edge_dir / "validation_summary.json"
with open(summary_path, 'r') as f:
    data = json.load(f)
global_stats = data["global_bin_stats"]

# Original schema from ca_schema.py (copied)
original = {
    "N_healthy": {
        "index": 0,
        "thresholds": [0.3, 0.56],
        "labels": ["depleted", "reduced", "adequate"],
        "centers": [0.15, 0.5, 0.85],
        "unit": "normalized copies",
        "source": "C2 copy homeostasis",
    },
    "N_deletion": {
        "index": 1,
        "thresholds": [0.1, 0.3, 0.5],
        "labels": ["minimal", "growing", "approaching_cliff", "past_cliff"],
        "centers": [0.05, 0.2, 0.4, 0.7],
        "unit": "deletion het fraction",
        "source": "HETEROPLASMY_CLIFF=0.50, Cramer Appendix 2",
    },
    "ATP": {
        "index": 2,
        "thresholds": [0.2, 0.5, 0.79],
        "labels": ["collapsed", "crisis", "compromised", "healthy"],
        "centers": [0.1, 0.35, 0.65, 0.9],
        "unit": "MU/day",
        "source": "ATP_CRISIS_FRACTION=0.5, Cramer Ch. VIII.A Table 3",
    },
    "ROS": {
        "index": 3,
        "thresholds": [0.1, 0.25],
        "labels": ["basal", "elevated", "pathological"],
        "centers": [0.05, 0.175, 0.4],
        "unit": "normalized",
        "source": "BASELINE_ROS=0.1, Cramer Ch. II.H",
    },
    "NAD": {
        "index": 4,
        "thresholds": [0.3, 0.7],
        "labels": ["depleted", "declining", "robust"],
        "centers": [0.15, 0.5, 0.85],
        "unit": "normalized",
        "source": "NAD_DECLINE_RATE=0.01/yr, Cramer Ch. VI.A.3",
    },
    "Senescent_fraction": {
        "index": 5,
        "thresholds": [0.1, 0.4],
        "labels": ["minimal", "emerging", "severe"],
        "centers": [0.05, 0.25, 0.6],
        "unit": "fraction",
        "source": "SENESCENCE_RATE=0.005/yr, Cramer Ch. VII.A",
    },
    "Membrane_potential": {
        "index": 6,
        "thresholds": [0.3, 0.7],
        "labels": ["collapsed", "impaired", "intact"],
        "centers": [0.15, 0.5, 0.85],
        "unit": "normalized ΔΨ",
        "source": "MITOPHAGY_ATP_MIDPOINT=0.6, Cramer Ch. VI.B",
    },
    "N_point": {
        "index": 7,
        "thresholds": [0.1, 0.3],
        "labels": ["low", "moderate", "high"],
        "centers": [0.05, 0.2, 0.5],
        "unit": "point het fraction",
        "source": "POINT_ERROR_RATE=0.001, Cramer Ch. II.H",
    },
}

# Proposed changes
# 1. Adjust Membrane_potential thresholds to [0.3, 0.5] (since intact >0.7 rarely reached)
# 2. Update centers to empirical means where count >=10
# 3. Keep thresholds otherwise unchanged

updated = original.copy()

# Update Membrane_potential thresholds
updated["Membrane_potential"]["thresholds"] = [0.3, 0.5]
# Recompute centers as midpoints (optional)
th = updated["Membrane_potential"]["thresholds"]
centers = [th[0]/2, (th[0]+th[1])/2, (th[1]+1.0)/2]
updated["Membrane_potential"]["centers"] = [round(c, 3) for c in centers]

# Update centers based on empirical means
for var_name, schema in updated.items():
    labels = schema["labels"]
    new_centers = []
    for label in labels:
        stats = global_stats.get(var_name, {}).get(label)
        if stats and stats['count'] >= 10:
            new_centers.append(round(stats['mean'], 3))
        else:
            # Keep original center for this label
            idx = labels.index(label)
            new_centers.append(schema["centers"][idx])
    schema["centers"] = new_centers

# Print updated schema in Python format
print("BIN_SCHEMA: dict[str, dict] = {")
for var_name, schema in updated.items():
    print(f'    "{var_name}": {{')
    for key, val in schema.items():
        if isinstance(val, list):
            formatted = "[" + ", ".join(str(v) for v in val) + "]"
        else:
            formatted = f'"{val}"' if isinstance(val, str) else str(val)
        print(f'        "{key}": {formatted},')
    print("    },")
print("}")

# Also output as JSON for verification
with open(edge_dir / "updated_bin_schema.json", 'w') as f:
    json.dump(updated, f, indent=2)
print(f"\nUpdated schema saved to {edge_dir}/updated_bin_schema.json")