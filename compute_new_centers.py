#!/usr/bin/env python3
"""
Compute new bin centers based on empirical ODE means from edge validation.
"""
import json
from pathlib import Path

edge_dir = Path("artifacts/ca_ode_validation_fixed_edge_20260222_213009")
summary_path = edge_dir / "validation_summary.json"
with open(summary_path, 'r') as f:
    data = json.load(f)

global_stats = data["global_bin_stats"]

# Current schema from ca_schema.py
CURRENT_SCHEMA = {
    "N_healthy": {"thresholds": [0.3, 0.56], "labels": ["depleted", "reduced", "adequate"]},
    "N_deletion": {"thresholds": [0.1, 0.3, 0.5], "labels": ["minimal", "growing", "approaching_cliff", "past_cliff"]},
    "ATP": {"thresholds": [0.2, 0.5, 0.79], "labels": ["collapsed", "crisis", "compromised", "healthy"]},
    "ROS": {"thresholds": [0.1, 0.25], "labels": ["basal", "elevated", "pathological"]},
    "NAD": {"thresholds": [0.3, 0.7], "labels": ["depleted", "declining", "robust"]},
    "Senescent_fraction": {"thresholds": [0.1, 0.4], "labels": ["minimal", "emerging", "severe"]},
    "Membrane_potential": {"thresholds": [0.3, 0.7], "labels": ["collapsed", "impaired", "intact"]},
    "N_point": {"thresholds": [0.1, 0.3], "labels": ["low", "moderate", "high"]},
}

print("Bin counts and means:")
for var_name, schema in CURRENT_SCHEMA.items():
    print(f"\n{var_name}:")
    for label in schema["labels"]:
        stats = global_stats.get(var_name, {}).get(label)
        if stats:
            print(f"  {label}: count={stats['count']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        else:
            print(f"  {label}: missing")

# Propose new centers = empirical mean (if count > 10)
new_centers = {}
for var_name, schema in CURRENT_SCHEMA.items():
    centers = []
    for label in schema["labels"]:
        stats = global_stats.get(var_name, {}).get(label)
        if stats and stats['count'] >= 10:
            centers.append(round(stats['mean'], 3))
        else:
            # keep existing center (need to fetch from ca_schema.py)
            # We'll skip for now
            continue
    if len(centers) == len(schema["labels"]):
        new_centers[var_name] = centers

print("\nProposed new centers (empirical means):")
for var_name, centers in new_centers.items():
    print(f"{var_name}: {centers}")

# Save proposal
with open(edge_dir / "new_centers_proposal.json", 'w') as f:
    json.dump(new_centers, f, indent=2)
print(f"\nSaved to {edge_dir}/new_centers_proposal.json")