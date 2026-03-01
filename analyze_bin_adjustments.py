#!/usr/bin/env python3
"""
Analyze edge validation data to suggest updated bin thresholds and centers.

Loads global_bin_stats from edge validation summary, computes percentiles
of ODE values per variable, and proposes new thresholds and centers that
better match empirical distribution while respecting biological constants.
"""
import json
import numpy as np
from pathlib import Path

# Load edge validation summary
edge_dir = Path("artifacts/ca_ode_validation_fixed_edge_20260222_213009")
summary_path = edge_dir / "validation_summary.json"
with open(summary_path, 'r') as f:
    data = json.load(f)

global_stats = data["global_bin_stats"]
print(f"Loaded stats for {len(global_stats)} variables")

# Current schema from ca_schema.py (for reference)
CURRENT_SCHEMA = {
    "N_healthy": {"thresholds": [0.3, 0.56], "centers": [0.15, 0.5, 0.85]},
    "N_deletion": {"thresholds": [0.1, 0.3, 0.5], "centers": [0.05, 0.2, 0.4, 0.7]},
    "ATP": {"thresholds": [0.2, 0.5, 0.79], "centers": [0.1, 0.35, 0.65, 0.9]},
    "ROS": {"thresholds": [0.1, 0.25], "centers": [0.05, 0.175, 0.4]},
    "NAD": {"thresholds": [0.3, 0.7], "centers": [0.15, 0.5, 0.85]},
    "Senescent_fraction": {"thresholds": [0.1, 0.4], "centers": [0.05, 0.25, 0.6]},
    "Membrane_potential": {"thresholds": [0.3, 0.7], "centers": [0.15, 0.5, 0.85]},
    "N_point": {"thresholds": [0.1, 0.3], "centers": [0.05, 0.2, 0.5]},
}

# Biological constants that should not be changed
FIXED_THRESHOLDS = {
    "N_deletion": [0.5],  # cliff threshold (index 2)
    # ATP crisis threshold 0.5? maybe keep
}

def flatten_values(var_name):
    """Flatten all ODE values for a variable across bins."""
    vals = []
    for bin_label, stats in global_stats.get(var_name, {}).items():
        # We don't have raw values, but we can approximate using mean and std
        # For simplicity, we'll use the mean as representative.
        # Since we have count, we can replicate mean count times.
        count = stats["count"]
        mean = stats["mean"]
        # Approximate by adding mean count times (crude but okay for percentiles)
        vals.extend([mean] * count)
    return np.array(vals)

def compute_percentiles(vals, percentiles=[10, 30, 50, 70, 90]):
    if len(vals) == 0:
        return []
    return np.percentile(vals, percentiles)

# Analyze each variable
for var_name in global_stats.keys():
    vals = flatten_values(var_name)
    if len(vals) == 0:
        continue
    p10, p30, p50, p70, p90 = compute_percentiles(vals, [10, 30, 50, 70, 90])
    print(f"\n{var_name}:")
    print(f"  n={len(vals)}, mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")
    print(f"  percentiles: 10%={p10:.3f}, 30%={p30:.3f}, 50%={p50:.3f}, 70%={p70:.3f}, 90%={p90:.3f}")
    # Current thresholds
    cur = CURRENT_SCHEMA.get(var_name, {})
    if cur:
        print(f"  current thresholds: {cur['thresholds']}")
        print(f"  current centers: {cur['centers']}")
    # Suggest new thresholds based on percentiles (choose 2-3 thresholds)
    # For variables with 3 bins (2 thresholds), use p30 and p70 maybe.
    # For variables with 4 bins (3 thresholds), use p20, p50, p80.
    # We'll just output percentiles for manual decision.

# Now propose new schema based on percentiles and fixed thresholds
print("\n" + "="*60)
print("Proposed updated BIN_SCHEMA thresholds and centers")
print("="*60)

# Helper to suggest thresholds
def suggest_thresholds(var_name, vals, n_thresholds):
    if n_thresholds == 2:
        # 3 bins
        percentiles = np.percentile(vals, [30, 70])
    elif n_thresholds == 3:
        # 4 bins
        percentiles = np.percentile(vals, [20, 50, 80])
    else:
        raise ValueError
    return [round(p, 3) for p in percentiles]

def suggest_centers(var_name, thresholds):
    # Centers are midpoints between thresholds, plus extremes
    centers = []
    # first center: halfway between min (0?) and first threshold
    # Assume min=0 for normalized variables
    centers.append(round(thresholds[0] / 2, 3))
    for i in range(len(thresholds)-1):
        centers.append(round((thresholds[i] + thresholds[i+1]) / 2, 3))
    # last center: halfway between last threshold and max (1?)
    centers.append(round((thresholds[-1] + 1.0) / 2, 3))
    return centers

proposed = {}
for var_name, cur in CURRENT_SCHEMA.items():
    vals = flatten_values(var_name)
    if len(vals) < 10:
        continue
    n_thresholds = len(cur["thresholds"])
    # Respect fixed thresholds
    fixed = FIXED_THRESHOLDS.get(var_name, [])
    if fixed:
        print(f"{var_name}: keeping fixed thresholds {fixed}")
        # TODO incorporate fixed thresholds into suggested list
        # For simplicity, skip automated suggestion for those variables
        continue
    try:
        new_thresh = suggest_thresholds(var_name, vals, n_thresholds)
        new_centers = suggest_centers(var_name, new_thresh)
        proposed[var_name] = {
            "thresholds": new_thresh,
            "centers": new_centers,
        }
    except Exception as e:
        print(f"{var_name}: error {e}")
        continue

# Print proposed updates
for var_name, prop in proposed.items():
    cur = CURRENT_SCHEMA[var_name]
    print(f"\n{var_name}:")
    print(f"  current thresholds: {cur['thresholds']}")
    print(f"  proposed thresholds: {prop['thresholds']}")
    print(f"  current centers: {cur['centers']}")
    print(f"  proposed centers: {prop['centers']}")

# Save proposal to file
with open(edge_dir / "bin_adjustment_proposal.json", 'w') as f:
    json.dump(proposed, f, indent=2)
print(f"\nProposal saved to {edge_dir}/bin_adjustment_proposal.json")