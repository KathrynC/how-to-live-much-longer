import json
import numpy as np
import sys
sys.path.insert(0, '.')
from simulator import simulate
from constants import PATIENT_NAMES

# Load edge patients
edge_path = "artifacts/sample_patients_edge.json"
with open(edge_path, 'r') as f:
    data = json.load(f)
edge_patients = data["patients"]
print(f"Loaded {len(edge_patients)} edge patients")

# Collect all state values across time for first 10 patients
all_vals = [[] for _ in range(8)]  # 8 variables
for p in edge_patients[:10]:
    patient_dict = {k: p[k] for k in PATIENT_NAMES}
    result = simulate(patient=patient_dict)
    states = result["states"]  # (n_steps+1, 8)
    for i in range(8):
        all_vals[i].extend(states[:, i])

var_names = ["N_healthy", "N_deletion", "ATP", "ROS", "NAD", "Senescent_fraction", "Membrane_potential", "N_point"]
print("\nPercentiles per variable (10th, 30th, 50th, 70th, 90th):")
for i, name in enumerate(var_names):
    vals = np.array(all_vals[i])
    print(f"\n{name}: n={len(vals)}")
    print(f"  min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")
    percentiles = np.percentile(vals, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    for p, val in zip([10,20,30,40,50,60,70,80,90], percentiles):
        print(f"  {p}%: {val:.3f}", end='')
    print()

# Also compute fraction of values below current thresholds
thresholds = {
    "N_healthy": [0.3, 0.56],
    "N_deletion": [0.1, 0.3, 0.5],
    "ATP": [0.2, 0.5, 0.79],
    "ROS": [0.1, 0.25],
    "NAD": [0.3, 0.7],
    "Senescent_fraction": [0.1, 0.4],
    "Membrane_potential": [0.3, 0.7],
    "N_point": [0.1, 0.3],
}
print("\nFraction of values below each threshold:")
for i, name in enumerate(var_names):
    vals = np.array(all_vals[i])
    thresh = thresholds[name]
    print(f"\n{name}:")
    for t in thresh:
        frac = np.sum(vals < t) / len(vals)
        print(f"  <{t}: {frac:.3f}")