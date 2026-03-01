import json
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

# Collect ATP values across time for first 5 patients
all_atp = []
for p in edge_patients[:5]:
    patient_dict = {k: p[k] for k in PATIENT_NAMES}
    result = simulate(patient=patient_dict)
    all_atp.extend(result["states"][:, 2])  # ATP column

print(f"Collected {len(all_atp)} ATP values")
print(f"Min ATP: {min(all_atp):.3f}, Max ATP: {max(all_atp):.3f}")
print(f"Mean ATP: {sum(all_atp)/len(all_atp):.3f}")
# percentiles
import numpy as np
for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    print(f"ATP {p}%: {np.percentile(all_atp, p):.3f}")