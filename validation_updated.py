#!/usr/bin/env python3
"""
Quick validation of updated bin schema with edge patients.
"""
import json
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
import datetime

sys.path.insert(0, '.')

from simulator import simulate
from ca_simulator import run_single_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state, continuous_exemplar, CA_VAR_ORDER, BIN_SCHEMA
from constants import PATIENT_NAMES, INTERVENTION_NAMES, DEFAULT_INTERVENTION
from ca_rules import load_rules

# Load tuned rules
TUNED_RULES = load_rules("final_tuned_rules.json")
print(f"Loaded {len(TUNED_RULES)} tuned rules")

# Load edge patients
edge_path = Path("artifacts/sample_patients_edge.json")
with open(edge_path, 'r') as f:
    data = json.load(f)
edge_patients = data["patients"]
# Select cliff_boundary category
cliff_patients = [p for p in edge_patients if p.get("_category") == "cliff_boundary"]
print(f"Selected {len(cliff_patients)} cliff_boundary patients")
patients = cliff_patients[:3]  # small subset

# Interventions
interventions = {
    "no_treatment": {k: 0.0 for k in INTERVENTION_NAMES},
    "conservative": {k: 0.0 for k in INTERVENTION_NAMES} | {"rapamycin_dose": 0.25, "nad_supplement": 0.5, "senolytic_dose": 0.25},
}

print(f"Running {len(patients)} × {len(interventions)} = {len(patients)*len(interventions)} combinations")

results = []
bin_distributions = defaultdict(lambda: defaultdict(list))

for p_idx, p in enumerate(patients):
    patient_dict = {k: p[k] for k in PATIENT_NAMES}
    for interv_name, interv in interventions.items():
        print(f"  Patient {p_idx+1} {interv_name}...")
        try:
            ode_result = simulate(patient=patient_dict, intervention=interv)
            ca_result = run_single_cell(patient=patient_dict, intervention=interv, rules=TUNED_RULES)
        except Exception as e:
            print(f"    Failed: {e}")
            continue
        
        fidelity = _fidelity_stats(ca_result["trajectory"], ode_result, patient_dict)
        # Collect bin distributions (simple)
        for ca_step, discrete_state in enumerate(ca_result["trajectory"]):
            ca_time = ca_step * 0.25
            ode_idx = min(int(round(ca_time / 0.01)), len(ode_result["states"]) - 1)
            ode_continuous = ode_result["states"][ode_idx]
            for var_idx, var_name in enumerate(CA_VAR_ORDER):
                bin_label = discrete_state[var_name]
                ode_val = ode_continuous[var_idx]
                bin_distributions[var_name][bin_label].append(ode_val)
        
        results.append({
            "patient_id": p["_id"],
            "intervention": interv_name,
            "agreement": fidelity["overall_agreement"],
            "rmse": fidelity.get("continuous_rmse", {}).get("overall", None),
        })

if not results:
    print("No successful runs.")
    sys.exit(1)

# Aggregate
agreements = [r["agreement"] for r in results]
rmse_vals = [r["rmse"] for r in results if r["rmse"] is not None]
print(f"\nValidation results (n={len(results)}):")
print(f"Average bin agreement: {np.mean(agreements):.3f}")
if rmse_vals:
    print(f"Average RMSE: {np.mean(rmse_vals):.3f}")

# Compute bin statistics
bin_stats = {}
for var_name, bins in bin_distributions.items():
    bin_stats[var_name] = {}
    for label, vals in bins.items():
        arr = np.array(vals)
        bin_stats[var_name][label] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "count": len(vals),
        }

# Compare centers
print("\nBin center vs ODE mean (updated centers):")
for var_name, schema in BIN_SCHEMA.items():
    print(f"\n{var_name}:")
    for label in schema["labels"]:
        center = schema["centers"][schema["labels"].index(label)]
        stats = bin_stats.get(var_name, {}).get(label)
        if stats:
            diff = center - stats["mean"]
            print(f"  {label}: center={center:.3f}, mean={stats['mean']:.3f}, diff={diff:.3f}")
        else:
            print(f"  {label}: no data")

# Save results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"artifacts/validation_updated_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "summary.json", 'w') as f:
    json.dump({
        "average_agreement": float(np.mean(agreements)),
        "average_rmse": float(np.mean(rmse_vals)) if rmse_vals else None,
        "bin_stats": bin_stats,
        "n_runs": len(results),
    }, f, indent=2)
print(f"\nResults saved to {output_dir}/")