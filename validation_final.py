#!/usr/bin/env python3
"""
Validation after final schema updates.
"""
import importlib
import sys
sys.path.insert(0, '.')
# Reload ca_schema to pick up changes
import ca_schema
importlib.reload(ca_schema)
from ca_schema import BIN_SCHEMA
print("Updated BIN_SCHEMA centers:")
for var, schema in BIN_SCHEMA.items():
    print(f"{var}: {schema['centers']}")

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import datetime

from simulator import simulate
from ca_simulator import run_single_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state, CA_VAR_ORDER
from constants import PATIENT_NAMES, INTERVENTION_NAMES
from ca_rules import load_rules

TUNED_RULES = load_rules("final_tuned_rules.json")
print(f"Loaded {len(TUNED_RULES)} tuned rules")

# Load edge patients, take 2 cliff_boundary
edge_path = Path("artifacts/sample_patients_edge.json")
with open(edge_path, 'r') as f:
    data = json.load(f)
edge_patients = data["patients"]
cliff = [p for p in edge_patients if p.get("_category") == "cliff_boundary"][:2]
patients = cliff

interventions = {
    "no_treatment": {k: 0.0 for k in INTERVENTION_NAMES},
    "conservative": {k: 0.0 for k in INTERVENTION_NAMES} | {"rapamycin_dose": 0.25, "nad_supplement": 0.5, "senolytic_dose": 0.25},
}

print(f"Running {len(patients)} × {len(interventions)} combos")
results = []
for p in patients:
    patient_dict = {k: p[k] for k in PATIENT_NAMES}
    for inv_name, inv in interventions.items():
        print(f"  {p['_id']} {inv_name}")
        try:
            ode = simulate(patient=patient_dict, intervention=inv)
            ca = run_single_cell(patient=patient_dict, intervention=inv, rules=TUNED_RULES)
        except Exception as e:
            print(f"    error: {e}")
            continue
        fidelity = _fidelity_stats(ca["trajectory"], ode, patient_dict)
        results.append({
            "agreement": fidelity["overall_agreement"],
            "rmse": fidelity.get("continuous_rmse", {}).get("overall", None),
        })

if results:
    avg_agree = np.mean([r["agreement"] for r in results])
    rmse_vals = [r["rmse"] for r in results if r["rmse"] is not None]
    avg_rmse = np.mean(rmse_vals) if rmse_vals else None
    print(f"\nAverage bin agreement: {avg_agree:.3f}")
    if avg_rmse:
        print(f"Average RMSE: {avg_rmse:.3f}")
else:
    print("No results")

# Save
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = Path(f"artifacts/validation_final_{timestamp}")
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / "summary.json", 'w') as f:
    json.dump({
        "avg_agreement": avg_agree if results else None,
        "avg_rmse": avg_rmse,
        "n": len(results),
    }, f, indent=2)
print(f"Saved to {outdir}/")