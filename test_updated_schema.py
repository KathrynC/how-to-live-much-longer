#!/usr/bin/env python3
"""
Test updated CA schema (centers only) on a small validation subset.
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '.')

# Load original schema
from ca_schema import BIN_SCHEMA as ORIGINAL_SCHEMA
# Load updated schema from patch
updated_schema_path = Path("artifacts/ca_ode_validation/schema_patch.py")
if not updated_schema_path.exists():
    print("Updated schema patch not found. Run adjust_ca_schema.py first.")
    sys.exit(1)

# Execute the patch file to get UPDATED_SCHEMA
exec(open(updated_schema_path).read())
# Now variable BIN_SCHEMA is defined in local namespace
UPDATED_SCHEMA = BIN_SCHEMA

print("Testing updated CA schema (centers only)")
print("=" * 60)

# Monkey-patch ca_schema module
import ca_schema
original_backup = ca_schema.BIN_SCHEMA
ca_schema.BIN_SCHEMA = UPDATED_SCHEMA
# Also need to update CA_VAR_ORDER? unchanged
# Also need to update functions that depend on BIN_SCHEMA? They reference module-level variable, so they'll use updated.

# Now import other CA modules that depend on ca_schema
from simulator import simulate
from ca_simulator import run_single_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state, continuous_exemplar, CA_VAR_ORDER

# Load a small subset of patients (2 normal)
def load_sample_patients():
    normal_path = Path("artifacts/sample_patients_100.json")
    with open(normal_path, 'r') as f:
        data = json.load(f)
    return data["patients"][:2]  # first two

patients = load_sample_patients()
from constants import PATIENT_NAMES, INTERVENTION_NAMES

# Define two interventions
interventions = {
    "no_treatment": {k: 0.0 for k in INTERVENTION_NAMES},
    "conservative": {
        "rapamycin_dose": 0.1,
        "nad_supplement": 0.3,
        "senolytic_dose": 0.1,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.3,
    },
}

print(f"Testing {len(patients)} patients × {len(interventions)} interventions = {len(patients)*len(interventions)} runs")

# Metrics storage
metrics_original = []
metrics_updated = []

# Helper functions from validate_ca_ode_bridge
def reconstruct_ca_continuous(ca_trajectory):
    """Convert CA discrete trajectory to continuous using exemplars."""
    cont_traj = []
    for discrete_state in ca_trajectory:
        cont = continuous_exemplar(discrete_state)
        cont_traj.append(cont)
    return np.array(cont_traj)

def compute_continuous_rmse(ode_states, ca_cont_traj, dt_ode=0.01, dt_ca=0.25):
    ca_n_steps = ca_cont_traj.shape[0] - 1
    errors = []
    for ca_step in range(ca_n_steps + 1):
        ca_time = ca_step * dt_ca
        ode_idx = min(int(round(ca_time / dt_ode)), len(ode_states) - 1)
        ode_vals = ode_states[ode_idx]
        ca_vals = ca_cont_traj[ca_step]
        sq_err = np.square(ode_vals - ca_vals)
        errors.append(sq_err)
    errors = np.array(errors)
    rmse_per_var = np.sqrt(np.mean(errors, axis=0))
    overall_rmse = np.sqrt(np.mean(errors))
    return rmse_per_var, overall_rmse

# Run simulations
for patient in patients:
    patient_dict = {k: patient.get(k, 0.0) for k in PATIENT_NAMES}
    for int_name, intervention in interventions.items():
        print(f"  {patient['_id']} {int_name}...")
        # ODE simulation (same for both schemas)
        ode_result = simulate(patient=patient_dict, intervention=intervention)
        # CA simulation with updated schema (already patched)
        ca_result = run_single_cell(patient=patient_dict, intervention=intervention)
        
        # Fidelity (bin agreement)
        fidelity = _fidelity_stats(ca_result["trajectory"], ode_result, patient_dict)
        # Continuous RMSE
        ca_cont_traj = reconstruct_ca_continuous(ca_result["trajectory"])
        rmse_per_var, overall_rmse = compute_continuous_rmse(ode_result["states"], ca_cont_traj)
        
        metrics_updated.append({
            "patient": patient['_id'],
            "intervention": int_name,
            "overall_agreement": fidelity.get("overall_agreement", 0.0),
            "overall_rmse": overall_rmse,
        })

# Restore original schema
ca_schema.BIN_SCHEMA = original_backup

# Now run with original schema (need to re-import ca_simulator because it may have cached schema?)
# For safety, we'll restart? Instead, we can re-import ca_simulator after restoring schema.
# Let's reload the module.
import importlib
importlib.reload(ca_schema)
# Re-import ca_simulator to pick up original schema
importlib.reload(sys.modules['ca_simulator'])
from ca_simulator import run_single_cell as run_single_cell_original

print("\nRunning with original schema...")
for patient in patients:
    patient_dict = {k: patient.get(k, 0.0) for k in PATIENT_NAMES}
    for int_name, intervention in interventions.items():
        print(f"  {patient['_id']} {int_name}...")
        ode_result = simulate(patient=patient_dict, intervention=intervention)
        ca_result = run_single_cell_original(patient=patient_dict, intervention=intervention)
        fidelity = _fidelity_stats(ca_result["trajectory"], ode_result, patient_dict)
        ca_cont_traj = reconstruct_ca_continuous(ca_result["trajectory"])
        rmse_per_var, overall_rmse = compute_continuous_rmse(ode_result["states"], ca_cont_traj)
        metrics_original.append({
            "patient": patient['_id'],
            "intervention": int_name,
            "overall_agreement": fidelity.get("overall_agreement", 0.0),
            "overall_rmse": overall_rmse,
        })

# Compare
print("\n" + "=" * 60)
print("Comparison (original vs updated centers):")
print(f"{'Run':<30} {'Agreement (orig/upd)':<25} {'RMSE (orig/upd)':<25}")
for i in range(len(metrics_original)):
    orig = metrics_original[i]
    upd = metrics_updated[i]
    label = f"{orig['patient']} {orig['intervention']}"
    agree_str = f"{orig['overall_agreement']:.3f} / {upd['overall_agreement']:.3f}"
    rmse_str = f"{orig['overall_rmse']:.3f} / {upd['overall_rmse']:.3f}"
    print(f"{label:<30} {agree_str:<25} {rmse_str:<25}")

# Averages
avg_agree_orig = np.mean([m['overall_agreement'] for m in metrics_original])
avg_agree_upd = np.mean([m['overall_agreement'] for m in metrics_updated])
avg_rmse_orig = np.mean([m['overall_rmse'] for m in metrics_original])
avg_rmse_upd = np.mean([m['overall_rmse'] for m in metrics_updated])

print("\nAverages:")
print(f"  Bin agreement: {avg_agree_orig:.3f} → {avg_agree_upd:.3f} (Δ{avg_agree_upd - avg_agree_orig:+.3f})")
print(f"  Continuous RMSE: {avg_rmse_orig:.3f} → {avg_rmse_upd:.3f} (Δ{avg_rmse_upd - avg_rmse_orig:+.3f})")
print(f"  RMSE improvement: {avg_rmse_orig - avg_rmse_upd:.3f} ({100*(avg_rmse_orig - avg_rmse_upd)/avg_rmse_orig:.1f}%)")

# Save results
output_path = Path("artifacts/ca_ode_validation/schema_test_results.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump({
        "original": metrics_original,
        "updated": metrics_updated,
        "averages": {
            "original_agreement": avg_agree_orig,
            "updated_agreement": avg_agree_upd,
            "original_rmse": avg_rmse_orig,
            "updated_rmse": avg_rmse_upd,
        }
    }, f, indent=2)
print(f"\nResults saved to {output_path}")