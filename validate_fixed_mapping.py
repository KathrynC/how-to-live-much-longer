#!/usr/bin/env python3
"""
Validate CA-ODE bridge with fixed copy-count vs fraction mapping.
Small subset for quick testing.
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
from ca_schema import (
    discretize_state, continuous_exemplar, 
    CA_VAR_ORDER, BIN_SCHEMA, bin_index, bin_count
)
from constants import PATIENT_NAMES, INTERVENTION_NAMES, DEFAULT_INTERVENTION
from ca_rules import load_rules

# Load tuned rules if available
TUNED_RULES = None
try:
    TUNED_RULES = load_rules("final_tuned_rules.json")
    print(f"Loaded {len(TUNED_RULES)} tuned rules from final_tuned_rules.json")
except FileNotFoundError:
    print("No tuned rules file found, using default rules")

def load_sample_patients():
    """Load sample patients from artifacts."""
    normal_path = Path("artifacts/sample_patients_100.json")
    edge_path = Path("artifacts/sample_patients_edge.json")
    patients = []
    
    if normal_path.exists():
        with open(normal_path, 'r') as f:
            data = json.load(f)
            patients.extend(data["patients"])
            print(f"Loaded {len(data['patients'])} normal patients")
    
    if edge_path.exists():
        with open(edge_path, 'r') as f:
            data = json.load(f)
            patients.extend(data["patients"])
            print(f"Loaded {len(data['patients'])} edge patients")
    
    return patients

def select_patient_subset(patients, n=5):
    """Select a diverse subset of patients."""
    # Simple selection: first n
    return patients[:n]

def collect_bin_distributions(ode_states, ca_trajectory, dt_ode=0.01, dt_ca=0.25):
    """Collect ODE continuous values for each bin of each variable.
    
    Args:
        ode_states: np.array shape (n_ode_steps+1, 8)
        ca_trajectory: list of dict discrete states (n_ca_steps+1)
        dt_ode, dt_ca: timesteps
    
    Returns:
        dict: variable -> bin_label -> list of ODE continuous values
    """
    bin_values = defaultdict(lambda: defaultdict(list))
    ca_n_steps = len(ca_trajectory) - 1
    
    for ca_step in range(ca_n_steps + 1):
        ca_time = ca_step * dt_ca
        ode_idx = min(int(round(ca_time / dt_ode)), len(ode_states) - 1)
        ode_continuous = ode_states[ode_idx]  # shape (8,)
        ca_discrete = ca_trajectory[ca_step]
        
        for var_idx, var_name in enumerate(CA_VAR_ORDER):
            bin_label = ca_discrete[var_name]
            ode_val = ode_continuous[var_idx]
            bin_values[var_name][bin_label].append(ode_val)
    
    return bin_values

def compute_bin_statistics(bin_values):
    """Compute mean, median, std of ODE values per bin."""
    stats = {}
    for var_name, bins in bin_values.items():
        stats[var_name] = {}
        for bin_label, values in bins.items():
            if not values:
                continue
            arr = np.array(values)
            stats[var_name][bin_label] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "count": len(values),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    return stats

def compare_exemplar_centers(bin_stats):
    """Compare bin centers (exemplars) to empirical ODE distribution.
    
    Returns dict with difference metrics.
    """
    comparison = {}
    for var_name, bins in bin_stats.items():
        comparison[var_name] = {}
        schema = BIN_SCHEMA[var_name]
        for bin_label, stat in bins.items():
            idx = schema["labels"].index(bin_label)
            center = schema["centers"][idx]
            diff_mean = center - stat["mean"]
            diff_median = center - stat["median"]
            comparison[var_name][bin_label] = {
                "center": center,
                "mean": stat["mean"],
                "median": stat["median"],
                "diff_mean": diff_mean,
                "diff_median": diff_median,
                "relative_diff_mean": diff_mean / (stat["std"] + 1e-8),
                "relative_diff_median": diff_median / (stat["std"] + 1e-8),
            }
    return comparison

def reconstruct_ca_continuous(ca_trajectory):
    """Convert CA discrete trajectory to continuous using exemplars."""
    cont_traj = []
    for discrete_state in ca_trajectory:
        cont = continuous_exemplar(discrete_state)
        cont_traj.append(cont)
    return np.array(cont_traj)  # shape (n_steps+1, 8)

def compute_continuous_rmse(ode_states, ca_cont_traj, dt_ode=0.01, dt_ca=0.25):
    """Compute RMSE between ODE and CA-reconstructed continuous trajectories.
    
    Aligns timesteps by subsampling ODE at CA timesteps.
    """
    ca_n_steps = ca_cont_traj.shape[0] - 1
    errors = []
    for ca_step in range(ca_n_steps + 1):
        ca_time = ca_step * dt_ca
        ode_idx = min(int(round(ca_time / dt_ode)), len(ode_states) - 1)
        error = ca_cont_traj[ca_step] - ode_states[ode_idx]
        errors.append(error)
    errors = np.array(errors)  # (n_steps+1, 8)
    rmse_per_var = np.sqrt(np.mean(errors**2, axis=0))
    overall_rmse = np.sqrt(np.mean(errors**2))
    return {
        "per_variable": {CA_VAR_ORDER[i]: float(rmse_per_var[i]) for i in range(8)},
        "overall": float(overall_rmse),
    }

def detect_critical_events(ode_states, time_ode, ca_trajectory, time_ca):
    """Detect timing of critical events in ODE and CA.
    
    Events:
        - N_deletion crosses cliff threshold (0.5)
        - ATP crosses crisis threshold (0.2)
        - Senescent_fraction crosses severe threshold (0.4)
    
    Returns dict with event times for ODE and CA (if occurred).
    """
    # Thresholds from BIN_SCHEMA
    cliff_threshold = 0.5  # N_deletion threshold between approaching_cliff and past_cliff
    atp_crisis_threshold = 0.2
    sen_severe_threshold = 0.4
    
    events = {}
    
    # ODE events
    ode_n_del = ode_states[:, CA_VAR_ORDER.index("N_deletion")]
    ode_atp = ode_states[:, CA_VAR_ORDER.index("ATP")]
    ode_sen = ode_states[:, CA_VAR_ORDER.index("Senescent_fraction")]
    
    # Find first crossing
    for i in range(1, len(time_ode)):
        if ode_n_del[i-1] < cliff_threshold <= ode_n_del[i]:
            events["cliff_crossing"] = {"ode": float(time_ode[i])}
            break
    for i in range(1, len(time_ode)):
        if ode_atp[i-1] > atp_crisis_threshold >= ode_atp[i]:
            events["atp_crisis"] = {"ode": float(time_ode[i])}
            break
    for i in range(1, len(time_ode)):
        if ode_sen[i-1] < sen_severe_threshold <= ode_sen[i]:
            events["senescent_severe"] = {"ode": float(time_ode[i])}
            break
    
    # CA events (based on discrete labels)
    ca_n_del_labels = [s["N_deletion"] for s in ca_trajectory]
    ca_atp_labels = [s["ATP"] for s in ca_trajectory]
    ca_sen_labels = [s["Senescent_fraction"] for s in ca_trajectory]
    
    # Map labels to thresholds
    # For simplicity, we'll just check when label changes to past_cliff, crisis, severe
    for i in range(1, len(time_ca)):
        if ca_n_del_labels[i-1] != "past_cliff" and ca_n_del_labels[i] == "past_cliff":
            events.setdefault("cliff_crossing", {})["ca"] = float(time_ca[i])
            break
    for i in range(1, len(time_ca)):
        if ca_atp_labels[i-1] != "crisis" and ca_atp_labels[i] == "crisis":
            events.setdefault("atp_crisis", {})["ca"] = float(time_ca[i])
            break
    for i in range(1, len(time_ca)):
        if ca_sen_labels[i-1] != "severe" and ca_sen_labels[i] == "severe":
            events.setdefault("senescent_severe", {})["ca"] = float(time_ca[i])
            break
    
    return events

def run_validation(patients, interventions):
    """Run validation for given patients and interventions."""
    results = []
    bin_distributions_global = defaultdict(lambda: defaultdict(list))
    
    for p_idx, p in enumerate(patients):
        patient_dict = {k: p[k] for k in PATIENT_NAMES}
        for interv_name, interv in interventions.items():
            print(f"  Patient {p_idx+1}/{len(patients)} {interv_name}...")
            
            # Run ODE simulation
            try:
                ode_result = simulate(patient=patient_dict, intervention=interv)
            except Exception as e:
                print(f"    ODE simulation failed: {e}")
                continue
            
            # Run CA simulation with tuned rules
            try:
                ca_result = run_single_cell(patient=patient_dict, intervention=interv, rules=TUNED_RULES)
            except Exception as e:
                print(f"    CA simulation failed: {e}")
                continue
            
            # Bin-level fidelity (existing metric)
            fidelity = _fidelity_stats(ca_result["trajectory"], ode_result, patient_dict)
            
            # Collect bin distributions for this run
            bin_values = collect_bin_distributions(
                ode_result["states"], 
                ca_result["trajectory"]
            )
            # Aggregate into global distribution
            for var, bins in bin_values.items():
                for label, vals in bins.items():
                    bin_distributions_global[var][label].extend(vals)
            
            # Compute bin statistics for this run
            bin_stats = compute_bin_statistics(bin_values)
            exemplar_comparison = compare_exemplar_centers(bin_stats)
            
            # Reconstruct continuous CA trajectory
            ca_continuous = reconstruct_ca_continuous(ca_result["trajectory"])
            rmse = compute_continuous_rmse(ode_result["states"], ca_continuous)
            
            # Critical event timing
            time_ode = np.arange(0, ode_result["states"].shape[0] * 0.01, 0.01)
            time_ca = np.arange(0, len(ca_result["trajectory"]) * 0.25, 0.25)
            events = detect_critical_events(ode_result["states"], time_ode,
                                            ca_result["trajectory"], time_ca)
            
            results.append({
                "patient_id": p["_id"],
                "patient_category": p.get("_category", "normal"),
                "intervention": interv_name,
                "fidelity": fidelity,
                "bin_stats": bin_stats,
                "exemplar_comparison": exemplar_comparison,
                "continuous_rmse": rmse,
                "critical_events": events,
                "ode_summary": {
                    "final_state": [float(x) for x in ode_result["states"][-1]],
                    "final_heteroplasmy": float(ode_result["heteroplasmy"][-1]),
                    "final_deletion_heteroplasmy": float(ode_result["deletion_heteroplasmy"][-1]),
                },
                "ca_summary": {
                    "final_state": ca_result["trajectory"][-1],
                    "final_attractor": ca_result.get("attractor", "unknown"),
                },
            })
    
    global_stats = compute_bin_statistics(bin_distributions_global)
    global_exemplar_comparison = compare_exemplar_centers(global_stats)
    
    return results, global_stats, global_exemplar_comparison

def aggregate_validation(results):
    """Aggregate metrics across all runs."""
    if not results:
        return {}
    
    n_runs = len(results)
    overall_agreement = np.mean([r["fidelity"]["overall_agreement"] for r in results])
    overall_rmse = np.mean([r["continuous_rmse"]["overall"] for r in results])
    
    # Per-variable RMSE
    rmse_per_var = defaultdict(list)
    for r in results:
        for var, val in r["continuous_rmse"]["per_variable"].items():
            rmse_per_var[var].append(val)
    avg_rmse_per_var = {var: np.mean(vals) for var, vals in rmse_per_var.items()}
    
    # Timing differences
    timing_diffs = defaultdict(list)
    for r in results:
        for event_name, event_data in r["critical_events"].items():
            if "ode" in event_data and "ca" in event_data:
                diff = event_data["ca"] - event_data["ode"]
                timing_diffs[event_name].append(diff)
    
    timing_stats = {}
    for event, diffs in timing_diffs.items():
        timing_stats[event] = {
            "mean": float(np.mean(diffs)),
            "std": float(np.std(diffs)),
            "median": float(np.median(diffs)),
            "n": len(diffs),
        }
    
    return {
        "average_overall_agreement": overall_agreement,
        "average_overall_rmse": overall_rmse,
        "average_continuous_rmse_per_variable": avg_rmse_per_var,
        "timing_differences": timing_stats,
        "n_runs": n_runs,
    }

def main():
    print("CA-ODE Bridge Validation with Fixed Mapping (Small Test)")
    print("=" * 60)
    
    # Load patients
    all_patients = load_sample_patients()
    normal_patients = [p for p in all_patients if p.get("_category", "normal") == "normal"]
    
    # Use small subset
    patients = select_patient_subset(normal_patients, n=5)
    print(f"Using {len(patients)} patients")
    
    # Define interventions (full 6D parameters, DEFAULT_INTERVENTION as base)
    interventions = {
        "no_treatment": {k: 0.0 for k in INTERVENTION_NAMES},
        "conservative": {k: 0.0 for k in INTERVENTION_NAMES} | {"rapamycin_dose": 0.25, "nad_supplement": 0.5, "senolytic_dose": 0.25},
        "aggressive": {k: 0.0 for k in INTERVENTION_NAMES} | {"rapamycin_dose": 0.75, "nad_supplement": 1.0, "senolytic_dose": 0.75},
        "transplant_focused": {k: 0.0 for k in INTERVENTION_NAMES} | {"transplant_rate": 1.0, "nad_supplement": 0.5},
    }
    print(f"Using {len(interventions)} interventions")
    
    print(f"Running {len(patients)} × {len(interventions)} = {len(patients)*len(interventions)} patient×intervention combinations...")
    
    results, global_stats, global_exemplar_comparison = run_validation(patients, interventions)
    
    if not results:
        print("No successful runs.")
        return
    
    # Aggregate statistics
    aggregated = aggregate_validation(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Validation Results Summary")
    print(f"Total runs: {aggregated['n_runs']}")
    print(f"Average overall bin agreement: {aggregated['average_overall_agreement']:.3f}")
    print(f"Average overall continuous RMSE: {aggregated['average_overall_rmse']:.3f}")
    
    print("\nAverage continuous RMSE per variable:")
    for var, rmse in aggregated['average_continuous_rmse_per_variable'].items():
        print(f"  {var}: {rmse:.4f}")
    
    print("\nTiming differences (CA - ODE years, positive = CA later):")
    for event, stats in aggregated.get('timing_differences', {}).items():
        print(f"  {event}: mean {stats['mean']:.2f} ± {stats['std']:.2f} yrs (n={stats['n']})")
    
    # Global exemplar comparison
    print("\nGlobal exemplar center vs ODE distribution mean (across all runs):")
    for var_name, bins in global_exemplar_comparison.items():
        print(f"\n  {var_name}:")
        for bin_label, comp in bins.items():
            print(f"    {bin_label}: center={comp['center']:.3f}, mean={comp['mean']:.3f}, "
                  f"diff={comp['diff_mean']:.3f} ({comp['relative_diff_mean']:.2f} σ)")
    
    # Identify largest discrepancies
    print("\n" + "=" * 60)
    print("Largest discrepancies (abs diff_mean > 0.05):")
    for var_name, bins in global_exemplar_comparison.items():
        for bin_label, comp in bins.items():
            if abs(comp['diff_mean']) > 0.05:
                print(f"  {var_name} {bin_label}: center={comp['center']:.3f}, "
                      f"mean={comp['mean']:.3f}, diff={comp['diff_mean']:.3f}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"artifacts/ca_ode_validation_fixed_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated results
    with open(output_dir / "validation_summary.json", 'w') as f:
        json.dump({
            "aggregated": aggregated,
            "global_bin_stats": global_stats,
            "global_exemplar_comparison": global_exemplar_comparison,
        }, f, indent=2, default=str)
    
    # Save detailed results (optional, large)
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    main()