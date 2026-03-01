#!/usr/bin/env python3
"""
Validate CA-ODE bridge with continuous metrics.

Beyond bin agreement, compute:
1. Bin occupancy statistics: distribution of ODE continuous values per bin
2. Exemplar center vs bin distribution mean/median
3. Continuous RMSE between CA-reconstructed trajectory (using exemplars) and ODE
4. Timing differences for critical events (cliff crossing, ATP collapse)
"""

import json
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, '.')

from simulator import simulate
from ca_simulator import run_single_cell
from ca_analytics import _fidelity_stats
from ca_schema import (
    discretize_state, continuous_exemplar, 
    CA_VAR_ORDER, BIN_SCHEMA, bin_index, bin_count
)
from constants import PATIENT_NAMES, INTERVENTION_NAMES
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

def select_patient_subset(patients, n=10):
    """Select a diverse subset of patients."""
    # Simple selection: first n
    return patients[:n]

def define_intervention_profiles():
    """Return a dict of named intervention profiles."""
    return {
        "no_treatment": {k: 0.0 for k in INTERVENTION_NAMES},
        "conservative": {
            "rapamycin_dose": 0.1,
            "nad_supplement": 0.3,
            "senolytic_dose": 0.1,
            "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0,
            "exercise_level": 0.3,
        },
        "aggressive": {
            "rapamycin_dose": 0.5,
            "nad_supplement": 0.9,
            "senolytic_dose": 0.8,
            "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0,
            "exercise_level": 0.5,
        },
        "transplant_focused": {
            "rapamycin_dose": 0.0,
            "nad_supplement": 0.0,
            "senolytic_dose": 0.0,
            "yamanaka_intensity": 0.0,
            "transplant_rate": 0.5,
            "exercise_level": 0.0,
        },
    }

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
                "count": len(arr),
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
        ode_vals = ode_states[ode_idx]
        ca_vals = ca_cont_traj[ca_step]
        sq_err = np.square(ode_vals - ca_vals)
        errors.append(sq_err)
    errors = np.array(errors)  # shape (n_steps+1, 8)
    rmse_per_var = np.sqrt(np.mean(errors, axis=0))
    overall_rmse = np.sqrt(np.mean(errors))
    return rmse_per_var, overall_rmse

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
    
    # Find first time crossing threshold
    def first_crossing(times, values, threshold, direction='above'):
        if direction == 'above':
            crosses = values > threshold
        else:
            crosses = values < threshold
        if np.any(crosses):
            idx = np.where(crosses)[0][0]
            return float(times[idx])
        return None
    
    events['ode'] = {
        'cliff_crossing': first_crossing(time_ode, ode_n_del, cliff_threshold, 'above'),
        'atp_crisis': first_crossing(time_ode, ode_atp, atp_crisis_threshold, 'below'),
        'senescent_severe': first_crossing(time_ode, ode_sen, sen_severe_threshold, 'above'),
    }
    
    # CA events: need to convert discrete bins to continuous exemplars
    ca_cont_traj = reconstruct_ca_continuous(ca_trajectory)
    ca_n_del = ca_cont_traj[:, CA_VAR_ORDER.index("N_deletion")]
    ca_atp = ca_cont_traj[:, CA_VAR_ORDER.index("ATP")]
    ca_sen = ca_cont_traj[:, CA_VAR_ORDER.index("Senescent_fraction")]
    
    events['ca'] = {
        'cliff_crossing': first_crossing(time_ca, ca_n_del, cliff_threshold, 'above'),
        'atp_crisis': first_crossing(time_ca, ca_atp, atp_crisis_threshold, 'below'),
        'senescent_severe': first_crossing(time_ca, ca_sen, sen_severe_threshold, 'above'),
    }
    
    # Compute timing differences (if both occurred)
    diffs = {}
    for event in ['cliff_crossing', 'atp_crisis', 'senescent_severe']:
        ode_t = events['ode'][event]
        ca_t = events['ca'][event]
        if ode_t is not None and ca_t is not None:
            diffs[event] = ca_t - ode_t  # positive = CA later than ODE
        else:
            diffs[event] = None
    events['timing_diff'] = diffs
    
    return events

def run_validation(patients, interventions):
    """Run validation for each patient×intervention combo."""
    results = []
    bin_distributions_global = defaultdict(lambda: defaultdict(list))
    
    total_combos = len(patients) * len(interventions)
    print(f"Running {total_combos} patient×intervention combinations...")
    
    for i, patient in enumerate(patients):
        patient_dict = {k: patient.get(k, 0.0) for k in PATIENT_NAMES}
        
        for int_name, intervention in interventions.items():
            print(f"  Patient {i+1}/{len(patients)} {int_name}...")
            
            # Run ODE simulation
            try:
                ode_result = simulate(patient=patient_dict, intervention=intervention)
            except Exception as e:
                print(f"    ODE simulation failed: {e}")
                continue
            
            # Run CA simulation
            try:
                ca_result = run_single_cell(patient=patient_dict, intervention=intervention, rules=TUNED_RULES)
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
            
            # Continuous RMSE
            ca_cont_traj = reconstruct_ca_continuous(ca_result["trajectory"])
            rmse_per_var, overall_rmse = compute_continuous_rmse(
                ode_result["states"], ca_cont_traj
            )
            
            # Critical event timing
            time_ode = np.arange(0, 30.0 + 0.01, 0.01)  # ODE timesteps
            time_ca = np.arange(0, 30.0 + 0.25, 0.25)   # CA timesteps
            events = detect_critical_events(
                ode_result["states"], time_ode,
                ca_result["trajectory"], time_ca
            )
            
            # Store results
            results.append({
                "patient_id": patient.get("_id", f"patient_{i}"),
                "patient_category": patient.get("_category", "normal"),
                "intervention": int_name,
                "fidelity": fidelity,
                "bin_stats": bin_stats,
                "exemplar_comparison": exemplar_comparison,
                "continuous_rmse": {
                    "per_variable": {var: float(rmse_per_var[idx]) 
                                    for idx, var in enumerate(CA_VAR_ORDER)},
                    "overall": float(overall_rmse),
                },
                "critical_events": events,
                "ode_summary": {
                    "final_het": float(ode_result["heteroplasmy"][-1]),
                    "final_atp": float(ode_result["states"][-1, 2]),
                },
                "ca_summary": {
                    "final_state": ca_result["final_state"],
                },
            })
    
    # Compute global bin statistics across all runs
    global_stats = compute_bin_statistics(bin_distributions_global)
    global_exemplar_comparison = compare_exemplar_centers(global_stats)
    
    return results, global_stats, global_exemplar_comparison

def aggregate_validation(results):
    """Aggregate validation metrics across all runs."""
    if not results:
        return {}
    
    # Aggregate continuous RMSE
    rmse_per_var_sums = {var: 0.0 for var in CA_VAR_ORDER}
    rmse_counts = {var: 0 for var in CA_VAR_ORDER}
    overall_rmse_list = []
    
    # Aggregate fidelity
    overall_agreements = []
    
    # Aggregate timing differences
    timing_diffs = defaultdict(list)
    
    for r in results:
        # RMSE
        rmse_dict = r.get("continuous_rmse", {}).get("per_variable", {})
        for var, val in rmse_dict.items():
            rmse_per_var_sums[var] += val
            rmse_counts[var] += 1
        overall_rmse = r.get("continuous_rmse", {}).get("overall", 0.0)
        if overall_rmse > 0:
            overall_rmse_list.append(overall_rmse)
        
        # Fidelity
        fidelity = r.get("fidelity", {})
        overall_agreement = fidelity.get("overall_agreement", 0.0)
        overall_agreements.append(overall_agreement)
        
        # Timing differences
        events = r.get("critical_events", {}).get("timing_diff", {})
        for event, diff in events.items():
            if diff is not None:
                timing_diffs[event].append(diff)
    
    # Averages
    avg_rmse_per_var = {}
    for var in CA_VAR_ORDER:
        if rmse_counts[var] > 0:
            avg_rmse_per_var[var] = rmse_per_var_sums[var] / rmse_counts[var]
        else:
            avg_rmse_per_var[var] = 0.0
    
    avg_overall_rmse = np.mean(overall_rmse_list) if overall_rmse_list else 0.0
    avg_overall_agreement = np.mean(overall_agreements) if overall_agreements else 0.0
    
    # Timing difference statistics
    timing_stats = {}
    for event, diffs in timing_diffs.items():
        timing_stats[event] = {
            "mean": float(np.mean(diffs)),
            "std": float(np.std(diffs)),
            "median": float(np.median(diffs)),
            "n": len(diffs),
        }
    
    return {
        "average_continuous_rmse_per_variable": avg_rmse_per_var,
        "average_overall_rmse": avg_overall_rmse,
        "average_overall_agreement": avg_overall_agreement,
        "timing_differences": timing_stats,
        "n_runs": len(results),
    }

def main():
    """Main validation routine."""
    print("CA-ODE Bridge Validation (Continuous Metrics)")
    print("=" * 60)
    
    # Load patients
    patients = load_sample_patients()
    if not patients:
        # Fallback: create a few synthetic patients
        print("No sample patients found, creating synthetic ones.")
        patients = [
            {"_id": "young_healthy", "baseline_age": 30.0, "baseline_heteroplasmy": 0.05,
             "baseline_nad_level": 0.95, "genetic_vulnerability": 0.8,
             "metabolic_demand": 1.0, "inflammation_level": 0.1},
            {"_id": "middle_aged", "baseline_age": 55.0, "baseline_heteroplasmy": 0.25,
             "baseline_nad_level": 0.7, "genetic_vulnerability": 1.0,
             "metabolic_demand": 1.0, "inflammation_level": 0.3},
            {"_id": "near_cliff", "baseline_age": 70.0, "baseline_heteroplasmy": 0.45,
             "baseline_nad_level": 0.5, "genetic_vulnerability": 1.2,
             "metabolic_demand": 1.0, "inflammation_level": 0.5},
        ]
    
    # Filter out edge patients (those with '_category' field) — keep only normal patients
    normal_patients = [p for p in patients if '_category' not in p]
    if len(normal_patients) < 10:
        # fallback: use all patients
        normal_patients = patients
    print(f"Using {len(normal_patients)} patients (normal subset)")
    
    # Select subset (limit for speed)
    patients = select_patient_subset(normal_patients, n=10)
    
    # Define interventions
    interventions = define_intervention_profiles()
    # Use all four interventions
    # interventions = {k: interventions[k] for k in list(interventions.keys())[:2]}
    
    # Run validation
    results, global_stats, global_exemplar_comparison = run_validation(
        patients, interventions
    )
    
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
    output_dir = Path("artifacts/ca_ode_validation")
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