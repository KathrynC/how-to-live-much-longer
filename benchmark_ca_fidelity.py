#!/usr/bin/env python3
"""Benchmark CA vs ODE fidelity across patient and intervention scenarios."""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

from simulator import simulate
from ca_simulator import run_single_cell
from ca_analytics import _fidelity_stats
from constants import PATIENT_NAMES, INTERVENTION_NAMES

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
        "metabolic_optimizer": {
            "rapamycin_dose": 0.25,
            "nad_supplement": 0.5,
            "senolytic_dose": 0.25,
            "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0,
            "exercise_level": 0.6,
        },
    }

def run_benchmark(patients, interventions):
    """Run CA and ODE simulations for each patient×intervention combo."""
    results = []
    
    total_combos = len(patients) * len(interventions)
    print(f"Running {total_combos} patient×intervention combinations...")
    
    for i, patient in enumerate(patients):
        # Extract patient parameters (ensure they match PATIENT_NAMES)
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
                ca_result = run_single_cell(patient=patient_dict, intervention=intervention)
            except Exception as e:
                print(f"    CA simulation failed: {e}")
                continue
            
            # Compute fidelity stats
            fidelity = _fidelity_stats(ca_result["trajectory"], ode_result, patient_dict)
            
            # Compute other metrics (optional)
            # ...
            
            results.append({
                "patient_id": patient.get("_id", f"patient_{i}"),
                "patient_category": patient.get("_category", "normal"),
                "intervention": int_name,
                "fidelity": fidelity,
                "ode_summary": {
                    "final_het": float(ode_result["heteroplasmy"][-1]),
                    "final_atp": float(ode_result["states"][-1, 2]),
                },
                "ca_summary": {
                    "final_state": ca_result["final_state"],
                },
            })
    
    return results

def aggregate_fidelity(results):
    """Aggregate fidelity statistics across all runs."""
    if not results:
        return {}
    
    # Per-variable agreement accumulation
    var_agreement_sums = {}
    var_counts = {}
    overall_agreements = []
    
    for r in results:
        fidelity = r.get("fidelity")
        if not fidelity:
            continue
        per_var = fidelity.get("per_variable_agreement", {})
        overall = fidelity.get("overall_agreement", 0.0)
        
        for var, agree in per_var.items():
            var_agreement_sums[var] = var_agreement_sums.get(var, 0.0) + agree
            var_counts[var] = var_counts.get(var, 0) + 1
        
        overall_agreements.append(overall)
    
    # Compute averages
    avg_per_var = {}
    for var in var_agreement_sums:
        avg_per_var[var] = var_agreement_sums[var] / var_counts[var]
    
    avg_overall = np.mean(overall_agreements) if overall_agreements else 0.0
    std_overall = np.std(overall_agreements) if overall_agreements else 0.0
    
    # Distribution of overall agreement (we'll compute counts directly)
    
    return {
        "average_per_variable_agreement": avg_per_var,
        "average_overall_agreement": avg_overall,
        "std_overall_agreement": std_overall,
        "n_runs": len(results),
        "overall_agreement_distribution": {
            "excellent": sum(1 for a in overall_agreements if a >= 0.9),
            "good": sum(1 for a in overall_agreements if 0.7 <= a < 0.9),
            "fair": sum(1 for a in overall_agreements if 0.5 <= a < 0.7),
            "poor": sum(1 for a in overall_agreements if a < 0.5),
        },
    }

def main():
    """Main benchmark routine."""
    print("CA vs ODE Fidelity Benchmark")
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
            {"_id": "post_cliff", "baseline_age": 80.0, "baseline_heteroplasmy": 0.65,
             "baseline_nad_level": 0.3, "genetic_vulnerability": 1.5,
             "metabolic_demand": 1.0, "inflammation_level": 0.7},
        ]
    
    # Select subset (limit for speed)
    patients = select_patient_subset(patients, n=6)
    
    # Define interventions
    interventions = define_intervention_profiles()
    
    # Run benchmark
    results = run_benchmark(patients, interventions)
    
    # Aggregate statistics
    stats = aggregate_fidelity(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Fidelity Benchmark Results")
    print(f"Total runs: {stats['n_runs']}")
    print(f"Average overall agreement: {stats['average_overall_agreement']:.3f} ± {stats['std_overall_agreement']:.3f}")
    print("\nPer-variable agreement:")
    for var, agree in stats['average_per_variable_agreement'].items():
        print(f"  {var}: {agree:.3f}")
    
    print("\nOverall agreement distribution:")
    dist = stats['overall_agreement_distribution']
    for cat, count in dist.items():
        print(f"  {cat}: {count} runs")
    
    # Save results to JSON
    output_path = Path("output/ca_fidelity_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "summary": stats,
            "detailed_results": results,
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")
    
    # Identify problematic variables (agreement < 0.5)
    problematic = [var for var, agree in stats['average_per_variable_agreement'].items() if agree < 0.5]
    if problematic:
        print(f"\n⚠️  Low agreement variables (<0.5): {problematic}")
    else:
        print("\n✅ All variables have reasonable agreement (≥0.5).")
    
    # Identify runs with poor agreement (<0.5 overall)
    poor_runs = []
    for r in results:
        fidelity = r.get("fidelity")
        if fidelity and fidelity.get("overall_agreement", 1.0) < 0.5:
            poor_runs.append({
                "patient_id": r["patient_id"],
                "intervention": r["intervention"],
                "agreement": fidelity["overall_agreement"],
            })
    
    if poor_runs:
        print(f"\n⚠️  {len(poor_runs)} runs with overall agreement <0.5:")
        for pr in poor_runs[:5]:  # limit display
            print(f"  {pr['patient_id']} + {pr['intervention']}: {pr['agreement']:.3f}")
        if len(poor_runs) > 5:
            print(f"  ... and {len(poor_runs)-5} more")

if __name__ == "__main__":
    main()