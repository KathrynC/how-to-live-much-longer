#!/usr/bin/env python3
"""Test improved rules on edge patients."""

import json
import sys
sys.path.insert(0, '.')

from simulator import simulate
from ca_simulator import run_single_cell
from ca_analytics import compute_ca_analytics

def main() -> None:
    # Load improved rules
    with open('final_tuned_rules_n_point_v2.json', 'r') as f:
        rules = json.load(f)

    # Load edge patients
    with open('artifacts/sample_patients_edge.json', 'r') as f:
        edge_data = json.load(f)
    patients = edge_data['patients'][:5]  # first 5 edge patients

    print("Testing on 5 edge patients...")
    overall_fidelities = []
    for p in patients:
        patient = {k: p[k] for k in ['baseline_age', 'baseline_heteroplasmy', 'baseline_nad_level',
                                     'genetic_vulnerability', 'metabolic_demand', 'inflammation_level']}
        intervention = {}  # no intervention
        ode_result = simulate(patient=patient, intervention=intervention)
        ca_result = run_single_cell(patient, intervention, rules=rules)
        analytics = compute_ca_analytics(ca_result, ode_result, patient)
        fid = analytics['fidelity_stats']['overall_agreement']
        overall_fidelities.append(fid)
        print(f"Patient {p['_label']}: fidelity {fid:.3f}")
        # Print N_point agreement
        npoint = analytics['fidelity_stats']['per_variable_agreement']['N_point']
        print(f"  N_point agreement: {npoint:.3f}")

    print(f"\nAverage fidelity: {sum(overall_fidelities)/len(overall_fidelities):.3f}")
    print(f"Min: {min(overall_fidelities):.3f}, Max: {max(overall_fidelities):.3f}")


if __name__ == "__main__":
    main()
