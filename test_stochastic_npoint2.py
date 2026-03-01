#!/usr/bin/env python3
"""Stochastic CA test for N_point transition timing."""

import json
import numpy as np
from simulator import simulate, initial_state
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_stochastic import run_single_cell_stochastic, compute_ensemble_analytics

def main() -> None:
    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)

    with open('final_tuned_rules_n_point_v2.json', 'r') as f:
        rules = json.load(f)

    # Override confidences for testing
    for r in rules:
        if r['name'] == 'ros_drives_points':
            r['confidence'] = 0.15
        elif r['name'] == 'point_mutation_pol_gamma_errors':
            r['confidence'] = 0.15
        elif r['name'] == 'N_point_suppression':
            r['confidence'] = 0.1

    print("Running stochastic ensemble (100 trials)...")
    ensemble = run_single_cell_stochastic(patient, intervention, rules=rules, n_trials=100, seed=42)
    analytics = compute_ensemble_analytics(ensemble)

    print("\nN_point final distribution:")
    final_dist = analytics["variable_distributions"]["N_point"]
    print(final_dist)

    print("\nFirst passage time to moderate:")
    first_passage = []
    for trial in ensemble["trajectories"]:
        for i, state in enumerate(trial):
            if state['N_point'] == 'moderate':
                first_passage.append(i * 0.25)
                break
        else:
            first_passage.append(float('inf'))
    finite = [t for t in first_passage if t != float('inf')]
    mean = np.mean(finite) if finite else float("inf")
    print(f"Mean first passage time to moderate: {mean:.2f} years")
    print(f"Fraction never reaching moderate: {sum(1 for t in first_passage if t==float('inf'))/len(first_passage):.3f}")

    # Compare to ODE
    ode_result = simulate(patient=patient, intervention=intervention)
    ode_traj = [discretize_state(s) for s in ode_result['states'][::int(0.25 / 0.01)]]
    print("\nODE N_point bin trajectory (every 5 years):")
    for year in [0, 5, 10, 15, 20, 25, 30]:
        idx = int(year / 0.25)
        if idx < len(ode_traj):
            print(f"Year {year}: {ode_traj[idx]['N_point']}")


if __name__ == "__main__":
    main()
