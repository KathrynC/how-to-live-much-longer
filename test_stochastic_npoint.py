#!/usr/bin/env python3
"""Test stochastic CA for N_point growth."""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from ca_stochastic import run_single_cell_stochastic, compute_ensemble_analytics
from simulator import simulate
from ca_schema import discretize_state
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

def main() -> None:
    # Load tuned rules
    with open('tuned_rules.json', 'r') as f:
        rules = json.load(f)

    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)

    # Run stochastic ensemble
    n_trials = 100
    print(f"Running {n_trials} stochastic trials...")
    ensemble = run_single_cell_stochastic(
        patient=patient,
        intervention=intervention,
        n_trials=n_trials,
        rules=rules,
        seed=42,
    )

    # Compute ensemble analytics
    analytics = compute_ensemble_analytics(ensemble)
    print("\nAttractor probabilities:")
    for attractor, prob in analytics['attractor_probabilities'].items():
        print(f"  {attractor}: {prob:.3f}")

    print("\nTerminal bin distributions:")
    for var, dist in analytics['variable_distributions'].items():
        print(f"  {var}: {dist}")

    # ODE trajectory
    ode = simulate(patient=patient, intervention=intervention)
    ode_bins = [discretize_state(s) for s in ode['states'][::int(0.25/0.01)]]  # sample quarterly
    ode_npoint = [b['N_point'] for b in ode_bins]
    print(f"\nODE N_point trajectory bins (quarterly): {ode_npoint[:10]} ...")

    # For each trial, compute bin trajectory
    trials_npoint = []
    for trial in ensemble['trajectories']:
        trials_npoint.append([s['N_point'] for s in trial])

    # Compute agreement per step
    total_steps = len(trials_npoint[0])
    agreement = []
    for step in range(total_steps):
        if step >= len(ode_npoint):
            break
        matches = sum(1 for t in trials_npoint if t[step] == ode_npoint[step])
        agreement.append(matches / n_trials)
    print(f"\nAverage per-step agreement (N_point): {np.mean(agreement):.3f}")

    # Show first 10 steps agreement
    print("First 10 steps agreement:")
    for step in range(min(10, total_steps)):
        print(f"  Step {step}: {agreement[step]:.3f}")

    # Overall fidelity (deterministic) for reference
    from ca_simulator import run_single_cell
    ca_det = run_single_cell(patient=patient, intervention=intervention, sim_years=30.0, dt=0.25)
    from ca_analytics import _fidelity_stats
    fidelity = _fidelity_stats(ca_det['trajectory'], ode, patient)
    print(f"\nDeterministic CA overall agreement: {fidelity['overall_agreement']:.3f}")
    print(f"Deterministic CA N_point agreement: {fidelity['per_variable_agreement']['N_point']:.3f}")


if __name__ == "__main__":
    main()
