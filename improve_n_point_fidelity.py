#!/usr/bin/env python3
"""Improve N_point fidelity by adjusting rule confidences."""

import sys
sys.path.insert(0, '.')
import json
import copy
from simulator import simulate
from ca_simulator import run_single_cell, _build_context, step_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

def load_rules(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_rules(rules, path):
    with open(path, 'w') as f:
        json.dump(rules, f, indent=2)

def modify_n_point_rules(rules):
    """Adjust confidences for N_point-related rules."""
    new_rules = copy.deepcopy(rules)
    name_to_idx = {r['name']: i for i, r in enumerate(new_rules)}
    
    # Adjustments
    adjustments = {
        # Increase growth rule confidences
        'ros_drives_points': 0.3,
        'point_mutation_pol_gamma_errors': 0.3,
        # Reduce suppression confidence
        'N_point_suppression': 0.4,
        # Adjust mitophagy weak on points (if rapamycin present)
        'mitophagy_weak_on_points': 0.3,
        # Yamanaka repairs (if yamanaka high)
        'yamanaka_repairs': 0.5,
    }
    
    for name, new_conf in adjustments.items():
        if name in name_to_idx:
            idx = name_to_idx[name]
            old = new_rules[idx]['confidence']
            new_rules[idx]['confidence'] = new_conf
            print(f"  {name}: {old} -> {new_conf}")
        else:
            print(f"  Warning: rule {name} not found")
    
    return new_rules

def test_fidelity(patient, intervention, rules):
    """Run CA with custom rules and compute fidelity."""
    from simulator import initial_state
    n_steps = int(30.0 / 0.25)
    dt = 0.25
    
    cont = initial_state(patient)
    init_discrete = discretize_state(cont)
    
    trajectory = [init_discrete]
    curr_state = init_discrete
    for step in range(n_steps):
        context = _build_context(step, patient, intervention,
                                 prev_state=trajectory[-1] if step > 0 else None,
                                 curr_state=curr_state)
        new_state, _ = step_cell(curr_state, context, rules=rules)
        trajectory.append(new_state)
        curr_state = new_state
    
    # Run ODE
    ode_result = simulate(patient=patient, intervention=intervention)
    
    # Compute fidelity
    fidelity = _fidelity_stats(trajectory, ode_result, patient)
    return fidelity, trajectory, ode_result

def main():
    print("Loading tuned_rules.json...")
    rules = load_rules('tuned_rules.json')
    print(f"Loaded {len(rules)} rules")
    
    # Modify
    print("\nAdjusting N_point rule confidences...")
    new_rules = modify_n_point_rules(rules)
    
    # Save
    out_path = 'tuned_rules_n_point.json'
    save_rules(new_rules, out_path)
    print(f"Saved to {out_path}")
    
    # Test with default patient
    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)
    
    print("\nTesting with default patient (no treatment)...")
    fidelity, trajectory, ode_result = test_fidelity(patient, intervention, new_rules)
    
    print(f"Overall agreement: {fidelity['overall_agreement']:.3f}")
    print("Per-variable agreement:")
    for var, agree in fidelity['per_variable_agreement'].items():
        print(f"  {var}: {agree:.3f}")
    
    # Compare final states
    ca_final = trajectory[-1]
    ode_final_discrete = discretize_state(ode_result["states"][-1])
    print("\nFinal state comparison:")
    for var in ca_final:
        ca_val = ca_final[var]
        ode_val = ode_final_discrete.get(var)
        match = "✓" if ca_val == ode_val else "✗"
        print(f"  {var}: CA {ca_val} vs ODE {ode_val} {match}")
    
    # ODE final N_point
    print(f"\nODE final N_point: {ode_result['states'][-1, 7]:.4f}")
    print(f"ODE final N_point bin: {ode_final_discrete['N_point']}")
    
    # CA trajectory N_point bins
    ca_npoint = [s['N_point'] for s in trajectory]
    print(f"CA N_point trajectory bins: {ca_npoint[:5]} ... {ca_npoint[-5:]}")
    
    return fidelity

if __name__ == "__main__":
    main()