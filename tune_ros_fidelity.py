#!/usr/bin/env python3
"""Tune ROS rule confidences to improve fidelity."""

import sys
sys.path.insert(0, '.')
import json
import copy
from simulator import simulate
from ca_simulator import _build_context, step_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

def load_rules(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_rules(rules, path):
    with open(path, 'w') as f:
        json.dump(rules, f, indent=2)

def modify_ros_rules(rules):
    """Adjust confidences for ROS-related rules."""
    new_rules = copy.deepcopy(rules)
    name_to_idx = {r['name']: i for i, r in enumerate(new_rules)}
    
    adjustments = {
        # Increase ROS production confidences
        'ros_from_deletions': 0.5,
        'ros_from_points': 0.5,
        'ros_membrane_damage': 0.5,
        'ros_drives_senescence': 0.5,
        'senescent_ros_amplification': 0.5,
        # Reduce ROS clearance confidence
        'ROS_clearance_basal': 0.5,
        # Keep NAD boost defense
        'nad_boosts_defense': 0.7,
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
    
    ode_result = simulate(patient=patient, intervention=intervention)
    fidelity = _fidelity_stats(trajectory, ode_result, patient)
    return fidelity, trajectory, ode_result

def main():
    print("Loading tuned_rules.json...")
    rules = load_rules('tuned_rules.json')
    print(f"Loaded {len(rules)} rules")
    
    print("\nAdjusting ROS rule confidences...")
    new_rules = modify_ros_rules(rules)
    
    out_path = 'tuned_rules_ros.json'
    save_rules(new_rules, out_path)
    print(f"Saved to {out_path}")
    
    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)
    
    print("\nTesting with default patient (no treatment)...")
    fidelity, trajectory, ode_result = test_fidelity(patient, intervention, new_rules)
    
    print(f"Overall agreement: {fidelity['overall_agreement']:.3f}")
    print("Per-variable agreement:")
    for var, agree in fidelity['per_variable_agreement'].items():
        print(f"  {var}: {agree:.3f}")
    
    ca_final = trajectory[-1]
    ode_final_discrete = discretize_state(ode_result["states"][-1])
    print("\nFinal state comparison:")
    for var in ca_final:
        ca_val = ca_final[var]
        ode_val = ode_final_discrete.get(var)
        match = "✓" if ca_val == ode_val else "✗"
        print(f"  {var}: CA {ca_val} vs ODE {ode_val} {match}")
    
    # Compute ODE ROS trajectory bins
    ode_ros = [discretize_state(s)['ROS'] for s in ode_result['states'][::int(0.25/0.01)]]
    ca_ros = [s['ROS'] for s in trajectory]
    matches = sum(1 for i in range(len(ca_ros)) if ca_ros[i] == ode_ros[i])
    print(f"\nROS bin matches: {matches}/{len(ca_ros)} ({matches/len(ca_ros):.3f})")
    
    return fidelity

if __name__ == "__main__":
    main()