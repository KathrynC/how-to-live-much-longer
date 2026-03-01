#!/usr/bin/env python3
"""Adjust N_point rule confidences and test fidelity."""

import json
import copy
from simulator import simulate
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_simulator import run_single_cell
from ca_analytics import compute_ca_analytics

def load_rules():
    with open('final_tuned_rules.json', 'r') as f:
        return json.load(f)

def save_rules(rules, filename):
    with open(filename, 'w') as f:
        json.dump(rules, f, indent=2)
    print(f"Saved to {filename}")

def adjust_rules(rules, adjustments):
    """Adjust rule confidences in-place."""
    name_to_idx = {r['name']: i for i, r in enumerate(rules)}
    for name, new_conf in adjustments.items():
        if name in name_to_idx:
            rules[name_to_idx[name]]['confidence'] = new_conf
            print(f"Adjusted {name} -> {new_conf}")
        else:
            print(f"Warning: {name} not found")
    return rules

def evaluate_rules(rules, patient=None, intervention=None):
    if patient is None:
        patient = dict(DEFAULT_PATIENT)
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    
    ode_result = simulate(patient=patient, intervention=intervention)
    ca_result = run_single_cell(patient, intervention, rules=rules)
    analytics = compute_ca_analytics(ca_result, ode_result, patient)
    
    print(f"Overall fidelity: {analytics['fidelity_stats']['overall_agreement']:.3f}")
    print("Per-variable agreement:")
    for var, agree in analytics['fidelity_stats']['per_variable_agreement'].items():
        print(f"  {var}: {agree:.3f}")
    
    # Print N_point trajectory
    ca_traj = ca_result['trajectory']
    ode_traj = discretize_state(ode_result['states'])
    print("\nN_point bin trajectory (CA vs ODE):")
    for i in range(0, len(ca_traj), 20):
        ca_bin = ca_traj[i]['N_point']
        ode_bin = ode_traj[i]['N_point']
        match = "✓" if ca_bin == ode_bin else "✗"
        print(f"  Year {i*0.25:5.2f}: CA {ca_bin} vs ODE {ode_bin} {match}")
    
    return analytics

if __name__ == '__main__':
    rules = load_rules()
    
    # Different adjustment strategies
    strategies = [
        {
            'name': 'reduce_suppression',
            'adjustments': {
                'N_point_suppression': 0.2,
                'ros_drives_points': 0.5,
                'point_mutation_pol_gamma_errors': 0.5,
            }
        },
        {
            'name': 'remove_suppression',
            'adjustments': {
                'N_point_suppression': 0.0,  # effectively disables
                'ros_drives_points': 0.5,
                'point_mutation_pol_gamma_errors': 0.5,
            }
        },
        {
            'name': 'boost_growth_only',
            'adjustments': {
                'ros_drives_points': 0.7,
                'point_mutation_pol_gamma_errors': 0.7,
                # suppression stays 0.9
            }
        },
        {
            'name': 'balanced',
            'adjustments': {
                'N_point_suppression': 0.3,
                'ros_drives_points': 0.6,
                'point_mutation_pol_gamma_errors': 0.6,
                'mitophagy_weak_on_points': 0.3,
            }
        },
    ]
    
    best_fidelity = 0
    best_strategy = None
    best_rules = None
    
    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strat['name']}")
        print(f"{'='*60}")
        test_rules = copy.deepcopy(rules)
        adjust_rules(test_rules, strat['adjustments'])
        analytics = evaluate_rules(test_rules)
        fidelity = analytics['fidelity_stats']['overall_agreement']
        n_point_agree = analytics['fidelity_stats']['per_variable_agreement']['N_point']
        
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_strategy = strat['name']
            best_rules = test_rules
    
    print(f"\n{'='*60}")
    print(f"Best strategy: {best_strategy} (fidelity {best_fidelity:.3f})")
    
    # Save best rules
    if best_rules:
        save_rules(best_rules, 'final_tuned_rules_n_point_fixed.json')
        print("\nAlso updating final_tuned_rules.json? (uncomment to apply)")
        # save_rules(best_rules, 'final_tuned_rules.json')