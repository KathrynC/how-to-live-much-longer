#!/usr/bin/env python3
"""Fix N_point discrepancy by adjusting rule confidences."""

import json
import copy
from simulator import simulate, initial_state
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_simulator import _build_context, step_cell
from ca_rules import get_applicable_rules
from ca_analytics import _fidelity_stats

def run_ca_with_rules(patient, intervention, rules, sim_years=30.0, dt=0.25):
    """Run CA with custom rules."""
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}
    n_steps = int(sim_years / dt)
    cont = initial_state(pat)
    state = discretize_state(cont)
    trajectory = [dict(state)]
    prev_state = None
    for step in range(n_steps):
        ctx = _build_context(step, pat, intv, prev_state, state)
        new_state, _ = step_cell(state, ctx, rules)
        trajectory.append(dict(new_state))
        prev_state = state
        state = new_state
    return trajectory

def evaluate(rules, patient=None, intervention=None):
    if patient is None:
        patient = dict(DEFAULT_PATIENT)
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    
    ode_result = simulate(patient=patient, intervention=intervention)
    ca_traj = run_ca_with_rules(patient, intervention, rules)
    fidelity = _fidelity_stats(ca_traj, ode_result, patient)
    
    print(f"Overall fidelity: {fidelity['overall_agreement']:.3f}")
    print("Per-variable agreement:")
    for var, agree in fidelity['per_variable_agreement'].items():
        print(f"  {var}: {agree:.3f}")
    
    # N_point specific
    ode_states = ode_result['states']
    n_point_matches = 0
    for i in range(len(ca_traj)):
        ode_bin = discretize_state(ode_states[i])['N_point']
        if ca_traj[i]['N_point'] == ode_bin:
            n_point_matches += 1
    print(f"N_point matches: {n_point_matches}/{len(ca_traj)} ({n_point_matches/len(ca_traj):.3f})")
    
    # When does CA transition?
    for i in range(len(ca_traj)):
        if ca_traj[i]['N_point'] == 'moderate':
            print(f"CA transitions to moderate at step {i} (year {i*0.25:.2f})")
            break
    else:
        print("CA never transitions to moderate")
    
    # When does ODE transition?
    for i in range(len(ca_traj)):
        ode_bin = discretize_state(ode_states[i])['N_point']
        if ode_bin == 'moderate':
            print(f"ODE transitions to moderate at step {i} (year {i*0.25:.2f})")
            break
    else:
        print("ODE never transitions to moderate")
    
    return fidelity

def main():
    with open('final_tuned_rules.json', 'r') as f:
        rules = json.load(f)
    
    # Identify N_point rules
    n_point_rules = []
    for i, r in enumerate(rules):
        outputs = r.get('outputs', {})
        if 'N_point' in outputs:
            n_point_rules.append((i, r['name'], r['confidence'], outputs['N_point']))
    print("Current N_point rules:")
    for idx, name, conf, out in n_point_rules:
        print(f"  {idx}: {name} (conf {conf}) -> {out}")
    
    # Try adjustments
    adjustments = [
        ('reduce suppression', {'N_point_suppression': 0.2, 'ros_drives_points': 0.5, 'point_mutation_pol_gamma_errors': 0.5}),
        ('remove suppression', {'N_point_suppression': 0.0, 'ros_drives_points': 0.5, 'point_mutation_pol_gamma_errors': 0.5}),
        ('boost growth only', {'ros_drives_points': 0.7, 'point_mutation_pol_gamma_errors': 0.7}),
        ('balanced', {'N_point_suppression': 0.3, 'ros_drives_points': 0.6, 'point_mutation_pol_gamma_errors': 0.6, 'mitophagy_weak_on_points': 0.3}),
    ]
    
    best_fid = 0
    best_name = None
    best_rules = None
    
    for name, adj in adjustments:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        test_rules = copy.deepcopy(rules)
        name_to_idx = {r['name']: i for i, r in enumerate(test_rules)}
        for rule_name, new_conf in adj.items():
            if rule_name in name_to_idx:
                test_rules[name_to_idx[rule_name]]['confidence'] = new_conf
                print(f"  {rule_name} -> {new_conf}")
            else:
                print(f"  Warning: {rule_name} not found")
        fid = evaluate(test_rules)
        if fid['overall_agreement'] > best_fid:
            best_fid = fid['overall_agreement']
            best_name = name
            best_rules = test_rules
    
    print(f"\n{'='*60}")
    print(f"Best: {best_name} (fidelity {best_fid:.3f})")
    
    # Save best rules
    if best_rules:
        with open('final_tuned_rules_n_point_fixed.json', 'w') as f:
            json.dump(best_rules, f, indent=2)
        print("Saved to final_tuned_rules_n_point_fixed.json")
        
        # Also update final_tuned_rules.json? Let's ask user
        print("\nTo apply these changes, run:")
        print("  cp final_tuned_rules_n_point_fixed.json final_tuned_rules.json")

if __name__ == '__main__':
    main()