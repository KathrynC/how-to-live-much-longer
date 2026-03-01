#!/usr/bin/env python3
"""Grid search over rule confidences to maximize fidelity."""

import sys
sys.path.insert(0, '.')
import json
import copy
import itertools
from simulator import simulate
from ca_simulator import _build_context, step_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

def load_rules(path):
    with open(path, 'r') as f:
        return json.load(f)

def modify_confidences(rules, adjustments):
    """Return new rules with confidences updated."""
    new_rules = copy.deepcopy(rules)
    name_to_idx = {r['name']: i for i, r in enumerate(new_rules)}
    for name, new_conf in adjustments.items():
        if name in name_to_idx:
            new_rules[name_to_idx[name]]['confidence'] = new_conf
        else:
            print(f"Warning: rule {name} not found")
    return new_rules

def evaluate_fidelity(rules):
    """Run CA with given rules and return overall agreement."""
    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)
    n_steps = int(30.0 / 0.25)
    from simulator import initial_state
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
    return fidelity['overall_agreement'], fidelity['per_variable_agreement']

def main():
    print("Loading tuned_rules.json...")
    base_rules = load_rules('tuned_rules.json')
    
    # Grid parameters
    ros_prod_conf = [0.1, 0.3, 0.5, 0.7]
    ros_clear_conf = [0.1, 0.3, 0.5, 0.7, 0.9]
    npoint_growth_conf = [0.1, 0.3, 0.5, 0.7]
    npoint_supp_conf = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    best_score = 0.0
    best_params = None
    best_rules = None
    
    total_combos = len(ros_prod_conf) * len(ros_clear_conf) * len(npoint_growth_conf) * len(npoint_supp_conf)
    print(f"Total combinations: {total_combos}")
    count = 0
    
    for rp, rc, ng, ns in itertools.product(ros_prod_conf, ros_clear_conf, npoint_growth_conf, npoint_supp_conf):
        count += 1
        if count % 10 == 0:
            print(f"  {count}/{total_combos}")
        adjustments = {
            'ros_from_deletions': rp,
            'ros_from_points': rp,
            'ros_membrane_damage': rp,
            'ros_drives_senescence': rp,
            'senescent_ros_amplification': rp,
            'ROS_clearance_basal': rc,
            'ros_drives_points': ng,
            'point_mutation_pol_gamma_errors': ng,
            'N_point_suppression': ns,
        }
        rules = modify_confidences(base_rules, adjustments)
        score, per_var = evaluate_fidelity(rules)
        if score > best_score:
            best_score = score
            best_params = (rp, rc, ng, ns)
            best_rules = rules
            print(f"New best: {score:.3f} with prod={rp}, clear={rc}, growth={ng}, supp={ns}")
            print(f"  Per-variable: {per_var}")
    
    print(f"\nBest overall agreement: {best_score:.3f}")
    print(f"Best parameters: prod={best_params[0]}, clear={best_params[1]}, growth={best_params[2]}, supp={best_params[3]}")
    
    # Save best rules
    out_path = 'tuned_rules_best.json'
    with open(out_path, 'w') as f:
        json.dump(best_rules, f, indent=2)
    print(f"Saved to {out_path}")
    
    # Final evaluation
    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)
    final_score, final_per_var = evaluate_fidelity(best_rules)
    print(f"\nFinal validation:")
    print(f"Overall agreement: {final_score:.3f}")
    for var, agree in final_per_var.items():
        print(f"  {var}: {agree:.3f}")
    
    # Compare with ODE final state
    ode_result = simulate(patient=patient, intervention=intervention)
    # Run CA trajectory
    n_steps = int(30.0 / 0.25)
    from simulator import initial_state
    cont = initial_state(patient)
    init_discrete = discretize_state(cont)
    trajectory = [init_discrete]
    curr_state = init_discrete
    for step in range(n_steps):
        context = _build_context(step, patient, intervention,
                                 prev_state=trajectory[-1] if step > 0 else None,
                                 curr_state=curr_state)
        new_state, _ = step_cell(curr_state, context, rules=best_rules)
        trajectory.append(new_state)
        curr_state = new_state
    ca_final = trajectory[-1]
    ode_final_discrete = discretize_state(ode_result["states"][-1])
    print("\nFinal state comparison:")
    for var in ca_final:
        ca_val = ca_final[var]
        ode_val = ode_final_discrete.get(var)
        match = "✓" if ca_val == ode_val else "✗"
        print(f"  {var}: CA {ca_val} vs ODE {ode_val} {match}")

if __name__ == "__main__":
    main()