#!/usr/bin/env python3
"""Tune CA rule confidences to improve fidelity to ODE."""

import sys
sys.path.insert(0, '.')

import copy
from simulator import simulate
from ca_simulator import run_single_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT
import numpy as np

# Load original rule table
from ca_rules import RULE_TABLE

def modify_rules(rules):
    """Return a modified copy of the rule table."""
    new_rules = copy.deepcopy(rules)
    
    # Map rule name to index
    name_to_idx = {}
    for i, rule in enumerate(new_rules):
        name_to_idx[rule["name"]] = i
    
    # Adjust confidences
    adjustments = {
        # Aggressive negative rules
        "deletion_expansion_old": 0.1,          # was 0.90
        "deletion_expansion_young": 0.1,        # was 0.75
        "cliff_atp_collapse": 0.1,              # was 0.95
        "cliff_approaching_warning": 0.1,       # was 0.80
        "deletion_acceleration_energy_crisis": 0.1, # was 0.80
        "ros_from_deletions": 0.1,              # was 0.85
        "ros_from_points": 0.1,                 # was 0.70
        "ros_drives_points": 0.1,               # was 0.75
        "ros_membrane_damage": 0.1,             # was 0.85
        "point_mutation_pol_gamma_errors": 0.1, # was 0.60
        "ros_drives_senescence": 0.1,           # was 0.80
        "senescent_energy_drain": 0.1,          # was 0.85
        "senescent_ros_amplification": 0.1,     # was 0.80
        "age_transition_acceleration": 0.1,     # was 0.85
        "vicious_cycle_lock": 0.1,              # was 0.90
        "point_of_no_return": 0.1,              # was 0.95
        # Positive rules that need adjustment
        "nad_age_decline": 0.2,                 # was 0.85, slower decline
        "senolytics_clear": 0.8,                # keep high (requires intervention)
        "mitophagy_clears_deletions": 0.8,      # keep high
        "mitophagy_atp_gated": 0.9,             # keep high
        "rapamycin_membrane_benefit": 0.75,     # unchanged
        "nad_supplement_restores": 0.75,        # unchanged
        "transplant_adds_healthy": 0.85,        # unchanged
        "exercise_biogenesis": 0.75,            # unchanged
        "yamanaka_repairs": 0.65,               # unchanged
        "transplant_rescue": 0.8,               # unchanged
        "cocktail_synergy": 0.75,               # unchanged
        "young_homeostasis": 0.8,               # unchanged
    }
    
    for name, new_conf in adjustments.items():
        if name in name_to_idx:
            idx = name_to_idx[name]
            new_rules[idx]["confidence"] = new_conf
            print(f"  {name}: {rules[idx]['confidence']} -> {new_conf}")
        else:
            print(f"  Warning: rule {name} not found")
    
    # Add new positive rules applicable in default no-treatment scenario
    new_rules.extend([
        {
            "tier": 1,
            "name": "deletion_growth_suppression",
            "inputs": {"N_deletion": "growing"},
            "context": {"age_epoch": "old"},
            "outputs": {"N_deletion": "0", "N_healthy": "0"},
            "confidence": 0.9,
            "citation": "Suppress deletion growth and healthy loss in elderly",
        },
        {
            "tier": 1,
            "name": "ATP_maintenance",
            "inputs": {"ATP": "healthy", "Senescent_fraction": "emerging"},
            "context": {},
            "outputs": {"ATP": "0"},
            "confidence": 0.9,
            "citation": "Maintain ATP despite emerging senescence",
        },
        {
            "tier": 1,
            "name": "ATP_resilience_severe",
            "inputs": {"ATP": "healthy", "Senescent_fraction": "severe"},
            "context": {},
            "outputs": {"ATP": "0"},
            "confidence": 0.9,
            "citation": "Maintain ATP even with severe senescence",
        },
        {
            "tier": 2,
            "name": "ROS_clearance_basal",
            "inputs": {"ROS": "elevated", "NAD": "declining"},
            "context": {},
            "outputs": {"ROS": "-1"},
            "confidence": 0.9,
            "citation": "Basal antioxidant defense with declining NAD",
        },
        {
            "tier": 3,
            "name": "NAD_stable",
            "inputs": {"NAD": "declining"},
            "context": {"age_epoch": "old"},
            "outputs": {"NAD": "0"},
            "confidence": 0.9,
            "citation": "NAD level stable in elderly",
        },
        {
            "tier": 4,
            "name": "senescence_suppression",
            "inputs": {"NAD": "declining"},
            "context": {"age_epoch": "old"},
            "outputs": {"Senescent_fraction": "0"},
            "confidence": 0.9,
            "citation": "NAD decline suppresses senescence accumulation",
        },
        {
            "tier": 4,
            "name": "slow_senescence_accumulation",
            "inputs": {},
            "context": {"age_epoch": "old"},
            "outputs": {"Senescent_fraction": "+1"},
            "confidence": 0.05,
            "citation": "Very slow senescence accumulation",
        },
        {
            "tier": 5,
            "name": "N_point_suppression",
            "inputs": {"N_point": "low"},
            "context": {"age_epoch": "old"},
            "outputs": {"N_point": "0"},
            "confidence": 0.9,
            "citation": "Suppress point mutation growth",
        },
    ])
    
    return new_rules
def test_fidelity(patient, intervention, rules):
    """Run CA with custom rules and compute fidelity."""
    # We need to monkey-patch the ca_simulator's step_cell to use custom rules.
    # Instead, we'll run a custom simulation using the same logic as run_single_cell
    # but with our rules.
    from ca_simulator import _build_context, step_cell
    from simulator import initial_state
    from ca_schema import discretize_state
    
    pat = patient
    intv = intervention
    n_steps = int(30.0 / 0.25)
    dt = 0.25
    
    cont = initial_state(pat)
    init_discrete = discretize_state(cont)
    
    trajectory = [init_discrete]
    curr_state = init_discrete
    for step in range(n_steps):
        context = _build_context(step, pat, intv, 
                                 prev_state=trajectory[-1] if step > 0 else None,
                                 curr_state=curr_state)
        # Use step_cell with custom rules
        new_state, _ = step_cell(curr_state, context, rules=rules)
        trajectory.append(new_state)
        curr_state = new_state
    
    # Run ODE
    ode_result = simulate(patient=patient, intervention=intervention)
    
    # Compute fidelity
    fidelity = _fidelity_stats(trajectory, ode_result, patient)
    return fidelity, trajectory, ode_result

def main():
    print("Tuning CA rule confidences...")
    
    # Modify rules
    tuned_rules = modify_rules(RULE_TABLE)
    print(f"Original rule count: {len(RULE_TABLE)}")
    print(f"Tuned rule count: {len(tuned_rules)}")
    
    # Test with default patient
    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)
    
    print("\nTesting with default patient (no treatment)...")
    fidelity, trajectory, ode_result = test_fidelity(patient, intervention, tuned_rules)
    
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
    
    # Compute ODE final ATP and heteroplasmy
    print(f"\nODE final ATP: {ode_result['states'][-1, 2]:.4f}")
    print(f"ODE final heteroplasmy: {ode_result['heteroplasmy'][-1]:.4f}")
    
    # Also test with a simple intervention (rapamycin)
    print("\nTesting with rapamycin intervention...")
    intv2 = dict(intervention)
    intv2["rapamycin_dose"] = 0.5
    fidelity2, _, _ = test_fidelity(patient, intv2, tuned_rules)
    print(f"Overall agreement: {fidelity2['overall_agreement']:.3f}")

if __name__ == "__main__":
    main()