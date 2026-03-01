#!/usr/bin/env python3
"""Diagnose N_point rule firing with custom rules."""

import sys
sys.path.insert(0, '.')

import json
from simulator import simulate
from ca_simulator import _build_context, step_cell
from ca_schema import discretize_state, BIN_SCHEMA
from ca_rules import get_applicable_rules
import numpy as np

# Use default patient and no intervention
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)

# Load tuned rules and modify
with open('tuned_rules.json', 'r') as f:
    rules = json.load(f)

# Modify N_point rules
name_to_idx = {r['name']: i for i, r in enumerate(rules)}
adjustments = {
    'ros_drives_points': 0.7,
    'point_mutation_pol_gamma_errors': 0.7,
    'N_point_suppression': 0.2,
    'mitophagy_weak_on_points': 0.3,
    'yamanaka_repairs': 0.5,
}
for name, new_conf in adjustments.items():
    if name in name_to_idx:
        rules[name_to_idx[name]]['confidence'] = new_conf
        print(f"Adjusted {name} -> {new_conf}")
    else:
        print(f"Warning: {name} not found")

print("\nRunning diagnosis...")

# Run ODE
ode_result = simulate(patient=patient, intervention=intervention)

# Run CA with custom rules
pat = patient
intv = intervention
n_steps = int(30.0 / 0.25)
dt = 0.25

from simulator import initial_state
cont = initial_state(pat)
init_discrete = discretize_state(cont)

trajectory = [init_discrete]
rule_log = []

curr_state = init_discrete
for step in range(n_steps):
    context = _build_context(step, pat, intv,
                             prev_state=trajectory[-1] if step > 0 else None,
                             curr_state=curr_state)
    
    applicable = get_applicable_rules(curr_state, context, rules)
    
    new_state, fired = step_cell(curr_state, context, rules)
    
    rule_log.append([r["name"] for r in fired])
    trajectory.append(new_state)
    
    # Track N_point changes
    if step < 10 or step % 20 == 0 or curr_state['N_point'] != trajectory[step]['N_point']:
        print(f"\nStep {step} (t={step*dt:.2f} yr):")
        print(f"  N_point: {trajectory[step]['N_point']} -> {curr_state['N_point']}")
        # Which rules affect N_point?
        n_point_rules = []
        for r in applicable:
            if 'N_point' in r.get('outputs', {}):
                n_point_rules.append((r['name'], r['confidence'], r['outputs']['N_point']))
        if n_point_rules:
            print(f"  N_point applicable rules: {n_point_rules}")
        else:
            print(f"  No N_point rules applicable")
        # ODE comparison
        ode_idx = min(int(round(step*dt / 0.01)), len(ode_result["states"])-1)
        ode_val = ode_result["states"][ode_idx, 7]
        ode_bin = discretize_state(ode_result["states"][ode_idx])['N_point']
        print(f"  ODE N_point: {ode_val:.4f} ({ode_bin})")
    
    curr_state = new_state

# Compute fidelity
from ca_analytics import _fidelity_stats
fidelity = _fidelity_stats(trajectory, ode_result, patient)
print(f"\nOverall fidelity: {fidelity['overall_agreement']:.3f}")
print("Per-variable agreement:")
for var, agree in fidelity['per_variable_agreement'].items():
    print(f"  {var}: {agree:.3f}")

# Final state comparison
ca_final = trajectory[-1]
ode_final_discrete = discretize_state(ode_result["states"][-1])
print("\nFinal state comparison:")
for var in ca_final:
    ca_val = ca_final[var]
    ode_val = ode_final_discrete.get(var)
    match = "✓" if ca_val == ode_val else "✗"
    print(f"  {var}: CA {ca_val} vs ODE {ode_val} {match}")