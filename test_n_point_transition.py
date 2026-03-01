#!/usr/bin/env python3
"""Test N_point transition with adjusted rules."""

import json
from simulator import simulate, initial_state
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_simulator import _build_context, step_cell
from ca_rules import get_applicable_rules

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)

with open('final_tuned_rules.json', 'r') as f:
    rules = json.load(f)

# Adjust
for r in rules:
    if r['name'] == 'N_point_suppression':
        r['confidence'] = 0.2
    elif r['name'] == 'ros_drives_points':
        r['confidence'] = 0.5
    elif r['name'] == 'point_mutation_pol_gamma_errors':
        r['confidence'] = 0.5

ode_result = simulate(patient=patient, intervention=intervention)
cont = initial_state(patient)
state = discretize_state(cont)
n_steps = int(30.0 / 0.25)
prev_state = None

print("Initial N_point:", state['N_point'])
print("ODE final N_point:", ode_result['states'][-1, 7])

for step in range(n_steps):
    year = step * 0.25
    context = _build_context(step, patient, intervention, prev_state, state)
    applicable = get_applicable_rules(state, context, rules)
    
    # Find N_point rules
    n_point_rules = []
    for r in applicable:
        if 'N_point' in r.get('outputs', {}):
            n_point_rules.append((r['name'], r['confidence'], r['outputs']['N_point']))
    
    new_state, fired = step_cell(state, context, rules)
    
    if state['N_point'] != new_state['N_point']:
        print(f"\nStep {step} ({year:.2f} yr): N_point {state['N_point']} -> {new_state['N_point']}")
        print(f"  Fired rules: {[r['name'] for r in fired]}")
        if n_point_rules:
            print(f"  N_point applicable: {n_point_rules}")
    
    # Compare with ODE
    if step % 20 == 0:
        ode_idx = min(int(round(year / 0.01)), len(ode_result['states'])-1)
        ode_val = ode_result['states'][ode_idx, 7]
        ode_bin = discretize_state(ode_result['states'][ode_idx])['N_point']
        if state['N_point'] != ode_bin:
            print(f"  MISMATCH: CA {state['N_point']} vs ODE {ode_bin} ({ode_val:.4f})")
    
    prev_state = state
    state = new_state

print("\nFinal CA N_point:", state['N_point'])
ode_final_bin = discretize_state(ode_result['states'][-1])['N_point']
print("Final ODE N_point bin:", ode_final_bin)