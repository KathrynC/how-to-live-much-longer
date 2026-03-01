#!/usr/bin/env python3
"""Add N_point stabilization rule and adjust confidences."""

import json

with open('final_tuned_rules.json', 'r') as f:
    rules = json.load(f)

# Remove existing N_point_suppression rule (or reduce confidence)
for r in rules:
    if r['name'] == 'N_point_suppression':
        r['confidence'] = 0.1  # reduce
        print(f"Reduced N_point_suppression confidence to {r['confidence']}")
    elif r['name'] == 'ros_drives_points':
        r['confidence'] = 0.15
        print(f"Reduced ros_drives_points confidence to {r['confidence']}")
    elif r['name'] == 'point_mutation_pol_gamma_errors':
        r['confidence'] = 0.15
        print(f"Reduced point_mutation_pol_gamma_errors confidence to {r['confidence']}")

# Add stabilization rule for moderate N_point
new_rule = {
    "tier": 5,
    "name": "N_point_stabilize_moderate",
    "inputs": {
        "N_point": "moderate"
    },
    "context": {},
    "outputs": {
        "N_point": "0"
    },
    "confidence": 0.9,
    "citation": "Stabilize point mutations at moderate level (ODE observation)"
}
rules.append(new_rule)
print("Added N_point_stabilize_moderate rule")

# Add slow accumulation rule that fires rarely
new_rule2 = {
    "tier": 2,
    "name": "N_point_slow_accumulation",
    "inputs": {
        "N_healthy": "adequate",
        "age_epoch": ["transition", "old"]
    },
    "context": {},
    "outputs": {
        "N_point": "+1"
    },
    "confidence": 0.05,
    "citation": "Slow linear accumulation of point mutations with age"
}
rules.append(new_rule2)
print("Added N_point_slow_accumulation rule")

# Save new rules
with open('final_tuned_rules_n_point_v2.json', 'w') as f:
    json.dump(rules, f, indent=2)
print("Saved to final_tuned_rules_n_point_v2.json")

# Quick test
from simulator import simulate, initial_state
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_simulator import _build_context, step_cell

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)
ode_result = simulate(patient=patient, intervention=intervention)
cont = initial_state(patient)
state = discretize_state(cont)
prev_state = None
n_steps = 20  # first 5 years

print("\nTesting first 5 years...")
for step in range(n_steps):
    year = step * 0.25
    context = _build_context(step, patient, intervention, prev_state, state)
    new_state, fired = step_cell(state, context, rules)
    if state['N_point'] != new_state['N_point']:
        print(f"Step {step} ({year:.2f} yr): N_point {state['N_point']} -> {new_state['N_point']}")
        print(f"  Fired rules affecting N_point:", [r['name'] for r in fired if 'N_point' in r.get('outputs', {})])
    prev_state = state
    state = new_state

print(f"\nFinal CA N_point after 5 years: {state['N_point']}")
ode_idx = min(int(round(5.0 / 0.01)), len(ode_result['states'])-1)
ode_bin = discretize_state(ode_result['states'][ode_idx])['N_point']
print(f"ODE N_point bin at 5 years: {ode_bin}")