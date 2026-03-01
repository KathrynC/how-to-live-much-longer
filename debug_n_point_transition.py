#!/usr/bin/env python3
"""Debug why N_point doesn't transition from low to moderate."""

import sys
sys.path.insert(0, '.')

import json
from simulator import simulate, initial_state
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state, BIN_SCHEMA
from ca_simulator import _build_context, step_cell
from ca_rules import get_applicable_rules

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)

# Load final tuned rules
with open('final_tuned_rules.json', 'r') as f:
    rules = json.load(f)

# Run ODE
ode_result = simulate(patient=patient, intervention=intervention)

# Run CA
cont = initial_state(patient)
init_discrete = discretize_state(cont)
n_steps = int(30.0 / 0.25)  # quarterly steps
dt = 0.25

trajectory = [init_discrete]
curr_state = init_discrete

print("Initial N_point:", cont[7], "bin:", init_discrete['N_point'])
print("ODE final N_point:", ode_result['states'][-1, 7])
print()

for step in range(n_steps):
    context = _build_context(step, patient, intervention,
                             prev_state=trajectory[-1] if step > 0 else None,
                             curr_state=curr_state)
    
    applicable = get_applicable_rules(curr_state, context, rules)
    
    new_state, fired = step_cell(curr_state, context, rules)
    
    # Check N_point rules
    n_point_rules = []
    for r in applicable:
        if 'N_point' in r.get('outputs', {}):
            n_point_rules.append((r['name'], r['confidence'], r['outputs']['N_point']))
    
    # Check suppression rules that output N_point: "0"
    suppression = []
    for r in applicable:
        outputs = r.get('outputs', {})
        if 'N_point' in outputs and outputs['N_point'] == '0':
            suppression.append((r['name'], r['confidence']))
    
    year = step * dt
    if step < 10 or step % 20 == 0 or curr_state['N_point'] != new_state['N_point']:
        print(f"Step {step} ({year:.2f} yr): N_point {curr_state['N_point']} -> {new_state['N_point']}")
        if n_point_rules:
            print(f"  Growth rules: {n_point_rules}")
        if suppression:
            print(f"  Suppression rules: {suppression}")
    
    trajectory.append(new_state)
    curr_state = new_state

# Compute fidelity
from ca_analytics import _fidelity_stats
fidelity = _fidelity_stats(trajectory, ode_result, patient)
print(f"\nOverall fidelity: {fidelity['overall_agreement']:.3f}")
print("N_point agreement:", fidelity['per_variable_agreement']['N_point'])

# Final state comparison
ca_final = trajectory[-1]
ode_final_discrete = discretize_state(ode_result["states"][-1])
print("\nFinal state comparison:")
for var in ca_final:
    ca_val = ca_final[var]
    ode_val = ode_final_discrete.get(var)
    match = "✓" if ca_val == ode_val else "✗"
    print(f"  {var}: CA {ca_val} vs ODE {ode_val} {match}")