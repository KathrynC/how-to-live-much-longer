#!/usr/bin/env python3
"""Detailed fidelity analysis for best rule set."""

import json
from simulator import simulate
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_simulator import _build_context, step_cell

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)
ode_result = simulate(patient=patient, intervention=intervention)

with open('final_tuned_rules_n_point_v2.json', 'r') as f:
    rules = json.load(f)

pat = {**DEFAULT_PATIENT, **patient}
intv = {**DEFAULT_INTERVENTION, **intervention}
n_steps = int(30.0 / 0.25)
from simulator import initial_state
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

from ca_analytics import _fidelity_stats
fid = _fidelity_stats(trajectory, ode_result, patient)
print("Per-variable agreement:")
for var, agree in fid['per_variable_agreement'].items():
    print(f"  {var}: {agree:.3f}")
print(f"Overall: {fid['overall_agreement']:.3f}")

# Print N_point trajectory mismatch steps
print("\nN_point mismatch steps (year, CA bin vs ODE bin):")
ode_states = ode_result['states']
matches = 0
for i in range(len(trajectory)):
    ode_bin = discretize_state(ode_states[i])['N_point']
    ca_bin = trajectory[i]['N_point']
    if ca_bin == ode_bin:
        matches += 1
    else:
        year = i * 0.25
        print(f"  Year {year:.2f}: CA {ca_bin} vs ODE {ode_bin}")
        if i < 10:
            val = ode_states[i, 7]
            print(f"    ODE value: {val:.4f}")
print(f"\nN_point matches: {matches}/{len(trajectory)} ({matches/len(trajectory):.3f})")

# Check when CA transitions
for i in range(len(trajectory)):
    if trajectory[i]['N_point'] == 'moderate':
        print(f"CA first moderate at step {i} (year {i*0.25:.2f})")
        break
# Check when ODE transitions
for i in range(len(trajectory)):
    ode_bin = discretize_state(ode_states[i])['N_point']
    if ode_bin == 'moderate':
        print(f"ODE first moderate at step {i} (year {i*0.25:.2f})")
        break