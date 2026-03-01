#!/usr/bin/env python3
"""Add ROS stabilization rule and test."""

import sys
sys.path.insert(0, '.')
import json
import copy
from simulator import simulate
from ca_simulator import _build_context, step_cell
from ca_analytics import _fidelity_stats
from ca_schema import discretize_state
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

with open('tuned_rules.json', 'r') as f:
    rules = json.load(f)

# Add stabilization rule
new_rule = {
    "tier": 2,
    "name": "ROS_stabilization_elevated",
    "inputs": {"ROS": "elevated"},
    "context": {},
    "outputs": {"ROS": "0"},
    "confidence": 0.95,
    "citation": "Stabilize ROS at elevated level to match ODE",
}
rules.append(new_rule)

print(f"Added rule, total {len(rules)} rules")

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

ode = simulate(patient=patient, intervention=intervention)
fidelity = _fidelity_stats(trajectory, ode, patient)

print(f"Overall agreement: {fidelity['overall_agreement']:.3f}")
print("Per-variable agreement:")
for var, agree in fidelity['per_variable_agreement'].items():
    print(f"  {var}: {agree:.3f}")

# ROS matches
ca_ros = [s['ROS'] for s in trajectory]
ode_ros = [discretize_state(s)['ROS'] for s in ode['states'][::int(0.25/0.01)]]
matches = sum(1 for i in range(len(ca_ros)) if ca_ros[i] == ode_ros[i])
print(f"\nROS bin matches: {matches}/{len(ca_ros)} ({matches/len(ca_ros):.3f})")

# Save new rules
with open('tuned_rules_stabilized.json', 'w') as f:
    json.dump(rules, f, indent=2)
print("Saved to tuned_rules_stabilized.json")