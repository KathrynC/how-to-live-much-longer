#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import json
from simulator import simulate
from ca_simulator import _build_context, step_cell
from ca_schema import discretize_state
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

with open('tuned_rules.json', 'r') as f:
    rules = json.load(f)

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)

# ODE
ode = simulate(patient=patient, intervention=intervention)
ode_states = ode['states']
ode_ros = [discretize_state(s)['ROS'] for s in ode_states[::int(0.25/0.01)]]

# CA
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

ca_ros = [s['ROS'] for s in trajectory]

print("Step CA_ROS ODE_ROS Match")
for step in range(len(ca_ros)):
    match = ca_ros[step] == ode_ros[step]
    if not match or step < 10:
        age = patient['baseline_age'] + step * 0.25
        print(f"{step} {ca_ros[step]} {ode_ros[step]} {'✓' if match else '✗'} age {age:.1f}")

# Count mismatches
mismatches = sum(1 for i in range(len(ca_ros)) if ca_ros[i] != ode_ros[i])
print(f"\nTotal mismatches: {mismatches}/{len(ca_ros)} ({mismatches/len(ca_ros):.3f})")

# First mismatch details
for step in range(len(ca_ros)):
    if ca_ros[step] != ode_ros[step]:
        age = patient['baseline_age'] + step * 0.25
        ode_idx = min(int(round(step * 0.25 / 0.01)), len(ode_states)-1)
        ros_val = ode_states[ode_idx, 3]
        print(f"\nFirst mismatch at step {step} (age {age:.2f}):")
        print(f"  CA bin: {ca_ros[step]}, ODE bin: {ode_ros[step]} (value {ros_val:.4f})")
        # Show applicable ROS rules
        context = _build_context(step, patient, intervention,
                                 prev_state=trajectory[step-1] if step > 0 else None,
                                 curr_state=trajectory[step])
        from ca_rules import get_applicable_rules
        applicable = get_applicable_rules(trajectory[step], context, rules)
        ros_rules = [r for r in applicable if 'ROS' in r.get('outputs', {})]
        if ros_rules:
            for r in ros_rules:
                print(f"  Rule {r['name']} conf {r['confidence']} output {r['outputs']['ROS']}")
        break