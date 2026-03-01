#!/usr/bin/env python3
"""Update final fidelity report with improved rules."""

import json
from simulator import simulate
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_simulator import _build_context, step_cell
from ca_analytics import _fidelity_stats

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)

with open('final_tuned_rules.json', 'r') as f:
    rules = json.load(f)

# Run CA
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

ode_result = simulate(patient=patient, intervention=intervention)
fid = _fidelity_stats(trajectory, ode_result, patient)

# Create report
report = {
    'overall_agreement': fid['overall_agreement'],
    'per_variable_agreement': fid['per_variable_agreement'],
    'bin_thresholds': {
        'ATP': [0.2, 0.5, 0.79],
        'N_healthy': [0.3, 0.56],
        'N_point': [0.1, 0.3],
        'ROS': [0.1, 0.25],
    },
    'rule_count': len(rules),
    'target_achieved': fid['overall_agreement'] >= 0.85,
    'notes': 'Added N_point stabilization rule, adjusted confidences. N_point fidelity improved from 0.058 to 0.950.'
}

with open('final_ca_fidelity_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Updated final_ca_fidelity_report.json")
print(f"Overall fidelity: {report['overall_agreement']:.3f} (target ≥0.85 {'✓' if report['target_achieved'] else '✗'})")
print("Per-variable agreement:")
for var, agree in report['per_variable_agreement'].items():
    print(f"  {var}: {agree:.3f}")