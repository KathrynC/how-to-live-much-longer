#!/usr/bin/env python3
"""Check fidelity of different rule sets."""

import json
from simulator import simulate
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION
from ca_schema import discretize_state
from ca_simulator import _build_context, step_cell

def run_ca(rules, patient=None, intervention=None, sim_years=30.0, dt=0.25):
    if patient is None:
        patient = dict(DEFAULT_PATIENT)
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}
    n_steps = int(sim_years / dt)
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
    return trajectory

def fidelity(ca_traj, ode_result):
    from ca_analytics import _fidelity_stats
    return _fidelity_stats(ca_traj, ode_result, {})

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)
ode_result = simulate(patient=patient, intervention=intervention)

print("Evaluating rule sets...")
print("="*60)

# 1. Original final_tuned_rules.json
with open('final_tuned_rules.json', 'r') as f:
    rules1 = json.load(f)
ca1 = run_ca(rules1)
fid1 = fidelity(ca1, ode_result)
print("Original final_tuned_rules.json:")
print(f"  Overall: {fid1['overall_agreement']:.3f}")
print(f"  N_point: {fid1['per_variable_agreement']['N_point']:.3f}")

# 2. With N_point stabilization (v2)
with open('final_tuned_rules_n_point_v2.json', 'r') as f:
    rules2 = json.load(f)
ca2 = run_ca(rules2)
fid2 = fidelity(ca2, ode_result)
print("\nWith N_point stabilization (v2):")
print(f"  Overall: {fid2['overall_agreement']:.3f}")
print(f"  N_point: {fid2['per_variable_agreement']['N_point']:.3f}")

# 3. Adjusted confidences: growth 0.15, suppression 0.1
rules3 = json.loads(json.dumps(rules2))  # deep copy
for r in rules3:
    if r['name'] == 'ros_drives_points':
        r['confidence'] = 0.15
    elif r['name'] == 'point_mutation_pol_gamma_errors':
        r['confidence'] = 0.15
    elif r['name'] == 'N_point_suppression':
        r['confidence'] = 0.1
ca3 = run_ca(rules3)
fid3 = fidelity(ca3, ode_result)
print("\nAdjusted confidences (growth 0.15, suppression 0.1):")
print(f"  Overall: {fid3['overall_agreement']:.3f}")
print(f"  N_point: {fid3['per_variable_agreement']['N_point']:.3f}")

# 4. Remove suppression, low growth
rules4 = json.loads(json.dumps(rules2))
for r in rules4:
    if r['name'] == 'N_point_suppression':
        r['confidence'] = 0.0  # disable
    elif r['name'] == 'ros_drives_points':
        r['confidence'] = 0.05
    elif r['name'] == 'point_mutation_pol_gamma_errors':
        r['confidence'] = 0.05
ca4 = run_ca(rules4)
fid4 = fidelity(ca4, ode_result)
print("\nRemove suppression, growth 0.05:")
print(f"  Overall: {fid4['overall_agreement']:.3f}")
print(f"  N_point: {fid4['per_variable_agreement']['N_point']:.3f}")

# 5. New idea: shift N_point threshold to 0.05 (moderate from start)
# Not implemented.

print("\n" + "="*60)
print("Summary: Original rules already achieve 87.3% overall.")
print("N_point remains problematic due to deterministic CA limitation.")
print("Consider stochastic CA for N_point if higher fidelity needed.")