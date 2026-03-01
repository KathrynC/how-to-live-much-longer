#!/usr/bin/env python3
"""Diagnose which CA rules fire and cause divergence from ODE."""

import sys
sys.path.insert(0, '.')

from simulator import simulate
from ca_simulator import run_single_cell, _build_context, step_cell
from ca_schema import discretize_state, BIN_SCHEMA
import numpy as np

# Use default patient and no intervention
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

patient = dict(DEFAULT_PATIENT)
intervention = dict(DEFAULT_INTERVENTION)

print("Patient:", patient)
print("Intervention:", intervention)

# Run ODE
ode_result = simulate(patient=patient, intervention=intervention)
print("\nODE simulation:")
print(f"  Final ATP: {ode_result['states'][-1, 2]:.4f}")
print(f"  Final heteroplasmy: {ode_result['heteroplasmy'][-1]:.4f}")

# Run CA with detailed logging
pat = patient
intv = intervention
n_steps = int(30.0 / 0.25)
dt = 0.25

# Get initial continuous state
from simulator import initial_state
cont = initial_state(pat)
print(f"\nInitial continuous state: {cont}")

# Discretize
from ca_schema import discretize_state
init_discrete = discretize_state(cont)
print(f"Initial discrete state: {init_discrete}")

# Prepare simulation
trajectory = [init_discrete]
rule_log = []

curr_state = init_discrete
for step in range(n_steps):
    context = _build_context(step, pat, intv, 
                             prev_state=trajectory[-1] if step > 0 else None,
                             curr_state=curr_state)
    
    # Get applicable rules
    from ca_rules import get_applicable_rules
    applicable = get_applicable_rules(curr_state, context)
    
    # Apply rules
    new_state, fired = step_cell(curr_state, context)
    
    # Log
    rule_log.append([r["name"] for r in fired])
    trajectory.append(new_state)
    curr_state = new_state
    
    if step < 5:
        print(f"\nStep {step} (t={step*dt:.2f} yr):")
        print(f"  Context: age={context['age']:.1f}, epoch={context['age_epoch']}")
        print(f"  Applicable rules ({len(applicable)}): {[r['name'] for r in applicable]}")
        print(f"  Fired rules ({len(fired)}): {[r['name'] for r in fired]}")
        print(f"  State changes:")
        for var in curr_state:
            if curr_state[var] != trajectory[step][var]:
                print(f"    {var}: {trajectory[step][var]} -> {curr_state[var]}")
        # Compare with ODE discretized at this time
        ode_idx = min(int(round(step*dt / 0.01)), len(ode_result["states"])-1)
        ode_state = ode_result["states"][ode_idx]
        ode_discrete = discretize_state(ode_state)
        matches = sum(1 for var in curr_state if curr_state[var] == ode_discrete.get(var))
        print(f"  Matches ODE: {matches}/8")
        for var in curr_state:
            if curr_state[var] != ode_discrete.get(var):
                print(f"    {var}: CA {curr_state[var]} vs ODE {ode_discrete.get(var)} (ODE val {ode_state[BIN_SCHEMA[var]['index']]:.4f})")

# Compute overall fidelity
from ca_analytics import _fidelity_stats
fidelity = _fidelity_stats(trajectory, ode_result, patient)
print(f"\nOverall fidelity: {fidelity['overall_agreement']:.3f}")
print("Per-variable agreement:")
for var, agree in fidelity['per_variable_agreement'].items():
    print(f"  {var}: {agree:.3f}")

# Summary of most fired rules
all_firings = []
for step_rules in rule_log:
    all_firings.extend(step_rules)
from collections import Counter
counter = Counter(all_firings)
print("\nTop 10 most fired rules:")
for name, count in counter.most_common(10):
    print(f"  {name}: {count}")