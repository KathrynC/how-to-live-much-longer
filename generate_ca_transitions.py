#!/usr/bin/env python3
"""
Generate dataset of CA state transitions for training HGNN.

For each random rule confidence vector, run a full CA trajectory (120 steps),
collect (state_t, confidence) -> state_{t+1} pairs.
State is represented as continuous exemplar values (8 floats).
"""

import json
import copy
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ca_simulator import run_single_cell, step_cell, _build_context
from ca_schema import discretize_state, continuous_exemplar, CA_VAR_ORDER
from simulator import initial_state
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION

def load_rules(path="final_tuned_rules.json"):
    with open(path, 'r') as f:
        return json.load(f)

def update_rule_confidences(rules, conf_dict):
    """Update confidence values in a copy of rules."""
    rules_copy = copy.deepcopy(rules)
    rule_index = {r['name']: i for i, r in enumerate(rules_copy)}
    for name, conf in conf_dict.items():
        if name in rule_index:
            rules_copy[rule_index[name]]['confidence'] = max(0.0, min(1.0, conf))
    return rules_copy

def generate_transitions(n_samples=100, seed=42):
    """Generate dataset of transitions.
    
    Args:
        n_samples: number of random rule confidence vectors.
        seed: random seed.
    
    Returns:
        X_state: array (n_transitions, 8) current state continuous exemplar.
        X_conf: array (n_transitions, 45) rule confidences.
        X_age: array (n_transitions, 1) age in years.
        Y_state: array (n_transitions, 8) next state continuous exemplar.
        rule_names: list of 45 rule names.
    """
    base_rules = load_rules()
    rule_names = [r['name'] for r in base_rules]
    n_rules = len(rule_names)
    
    rng = np.random.default_rng(seed)
    
    # Pre-allocate lists (we'll collect then convert)
    X_state_list = []
    X_conf_list = []
    X_age_list = []
    Y_state_list = []
    
    patient = dict(DEFAULT_PATIENT)
    intervention = dict(DEFAULT_INTERVENTION)
    sim_years = 30.0
    dt = 0.25
    n_steps = int(sim_years / dt)  # 120
    
    for sample_idx in range(n_samples):
        # Random rule confidences
        conf_vec = rng.random(n_rules)
        conf_dict = {name: float(conf_vec[i]) for i, name in enumerate(rule_names)}
        rules = update_rule_confidences(base_rules, conf_dict)
        
        # Initialize CA
        continuous_init = initial_state(patient)
        state = discretize_state(continuous_init)
        prev_state = None
        
        # Run trajectory
        for step in range(n_steps):
            ctx = _build_context(step, patient, intervention, prev_state, state)
            new_state, fired = step_cell(state, ctx, rules)
            
            # Convert states to continuous exemplar
            curr_cont = continuous_exemplar(state)
            next_cont = continuous_exemplar(new_state)
            age = patient.get("baseline_age", 70.0) + step * dt
            
            # Store transition (except last step where there is no next state)
            # We'll store current state and confidence, age; target is next state
            X_state_list.append(curr_cont)
            X_conf_list.append(conf_vec)
            X_age_list.append(age)
            Y_state_list.append(next_cont)
            
            prev_state = state
            state = new_state
        
        if (sample_idx + 1) % 10 == 0:
            print(f"Generated {sample_idx + 1}/{n_samples} samples")
    
    # Convert to numpy arrays
    X_state = np.array(X_state_list, dtype=np.float32)
    X_conf = np.array(X_conf_list, dtype=np.float32)
    X_age = np.array(X_age_list, dtype=np.float32).reshape(-1, 1)
    Y_state = np.array(Y_state_list, dtype=np.float32)
    
    # Sanity check
    print(f"Generated {X_state.shape[0]} transitions")
    print(f"State shape: {X_state.shape}")
    print(f"Conf shape: {X_conf.shape}")
    print(f"Age shape: {X_age.shape}")
    print(f"Target shape: {Y_state.shape}")
    
    return X_state, X_conf, X_age, Y_state, rule_names

def save_dataset(output_path="artifacts/ca_transitions.npz", n_samples=100, seed=42):
    """Generate and save dataset as compressed numpy file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    X_state, X_conf, X_age, Y_state, rule_names = generate_transitions(n_samples, seed)
    np.savez_compressed(
        output_path,
        X_state=X_state,
        X_conf=X_conf,
        X_age=X_age,
        Y_state=Y_state,
        rule_names=rule_names
    )
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CA transition dataset")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of random rule confidence vectors")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str,
                        default="artifacts/ca_transitions.npz",
                        help="Output .npz file path")
    args = parser.parse_args()
    
    save_dataset(args.output, args.n_samples, args.seed)