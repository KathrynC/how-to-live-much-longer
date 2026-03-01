#!/usr/bin/env python3
"""
Compute hypergraph incidence matrix H (variables × rules) from rule table.
"""

import json
import numpy as np
from pathlib import Path

def load_rules(path="final_tuned_rules.json"):
    with open(path, 'r') as f:
        return json.load(f)

def build_incidence_matrix(rules):
    """Return incidence matrix H (variables × rules) as dense numpy array.
    
    Variables order: CA_VAR_ORDER from ca_schema.
    Rules order: same as rule list.
    
    Entry H[i,j] = 1 if variable i participates in rule j (as input or output).
    """
    from ca_schema import CA_VAR_ORDER
    var_index = {name: idx for idx, name in enumerate(CA_VAR_ORDER)}
    n_vars = len(CA_VAR_ORDER)
    n_rules = len(rules)
    H = np.zeros((n_vars, n_rules), dtype=np.float32)
    
    for j, rule in enumerate(rules):
        # Input variables
        for var_name in rule.get("inputs", {}):
            if var_name in var_index:
                H[var_index[var_name], j] = 1.0
        # Output variables
        for var_name in rule.get("outputs", {}):
            if var_name in var_index:
                H[var_index[var_name], j] = 1.0
        # Context variables are not state variables, ignore for now
    return H

def build_directed_incidence(rules):
    """Return two matrices: H_in and H_out (variables × rules).
    
    H_in[i,j] = 1 if variable i is input to rule j.
    H_out[i,j] = 1 if variable i is output of rule j.
    """
    from ca_schema import CA_VAR_ORDER
    var_index = {name: idx for idx, name in enumerate(CA_VAR_ORDER)}
    n_vars = len(CA_VAR_ORDER)
    n_rules = len(rules)
    H_in = np.zeros((n_vars, n_rules), dtype=np.float32)
    H_out = np.zeros((n_vars, n_rules), dtype=np.float32)
    
    for j, rule in enumerate(rules):
        for var_name in rule.get("inputs", {}):
            if var_name in var_index:
                H_in[var_index[var_name], j] = 1.0
        for var_name in rule.get("outputs", {}):
            if var_name in var_index:
                H_out[var_index[var_name], j] = 1.0
    return H_in, H_out

def compute_degrees(H):
    """Compute variable degrees (row sums) and hyperedge degrees (column sums)."""
    D_v = np.sum(H, axis=1)  # shape (n_vars,)
    D_e = np.sum(H, axis=0)  # shape (n_rules,)
    return D_v, D_e

def save_incidence(output_path="artifacts/hypergraph_incidence.npz"):
    """Save incidence matrix and metadata."""
    rules = load_rules()
    H = build_incidence_matrix(rules)
    H_in, H_out = build_directed_incidence(rules)
    D_v, D_e = compute_degrees(H)
    
    from ca_schema import CA_VAR_ORDER
    rule_names = [r['name'] for r in rules]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        H=H,
        H_in=H_in,
        H_out=H_out,
        D_v=D_v,
        D_e=D_e,
        variable_names=CA_VAR_ORDER,
        rule_names=rule_names
    )
    print(f"Saved hypergraph incidence to {output_path}")
    print(f"Shape: {H.shape}")
    print(f"Variable degrees: {D_v}")
    print(f"Hyperedge degrees: {D_e}")

if __name__ == "__main__":
    save_incidence()