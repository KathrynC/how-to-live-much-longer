#!/usr/bin/env python3
"""
RuleConfidenceSimulator: Zimmerman protocol adapter for sensitivity analysis
of rule confidence parameters.

Treats each rule's confidence (0.0-1.0) as an input parameter, runs the
semantic CA with default patient/intervention, returns discrete bin indices
of the final state (8 variables). Allows Sobol analysis to identify which
rules (and combinations) most influence cellular fate.
"""

import json
import copy
import numpy as np
from typing import Dict, Tuple

from ca_simulator import run_single_cell
from constants import DEFAULT_PATIENT, DEFAULT_INTERVENTION

class RuleConfidenceSimulator:
    """Simulator that varies rule confidences."""
    
    def __init__(self, rule_path="final_tuned_rules.json"):
        with open(rule_path, 'r') as f:
            self.base_rules = json.load(f)
        self.rule_names = [r['name'] for r in self.base_rules]
        self.n_rules = len(self.rule_names)
        self.rule_index = {name: i for i, name in enumerate(self.rule_names)}
        
    def param_spec(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for all rule confidences."""
        return {name: (0.0, 1.0) for name in self.rule_names}
    
    def run(self, params: Dict[str, float]) -> Dict[str, float]:
        """Run CA with given rule confidences.
        
        Args:
            params: dict mapping rule names to confidence values (0-1).
                   Missing keys keep default confidence from base_rules.
        
        Returns:
            Dict with keys:
                - 'final_bin_N_healthy', 'final_bin_N_deletion', ...
                  (integer bin index: 0 for first bin, 1 for second, etc.)
                - 'final_continuous_ATP', ... (continuous exemplar values)
                - 'final_heteroplasmy' (deletion het)
                - 'final_atp' (continuous ATP)
        """
        # Copy base rules and update confidences
        rules = copy.deepcopy(self.base_rules)
        for name, conf in params.items():
            if name in self.rule_index:
                idx = self.rule_index[name]
                # Clip to [0,1] just in case
                rules[idx]['confidence'] = max(0.0, min(1.0, conf))
            else:
                # Unknown rule name - ignore (could be context param)
                pass
        
        # Run CA with custom rules (cannot use run_single_cell directly)
        from simulator import initial_state
        from ca_schema import discretize_state
        from ca_simulator import _build_context, step_cell
        
        patient = dict(DEFAULT_PATIENT)
        intervention = dict(DEFAULT_INTERVENTION)
        sim_years = 30.0
        dt = 0.25
        n_steps = int(sim_years / dt)
        
        # Initialize from ODE initial state
        continuous_init = initial_state(patient)
        state = discretize_state(continuous_init)
        trajectory = [dict(state)]
        rule_log = []
        prev_state = None
        
        for step in range(n_steps):
            ctx = _build_context(step, patient, intervention, prev_state, state)
            new_state, fired = step_cell(state, ctx, rules)
            rule_log.append([r["name"] for r in fired])
            prev_state = state
            state = new_state
            trajectory.append(dict(state))
        
        final_state = state
        
        # Map bin labels to integer indices
        from ca_schema import BIN_SCHEMA
        bin_indices = {}
        for var_name, label in final_state.items():
            schema = BIN_SCHEMA[var_name]
            idx = schema['labels'].index(label)
            bin_indices[f'final_bin_{var_name}'] = float(idx)
        
        # Continuous exemplar values
        from ca_schema import continuous_exemplar
        cont_exemplar = continuous_exemplar(final_state)
        var_order = ['N_healthy', 'N_deletion', 'ATP', 'ROS',
                     'NAD', 'Senescent_fraction', 'Membrane_potential', 'N_point']
        for i, var in enumerate(var_order):
            bin_indices[f'final_continuous_{var}'] = float(cont_exemplar[i])
        
        # Compute deletion heteroplasmy (approximate)
        n_healthy = cont_exemplar[0]
        n_del = cont_exemplar[1]
        n_point = cont_exemplar[7]
        total = n_healthy + n_del + n_point
        if total > 1e-12:
            het = n_del / total
        else:
            het = 0.0
        bin_indices['final_heteroplasmy'] = het
        bin_indices['final_atp'] = float(cont_exemplar[2])
        
        return bin_indices

def test_simulator():
    """Quick test of the simulator."""
    sim = RuleConfidenceSimulator()
    print(f"Rules: {sim.n_rules}")
    spec = sim.param_spec()
    print(f"Parameters: {len(spec)}")
    print("Sample bounds:", list(spec.items())[:3])
    
    # Test with default confidences (empty params)
    result = sim.run({})
    print("\nDefault confidences result:")
    for k, v in list(result.items())[:5]:
        print(f"  {k}: {v}")
    
    # Test with one rule boosted
    params = {'ros_drives_points': 0.9}
    result2 = sim.run(params)
    print(f"\nWith ros_drives_points=0.9:")
    print(f"  final_bin_N_point: {result2['final_bin_N_point']}")
    print(f"  final_continuous_N_point: {result2['final_continuous_N_point']:.3f}")
    
    return sim

if __name__ == '__main__':
    test_simulator()