#!/usr/bin/env python3
"""
Test the fixed copy-count vs fraction mapping.
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from simulator import simulate
from ca_schema import discretize_state, continuous_exemplar, CA_VAR_ORDER, BIN_SCHEMA
from constants import DEFAULT_INTERVENTION

def load_sample_patients(n=5):
    """Load first n normal patients."""
    path = Path("artifacts/sample_patients_100.json")
    with open(path, 'r') as f:
        data = json.load(f)
    return data["patients"][:n]

def test_mapping():
    patients = load_sample_patients(3)
    interventions = [
        dict(DEFAULT_INTERVENTION),  # no treatment (all zeros)
        {"rapamycin_dose": 0.5, "nad_supplement": 0.75, "senolytic_dose": 0.5,
         "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.5}
    ]
    
    for p in patients:
        patient_dict = {k: p[k] for k in ["baseline_age", "baseline_heteroplasmy", 
                                         "baseline_nad_level", "genetic_vulnerability",
                                         "metabolic_demand", "inflammation_level"]}
        for interv in interventions:
            print(f"\nPatient {p['_id']}, intervention {interv}")
            result = simulate(patient=patient_dict, intervention=interv)
            states = result["states"]
            # Take first, middle, last state
            for idx in [0, len(states)//2, -1]:
                state = states[idx]
                discrete = discretize_state(state)
                print(f"  Step {idx}: N_h={state[0]:.3f}, N_d={state[1]:.3f}, N_p={state[7]:.3f}")
                print(f"    Total: {state[0]+state[1]+state[7]:.3f}")
                print(f"    Deletion fraction: {state[1]/(state[0]+state[1]+state[7]):.3f}")
                print(f"    Bins: N_del={discrete['N_deletion']}, N_point={discrete['N_point']}")
                # Check if fraction matches bin
                del_frac = state[1]/(state[0]+state[1]+state[7])
                thr = BIN_SCHEMA["N_deletion"]["thresholds"]
                if del_frac < thr[0]:
                    expected = "minimal"
                elif del_frac < thr[1]:
                    expected = "growing"
                elif del_frac < thr[2]:
                    expected = "approaching_cliff"
                else:
                    expected = "past_cliff"
                print(f"    Expected bin: {expected}, got: {discrete['N_deletion']}")
                if expected != discrete["N_deletion"]:
                    print("    *** MISMATCH ***")

if __name__ == "__main__":
    test_mapping()