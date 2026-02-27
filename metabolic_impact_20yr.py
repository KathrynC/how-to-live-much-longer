"""metabolic_impact_20yr.py

Compares the 'Sustainable Scholar' path against 'Metabolic Sabotage' 
(High Fructose, Low Insulin Sensitivity) over 20 years.
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_metabolic_comparison():
    print("Running 20-Year Metabolic Impact Simulation...")
    
    # 1. Profile
    patient_base = dict(DEFAULT_PATIENT)
    patient_base.update({"baseline_age": 63.83, "apoe_genotype": "apoe4_het", "sex": "female", "intellectual_engagement": 1.0, "grief_intensity": 0.5})
    
    # 2. Scenario A: Sustainable Scholar (Guard on Metabolism)
    patient_a = dict(patient_base)
    patient_a["fructose_intake"] = 0.1
    intervention_a = dict(DEFAULT_INTERVENTION)
    intervention_a.update({"fasting_regimen": 0.8, "exercise_level": 1.0, "alcohol_intake": 0.1})
    
    # 3. Scenario B: Metabolic Sabotage (High Sugar, No Fasting)
    patient_b = dict(patient_base)
    patient_b["fructose_intake"] = 0.8
    intervention_b = dict(DEFAULT_INTERVENTION)
    intervention_b.update({"fasting_regimen": 0.0, "exercise_level": 1.0, "alcohol_intake": 0.1})
    
    res_a = unified_simulate(patient=patient_a, intervention=intervention_a, sleep_quality=0.8, sim_years=20.0)
    res_b = unified_simulate(patient=patient_b, intervention=intervention_b, sleep_quality=0.8, sim_years=20.0)
    
    print(f"\nResults at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Sustainable Scholar':<20} | {'Metabolic Sabotage'}")
    print("-" * 75)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("Insulin Sensitivity", 32, "state"),
        ("Brain ROS", 3, "state"),
        ("Amyloid Burden", 12, "state"),
        ("ATP Production", 2, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            val_a, val_b = res_a["memory_index"][-1], res_b["memory_index"][-1]
        else:
            val_a, val_b = res_a["states"][-1, idx], res_b["states"][-1, idx]
        print(f"{label:<25} | {val_a:20.4f} | {val_b:.4f}")

if __name__ == "__main__":
    run_metabolic_comparison()
