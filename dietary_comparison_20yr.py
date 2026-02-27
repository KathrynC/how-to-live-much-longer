"""dietary_comparison_20yr.py

Compares three dietary profiles over 20 years for a 63.83yo scholar:
1. Mediterranean-Longevity (Anti-inflammatory)
2. Therapeutic Ketogenic (Energy Efficient)
3. Standard Western (Metabolic Stressor)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_diet_comparison():
    print("Running 20-Year Dietary Comparison...")
    
    # Base Profile
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.5,
        "fructose_intake": 0.3 # Moderate baseline
    })
    
    # Baseline intervention (Sustainable Scholar protocol)
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "exercise_level": 0.8,
        "fasting_regimen": 0.5,
        "alcohol_intake": 0.1
    })
    
    sleep_q = 0.8
    horizon = 20.0
    
    # Run Scenarios
    print("Simulating Mediterranean...")
    int_med = dict(intervention_base)
    int_med["diet_type"] = "mediterranean"
    res_med = unified_simulate(patient=patient, intervention=int_med, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Ketogenic...")
    int_keto = dict(intervention_base)
    int_keto["diet_type"] = "keto"
    res_keto = unified_simulate(patient=patient, intervention=int_keto, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Western...")
    int_west = dict(intervention_base)
    int_west["diet_type"] = "western"
    res_west = unified_simulate(patient=patient, intervention=int_west, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nDietary Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Mediterranean':<15} | {'Ketogenic':<15} | {'Western'}")
    print("-" * 80)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("Insulin Sensitivity", 32, "state"),
        ("Brain ROS", 3, "state"),
        ("Amyloid Burden", 12, "state"),
        ("ATP Production", 2, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_m, v_k, v_w = res_med["memory_index"][-1], res_keto["memory_index"][-1], res_west["memory_index"][-1]
        else:
            v_m, v_k, v_w = res_med["states"][-1, idx], res_keto["states"][-1, idx], res_west["states"][-1, idx]
        print(f"{label:<25} | {v_m:<15.4f} | {v_k:<15.4f} | {v_w:.4f}")

if __name__ == "__main__":
    run_diet_comparison()
