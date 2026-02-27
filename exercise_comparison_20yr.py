"""exercise_comparison_20yr.py

Compares three exercise protocols over 20 years for a 63.83yo scholar:
1. Aerobic Base (Mitochondrial Biogenesis & Lung Capacity)
2. HIIT (Cerebrovascular flushing & BP reduction)
3. Progressive Resistance (Muscle mass & Myokines/BDNF)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_exercise_comparison():
    print("Running 20-Year Exercise Protocol Comparison...")
    
    # Base Profile
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.5
    })
    
    # Use the Mediterranean-Keto hybrid diet as established
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "diet_type": "mediterranean", # Hybrid proxy
        "fasting_regimen": 0.5,
        "alcohol_intake": 0.1,
        "exercise_level": 0.8 # Consistent intensity
    })
    
    sleep_q = 0.8
    horizon = 20.0
    
    # Run Scenarios
    print("Simulating Aerobic Base...")
    int_aero = dict(intervention_base)
    int_aero["exercise_type"] = "aerobic"
    res_aero = unified_simulate(patient=patient, intervention=int_aero, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating HIIT...")
    int_hiit = dict(intervention_base)
    int_hiit["exercise_type"] = "hiit"
    res_hiit = unified_simulate(patient=patient, intervention=int_hiit, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Resistance...")
    int_res = dict(intervention_base)
    int_res["exercise_type"] = "resistance"
    res_res = unified_simulate(patient=patient, intervention=int_res, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nExercise Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Aerobic':<15} | {'HIIT':<15} | {'Resistance'}")
    print("-" * 80)
    
    # Metrics to track: 
    # Memory Index(-1), ATP(2), CBF(21), BDNF(27), Muscle(26), Lung(28), Heteroplasmy
    def get_het(res, idx):
        total = res["states"][idx, 0] + res["states"][idx, 1] + res["states"][idx, 7]
        return (res["states"][idx, 1] + res["states"][idx, 7]) / max(total, 1e-12)

    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("ATP Production", 2, "state"),
        ("Cerebrovascular (CBF)", 21, "state"),
        ("BDNF / Myokines", 27, "state"),
        ("Muscle Mass", 26, "state"),
        ("Lung Capacity", 28, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_a, v_h, v_r = res_aero["memory_index"][-1], res_hiit["memory_index"][-1], res_res["memory_index"][-1]
        else:
            v_a, v_h, v_r = res_aero["states"][-1, idx], res_hiit["states"][-1, idx], res_res["states"][-1, idx]
        print(f"{label:<25} | {v_a:<15.4f} | {v_h:<15.4f} | {v_r:.4f}")
        
    print(f"{'Heteroplasmy':<25} | {get_het(res_aero, -1):<15.4f} | {get_het(res_hiit, -1):<15.4f} | {get_het(res_res, -1):.4f}")

if __name__ == "__main__":
    run_exercise_comparison()
