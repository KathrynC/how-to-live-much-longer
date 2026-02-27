"""supplement_comparison_20yr.py

Compares three dietary supplement protocols over 20 years for a 63.83yo scholar:
1. Mitochondrial Stack (NR, CoQ10, PQQ, Resveratrol)
2. Inflammation/Synapse Stack (DHA, Magnesium, Vit D, Zinc)
3. Comprehensive Stack (All 11 nutraceuticals)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_supplement_comparison():
    print("Running 20-Year Supplement Protocol Comparison...")
    
    # Base Profile: Using the hybrid Mediterranean-Keto path and Aerobic Base
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.5,
        "fructose_intake": 0.1
    })
    
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "diet_type": "mediterranean",
        "exercise_type": "aerobic",
        "exercise_level": 0.8,
        "fasting_regimen": 0.5,
        "alcohol_intake": 0.1
    })
    
    sleep_q = 0.8
    horizon = 20.0
    
    # Define Protocols
    
    # 1. Mitochondrial Stack
    prot_mito = {
        "nr_dose": 0.8,
        "coq10_dose": 0.8,
        "pqq_dose": 0.8,
        "resveratrol_dose": 0.8
    }
    
    # 2. Inflammation/Synapse Stack
    prot_inf = {
        "dha_dose": 0.8,
        "magnesium_dose": 0.8,
        "vitamin_d_dose": 0.8,
        "zinc_dose": 0.8
    }
    
    # 3. Comprehensive Stack
    prot_comp = {
        "nr_dose": 0.8, "dha_dose": 0.8, "coq10_dose": 0.8,
        "resveratrol_dose": 0.8, "pqq_dose": 0.8, "ala_dose": 0.8,
        "vitamin_d_dose": 0.8, "b_complex_dose": 0.8,
        "magnesium_dose": 0.8, "zinc_dose": 0.8, "selenium_dose": 0.8
    }
    
    # Run Scenarios
    print("Simulating Mitochondrial Stack...")
    int_mito = {**intervention_base, **prot_mito}
    res_mito = unified_simulate(patient=patient, intervention=int_mito, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Inflammation Stack...")
    int_inf = {**intervention_base, **prot_inf}
    res_inf = unified_simulate(patient=patient, intervention=int_inf, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Comprehensive Stack...")
    int_comp = {**intervention_base, **prot_comp}
    res_comp = unified_simulate(patient=patient, intervention=int_comp, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nSupplement Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Mitochondrial':<15} | {'Inflammation':<15} | {'Comprehensive'}")
    print("-" * 85)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("ATP Production", 2, "state"),
        ("Brain ROS", 3, "state"),
        ("Amyloid Burden", 12, "state"),
        ("Insulin Sensitivity", 32, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_m, v_i, v_c = res_mito["memory_index"][-1], res_inf["memory_index"][-1], res_comp["memory_index"][-1]
        else:
            v_m, v_i, v_c = res_mito["states"][-1, idx], res_inf["states"][-1, idx], res_comp["states"][-1, idx]
        print(f"{label:<25} | {v_m:<15.4f} | {v_i:<15.4f} | {v_c:.4f}")

if __name__ == "__main__":
    run_supplement_comparison()
