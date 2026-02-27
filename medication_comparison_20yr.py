"""medication_comparison_20yr.py

Compares three medication protocols over 20 years for a 63.83yo scholar:
1. Clean Sweep: Rapamycin (0.5) + Senolytics (0.5)
2. Metabolic Shield: Metformin (proxy) + Acarbose (proxy)
3. Longevity Triplet: Rapa (0.3) + Metformin (proxy) + Low-Dose Seno (0.2)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_medication_comparison():
    print("Evaluating 20-Year Medication Protocol Comparison...")
    
    # 1. Base Profile (Sustainable Scholar + Supplement Duo)
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.5,
        "fructose_intake": 0.2
    })
    
    # Baseline intervention (Hybrid Diet + 80/20 Exercise + Power Duo Supplements)
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "diet_type": "mediterranean",
        "exercise_type": "aerobic",
        "exercise_level": 0.8,
        "fasting_regimen": 0.5,
        "alcohol_intake": 0.1,
        "nr_dose": 0.8, # Power Duo Proxy
        "ala_dose": 0.8,
        "magnesium_dose": 0.8
    })
    
    sleep_q = 0.8
    horizon = 20.0
    
    # 2. Define Medication Protocols
    
    # Protocol 1: Clean Sweep
    int_clean = dict(intervention_base)
    int_clean.update({
        "rapamycin_dose": 0.5,
        "senolytic_dose": 0.5
    })
    
    # Protocol 2: Metabolic Shield
    # Metformin maps to IS boost and demand reduction
    # Acarbose maps to fructose mitigation
    int_metabolic = dict(intervention_base)
    int_metabolic.update({
        "fasting_regimen": 0.9, # Metformin IS proxy
        "coq10_dose": 0.5 # Metformin metabolic demand proxy
    })
    patient_metabolic = dict(patient)
    patient_metabolic["fructose_intake"] = 0.05 # Acarbose proxy
    
    # Protocol 3: Longevity Triplet
    int_triplet = dict(intervention_base)
    int_triplet.update({
        "rapamycin_dose": 0.3,
        "senolytic_dose": 0.2,
        "fasting_regimen": 0.7 # Metformin IS proxy
    })
    
    # 3. Run Scenarios
    print("Simulating Clean Sweep (Rapa/Seno)...")
    res_clean = unified_simulate(patient=patient, intervention=int_clean, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Metabolic Shield (Metformin/Acarbose)...")
    res_metab = unified_simulate(patient=patient_metabolic, intervention=int_metabolic, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Longevity Triplet...")
    res_triplet = unified_simulate(patient=patient, intervention=int_triplet, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nMedication Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Clean Sweep':<15} | {'Metabolic':<15} | {'Triplet'}")
    print("-" * 80)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("ATP Production", 2, "state"),
        ("Brain ROS", 3, "state"),
        ("Amyloid Burden", 12, "state"),
        ("Senescent Fraction", 5, "state"),
        ("Insulin Sensitivity", 32, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_c, v_m, v_t = res_clean["memory_index"][-1], res_metab["memory_index"][-1], res_triplet["memory_index"][-1]
        else:
            v_c, v_m, v_t = res_clean["states"][-1, idx], res_metab["states"][-1, idx], res_triplet["states"][-1, idx]
        print(f"{label:<25} | {v_c:<15.4f} | {v_m:<15.4f} | {v_t:.4f}")

if __name__ == "__main__":
    run_medication_comparison()
