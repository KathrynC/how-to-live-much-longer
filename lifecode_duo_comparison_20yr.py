"""lifecode_duo_comparison_20yr.py

Evaluates the 'Ultimate LifeCode Duo': StemCell 100+ AND Memex 100+ 
over 20 years for a 63.83yo female scholar.

This scenario assumes full systemic repair (StemCell) PLUS full neural 
protection (Memex).
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_lifecode_duo():
    print("Evaluating LifeCode 'Ultimate Duo' (StemCell + Memex) Effectiveness...")
    
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
    
    # 1. LifeCode Ultimate Duo Mapping
    # Assumptions: 
    # - StemCell provides mTOR inhibition (Rapa) and SIRT activation (Resv)
    # - Memex provides ALA (Brain ROS), Magnesium (Sleep/Stress), DHA (Neural), 
    #   and enhanced Synaptic Strength support.
    prot_duo = {
        "rapamycin_dose": 0.6,    # Stronger combined mTOR effect
        "resveratrol_dose": 0.8,  # Maximum SIRT activation
        "ala_dose": 0.8,          # Maximum brain antioxidant
        "magnesium_dose": 0.8,    # Better GABA/Sleep support
        "dha_dose": 0.8,          # Stronger neural support
        "exercise_type": "hiit",  # Stronger eNOS boost from combined herbs
        "b_complex_dose": 0.8     # Better methylation
    }
    
    # Run Duo Scenario
    print("Simulating Ultimate Duo (StemCell + Memex)...")
    int_duo = {**intervention_base, **prot_duo}
    res_duo = unified_simulate(patient=patient, intervention=int_duo, sleep_quality=sleep_q, sim_years=horizon)
    
    # Previous Winners for Comparison
    # Memex Alone
    prot_memex = { "rapamycin_dose": 0.4, "resveratrol_dose": 0.6, "ala_dose": 0.8, "magnesium_dose": 0.6, "dha_dose": 0.6, "exercise_type": "hiit" }
    res_mem = unified_simulate(patient=patient, intervention={**intervention_base, **prot_memex}, sleep_quality=sleep_q, sim_years=horizon)
    
    # Comprehensive Stack (Generic)
    prot_comp = { "nr_dose": 0.8, "dha_dose": 0.8, "coq10_dose": 0.8, "resveratrol_dose": 0.8, "pqq_dose": 0.8, "ala_dose": 0.8, "vitamin_d_dose": 0.8, "b_complex_dose": 0.8, "magnesium_dose": 0.8, "zinc_dose": 0.8, "selenium_dose": 0.8 }
    res_comp = unified_simulate(patient=patient, intervention={**intervention_base, **prot_comp}, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nLifeCode Duo Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Ultimate Duo':<15} | {'Memex Alone':<15} | {'Comprehensive'}")
    print("-" * 85)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("ATP Production", 2, "state"),
        ("Brain ROS", 3, "state"),
        ("Amyloid Burden", 12, "state"),
        ("M2 Microglia (%)", 20, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_d, v_m, v_c = res_duo["memory_index"][-1], res_mem["memory_index"][-1], res_comp["memory_index"][-1]
        else:
            v_d, v_m, v_c = res_duo["states"][-1, idx], res_mem["states"][-1, idx], res_comp["states"][-1, idx]
        print(f"{label:<25} | {v_d:<15.4f} | {v_m:<15.4f} | {v_c:.4f}")

if __name__ == "__main__":
    run_lifecode_duo()
