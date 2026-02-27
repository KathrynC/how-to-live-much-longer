"""lifecode_comparison_20yr.py

Evaluates LifeCode products StemCell 100 and Memex 100 over 20 years 
for a 63.83yo female scholar.

Mappings:
- StemCell 100: Boosts mitophagy_boost, reduces senescence, improves CBF/BP.
- Memex 100: Adds Synaptic Strength boost, Brain ROS reduction, and M2 activation.
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_lifecode_comparison():
    print("Evaluating LifeCode Protocol Effectiveness (20-Year Horizon)...")
    
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
    
    # Define LifeCode Protocols
    
    # 1. StemCell 100 Mapping
    # (mTOR inhibition, SIRTs activation, eNOS support)
    prot_stemcell = {
        "rapamycin_dose": 0.4, # mTOR inhibition proxy
        "resveratrol_dose": 0.6, # SIRT1 proxy
        "exercise_type": "hiit", # Vascular eNOS proxy
        "b_complex_dose": 0.5  # Methylation proxy
    }
    
    # 2. Memex 100 Mapping
    # (StemCell 100 base + L-Theanine/Gingerols/ALA for brain)
    prot_memex = {
        **prot_stemcell,
        "ala_dose": 0.8, # Antioxidant/Chelation
        "magnesium_dose": 0.6, # Theanine proxy (GABA/stress)
        "dha_dose": 0.6 # Neural support
    }
    
    # Run Scenarios
    print("Simulating StemCell 100+...")
    int_stem = {**intervention_base, **prot_stemcell}
    res_stem = unified_simulate(patient=patient, intervention=int_stem, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Memex 100+...")
    int_mem = {**intervention_base, **prot_memex}
    res_mem = unified_simulate(patient=patient, intervention=int_mem, sleep_quality=sleep_q, sim_years=horizon)
    
    # Reference: Comprehensive Stack from previous run
    prot_comp = {
        "nr_dose": 0.8, "dha_dose": 0.8, "coq10_dose": 0.8,
        "resveratrol_dose": 0.8, "pqq_dose": 0.8, "ala_dose": 0.8,
        "vitamin_d_dose": 0.8, "b_complex_dose": 0.8,
        "magnesium_dose": 0.8, "zinc_dose": 0.8, "selenium_dose": 0.8
    }
    print("Simulating Comprehensive Stack (Reference)...")
    int_comp = {**intervention_base, **prot_comp}
    res_comp = unified_simulate(patient=patient, intervention=int_comp, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nLifeCode Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'StemCell 100+':<15} | {'Memex 100+':<15} | {'Comprehensive'}")
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
            v_s, v_m, v_c = res_stem["memory_index"][-1], res_mem["memory_index"][-1], res_comp["memory_index"][-1]
        else:
            v_s, v_m, v_c = res_stem["states"][-1, idx], res_mem["states"][-1, idx], res_comp["states"][-1, idx]
        print(f"{label:<25} | {v_s:<15.4f} | {v_m:<15.4f} | {v_c:.4f}")

if __name__ == "__main__":
    run_lifecode_comparison()
