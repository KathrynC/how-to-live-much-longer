"""social_comparison_20yr.py

Compares three social protocols over 20 years for a 63.83yo scholar:
1. Intellectual Collaborative (Max CR growth & MEF2)
2. Emotional Resilience (Max stress buffer / Grief decay)
3. Intergenerational Teaching (Purpose-driven stability)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_social_comparison():
    print("Running 20-Year Social Protocol Comparison...")
    
    # Base Profile
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.8, # Starting with high stress
        "social_support": 0.5
    })
    
    # Baseline intervention (Sustainable Scholar Full Stack)
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "diet_type": "mediterranean",
        "exercise_type": "aerobic",
        "exercise_level": 0.8,
        "fasting_regimen": 0.5,
        "alcohol_intake": 0.1,
        "nr_dose": 0.8,
        "rapamycin_dose": 0.3
    })
    
    sleep_q = 0.8
    horizon = 20.0
    
    # Run Scenarios
    print("Simulating Intellectual Collaborative...")
    int_collab = dict(intervention_base)
    int_collab["social_protocol"] = "collaborative"
    res_collab = unified_simulate(patient=patient, intervention=int_collab, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Emotional Resilience...")
    int_emot = dict(intervention_base)
    int_emot["social_protocol"] = "emotional"
    res_emot = unified_simulate(patient=patient, intervention=int_emot, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Intergenerational Teaching...")
    int_teach = dict(intervention_base)
    int_teach["social_protocol"] = "teaching"
    res_teach = unified_simulate(patient=patient, intervention=int_teach, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nSocial Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Collaborative':<15} | {'Emotional':<15} | {'Teaching'}")
    print("-" * 80)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("Cognitive Reserve (CR)", 11, "state"),
        ("MEF2 Activity", 8, "state"),
        ("Grief / Stress", 18, "state"),
        ("Brain ROS", 3, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_c, v_e, v_t = res_collab["memory_index"][-1], res_emot["memory_index"][-1], res_teach["memory_index"][-1]
        else:
            v_c, v_e, v_t = res_collab["states"][-1, idx], res_emot["states"][-1, idx], res_teach["states"][-1, idx]
        print(f"{label:<25} | {v_c:<15.4f} | {v_e:<15.4f} | {v_t:.4f}")

if __name__ == "__main__":
    run_social_comparison()
