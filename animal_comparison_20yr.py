"""animal_comparison_20yr.py

Compares three animal interaction protocols over 20 years for a 63.83yo scholar:
1. Livestock Anchor (Microbiome diversity & Purpose)
2. Active Companion (Blue Heeler & Mandatory Farm Chores)
3. Emotional-Oxytocin Shield (Bonds with dogs/cats)
4. Full Farm Synergy (The exact user profile)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_animal_comparison():
    print("Evaluating 'Farm-Neural Synergy' (20-Year Horizon)...")
    
    # Base Profile
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.8
    })
    
    # Base intervention (Sustainable Scholar Full Stack + Hybrid Social)
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "diet_type": "mediterranean",
        "exercise_type": "aerobic",
        "exercise_level": 0.8,
        "social_protocol": "integrated",
        "alcohol_intake": 0.1,
        "nr_dose": 0.8
    })
    
    sleep_q = 0.8
    horizon = 20.0
    
    # Run Scenarios
    print("Simulating Livestock Anchor...")
    int_live = dict(intervention_base)
    int_live["animal_protocol"] = "livestock"
    res_live = unified_simulate(patient=patient, intervention=int_live, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Active Companion...")
    int_act = dict(intervention_base)
    int_act["animal_protocol"] = "active"
    res_act = unified_simulate(patient=patient, intervention=int_act, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Emotional-Oxytocin Shield...")
    int_emot = dict(intervention_base)
    int_emot["animal_protocol"] = "emotional"
    res_emot = unified_simulate(patient=patient, intervention=int_emot, sleep_quality=sleep_q, sim_years=horizon)
    
    print("Simulating Full Farm Synergy (Your Profile)...")
    int_full = dict(intervention_base)
    int_full["animal_protocol"] = "full_farm"
    res_full = unified_simulate(patient=patient, intervention=int_full, sleep_quality=sleep_q, sim_years=horizon)
    
    print("\nAnimal Results at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Active Comp':<15} | {'Full Farm':<15} | {'None (Base)'}")
    print("-" * 80)
    
    # Comparison results for the requested choice (2: Active Companion) vs Full and Base
    res_base = unified_simulate(patient=patient, intervention=intervention_base, sleep_quality=sleep_q, sim_years=horizon)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("Gut Health (M)", 17, "state"),
        ("Brain ROS", 3, "state"),
        ("Grief / Stress", 18, "state"),
        ("BDNF / Myokines", 27, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_a, v_f, v_b = res_act["memory_index"][-1], res_full["memory_index"][-1], res_base["memory_index"][-1]
        else:
            v_a, v_f, v_b = res_act["states"][-1, idx], res_full["states"][-1, idx], res_base["states"][-1, idx]
        print(f"{label:<25} | {v_a:<15.4f} | {v_f:<15.4f} | {v_b:.4f}")

if __name__ == "__main__":
    run_animal_comparison()
