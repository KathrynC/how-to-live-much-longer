"""integrated_social_sim_20yr.py

Evaluates the 'Integrated Scholar-Mentor' protocol over 20 years:
- UVM Research Group (Collaborative)
- Creative Grief Interventions + Daily Friendships (Emotional)
- Intensive Intergenerational Work (Mentoring)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_integrated_sim():
    print("Evaluating the 'Integrated Scholar-Mentor' Protocol (20-Year Horizon)...")
    
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.8,
        "social_support": 0.5
    })
    
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
    
    # Run Scenario: Integrated Scholar-Mentor
    print("Simulating Integrated Scholar-Mentor...")
    int_custom = dict(intervention_base)
    int_custom["social_protocol"] = "integrated"
    res_custom = unified_simulate(patient=patient, intervention=int_custom, sleep_quality=sleep_q, sim_years=horizon)
    
    # Previous Winner for Comparison: Collaborative
    print("Simulating Collaborative (Previous Winner)...")
    int_collab = dict(intervention_base)
    int_collab["social_protocol"] = "collaborative"
    res_collab = unified_simulate(patient=patient, intervention=int_collab, sleep_quality=sleep_q, sim_years=horizon)
    
    print(f"\nFinal Comparison at Year 20 (Age 83.8):")
    print(f"{'Metric':<25} | {'Previous Winner':<20} | {'Integrated Scholar-Mentor'}")
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
            v_old, v_new = res_collab["memory_index"][-1], res_custom["memory_index"][-1]
        else:
            v_old, v_new = res_collab["states"][-1, idx], res_custom["states"][-1, idx]
        print(f"{label:<25} | {v_old:20.4f} | {v_new:.4f}")

if __name__ == "__main__":
    run_integrated_sim()
