"""grad_school_resilience_sim.py

Compares two grad school scenarios in the 33-state Universal Human Digital Twin:
1. The Burnout Path (High Stress, Low Sleep)
2. The Sustainable Scholar (High Stress, Protected Sleep & Metabolism)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_grad_school_sim():
    print("Running Grad School Resilience Simulation (33-State Model)...")
    
    # Base Grad Student Profile
    grad_student = dict(DEFAULT_PATIENT)
    grad_student.update({
        "baseline_age": 30.0, # Assuming a typical grad student age for starting
        "apoe_genotype": "apoe4_het", # Testing against the risk factor
        "intellectual_engagement": 1.0, # Graduate school intensity
        "grief_intensity": 0.8, # Chronic stress / high pressure
        "pollution_exposure": 0.2,
        "social_support": 0.5
    })
    
    # Scenario A: The Burnout Path
    # High sugar/caffeine reliance, poor sleep, no exercise
    intervention_a = dict(DEFAULT_INTERVENTION)
    intervention_a.update({
        "exercise_level": 0.1,
        "alcohol_intake": 0.3,
        "fasting_regimen": 0.0
    })
    patient_a = dict(grad_student)
    patient_a["fructose_intake"] = 0.8 # Reliance on processed snacks/sugar
    sleep_a = 0.4 # Typical "all-nighter" / disrupted sleep
    
    # Scenario B: The Sustainable Scholar
    # Strategic guards: Protected sleep, low sugar, metabolic support
    intervention_b = dict(DEFAULT_INTERVENTION)
    intervention_b.update({
        "exercise_level": 0.5, # Moderate aerobic exercise for CBF
        "fasting_regimen": 0.5, # Intermittent fasting for insulin sensitivity
        "therapy_intensity": 0.3 # Stress management
    })
    patient_b = dict(grad_student)
    patient_b["fructose_intake"] = 0.1 # Low sugar diet
    sleep_b = 0.8 # Protected 7-8 hour sleep window
    
    # Run simulations for 10 years (The grad school + early postdoc window)
    horizon = 10.0
    res_a = unified_simulate(patient=patient_a, intervention=intervention_a, sleep_quality=sleep_a, sim_years=horizon)
    res_b = unified_simulate(patient=patient_b, intervention=intervention_b, sleep_quality=sleep_b, sim_years=horizon)
    
    print(f"\nResults after {horizon} years of Graduate School Intensity:")
    print(f"{'Metric':<25} | {'Burnout Path':<20} | {'Sustainable Scholar'}")
    print("-" * 75)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("Amyloid Burden", 12, "state"),
        ("ATP Production", 2, "state"),
        ("Brain ROS", 3, "state"),
        ("Insulin Sensitivity", 32, "state"),
        ("M1 Microglia (%)", 19, "state"),
        ("Grief / Stress", 18, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            val_a = res_a["memory_index"][-1]
            val_b = res_b["memory_index"][-1]
        else:
            val_a = res_a["states"][-1, idx]
            val_b = res_b["states"][-1, idx]
        print(f"{label:<25} | {val_a:20.4f} | {val_b:.4f}")

if __name__ == "__main__":
    run_grad_school_sim()
