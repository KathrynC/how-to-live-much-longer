"""son_biologist_sim_20yr.py

Evaluates the 20-year trajectory for the 'Wildlife Biologist' (Son):
- 28yo Male, APOE4 Het
- Probable EDS (Structural metabolic drag)
- Processing speed LD (Neuro-metabolic debt)
- Goal: Wildlife Biology career vs. low energy/anxiety bottlenecks.
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_son_sim():
    print("Running 20-Year 'Wildlife Biologist' Simulation (Son's Profile)...")
    
    # 1. Peter's Profile (Recalibrated with 23andMe)
    son_patient = dict(DEFAULT_PATIENT)
    son_patient.update({
        "baseline_age": 28.0,
        "profile": "biologist",
        "apoe_genotype": "apoe4_het", # Confirmed by 'Slightly Increased Risk' report
        "sex": "male",
        "intellectual_engagement": 0.8,
        "grief_intensity": 0.7,
        "social_support": 0.8,
        "protein_intake": 0.8,
        "deep_sleep_genetic_penalty": 0.7 # Less likely to be a deep sleeper
    })
    
    # 2. Scenario A: Baseline (Current status)
    # Chaotic sleep, moderate fructose, low targeted support
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "diet_type": "standard",
        "exercise_type": "balanced",
        "exercise_level": 0.3,
        "fasting_regimen": 0.0,
        "animal_protocol": "active" # He does farm chores
    })
    sleep_base = 0.4 # Chaotic sleep
    
    # 3. Scenario B: The 'Mito-Biologist' Optimized Protocol
    # Strategic guards: Max Sleep quality, Mitochondrial Fuel (NR/NAD), 
    # High-dose protein, and Mediterranean-Keto hybrid diet.
    intervention_opt = dict(DEFAULT_INTERVENTION)
    intervention_opt.update({
        "diet_type": "mediterranean",
        "exercise_type": "resistance", # High ROI for low muscle tone
        "exercise_level": 0.6,
        "fasting_regimen": 0.5,
        "nr_dose": 0.9, # High priority for energy debt
        "magnesium_dose": 0.9, # For anxiety and sleep stability
        "animal_protocol": "full_farm" # Maximize the Farm-Neural synergy
    })
    sleep_opt = 0.85 # Goal: Stabilized 8-hour window
    
    horizon = 20.0
    res_base = unified_simulate(patient=son_patient, intervention=intervention_base, sleep_quality=sleep_base, sim_years=horizon)
    res_opt = unified_simulate(patient=son_patient, intervention=intervention_opt, sleep_quality=sleep_opt, sim_years=horizon)
    
    print(f"\nSon's Results at Year 20 (Age 48.0):")
    print(f"{'Metric':<25} | {'Baseline (Current)':<20} | {'Mito-Biologist'}")
    print("-" * 75)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("ATP Production", 2, "state"),
        ("Brain ROS", 3, "state"),
        ("Muscle Mass", 26, "state"),
        ("Grief / Anxiety", 18, "state"),
        ("BDNF / Growth Signal", 27, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_b, v_o = res_base["memory_index"][-1], res_opt["memory_index"][-1]
        else:
            v_b, v_o = res_base["states"][-1, idx], res_opt["states"][-1, idx]
        print(f"{label:<25} | {v_b:20.4f} | {v_o:.4f}")

if __name__ == "__main__":
    run_son_sim()
