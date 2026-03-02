"""ea_john_iii_optimizer.py

Deep optimization for Subject J: John G. Cramer III (Age 62.1).
Investigates the "Pre-Shift" window before the age-65 acceleration.

NOTE: This file is ignored by git.
"""

import numpy as np
import os
import json
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def evaluate_john_iii_fitness(params):
    # ── SUBJECT J: JOHN G. CRAMER III (Age 62.1) ──
    j_p = dict(DEFAULT_PATIENT)
    j_p.update({
        "baseline_age": 62.1,
        "profile": "scholar", 
        "apoe_genotype": "apoe4_het", # Assume familial APOE4
        "sex": "male",
        "intellectual_engagement": 1.0, # Photographer/Technical
        "activity_type": "solitary_novel", # Visual composition / New locations
        "post_viral_load": 0.0 # "Never COVID" baseline
    })
    
    j_i = dict(DEFAULT_INTERVENTION)
    j_i.update({
        "diet_type": "mediterranean",
        "exercise_type": "aerobic", # Avid Hiker
        "exercise_level": params.get("exercise", 0.7), # Floor at 0.7 for "Avid"
        "sleep_quality": params.get("sleep", 0.8),
        "nr_dose": 0.5,
        "magnesium_dose": 0.7,
        "red_light_therapy": params.get("red_light", 0.5),
        "sauna_use": params.get("sauna", 0.5),
        "restorative_yoga": params.get("yoga", 0.5)
    })
    
    # Simulate 30 years (Age 62.1 -> 92.1)
    res = unified_simulate(patient=j_p, intervention=j_i, sim_years=30.0)
    
    atp = res["states"][-1, 2]
    mi = res["memory_index"][-1]
    ros = res["states"][-1, 3]
    
    # Fitness: Stability > 0.85 ATP and high Memory Index
    atp_penalty = max(0, 0.85 - atp) ** 2 * 50
    fitness = (atp * 10) + (mi * 5) - atp_penalty - (ros * 1)
    
    return {"fitness": fitness, "atp": atp, "mi": mi, "ros": ros}

def run_optimization(budget=40):
    print(f"Optimizing Protocol for John G. Cramer III (Age 62.1 -> 92.1)...")
    param_names = ["exercise", "sleep", "red_light", "sauna", "yoga"]
    current_params = {name: 0.6 for name in param_names}
    current_result = evaluate_john_iii_fitness(current_params)
    
    sigma = 0.05
    for i in range(budget):
        candidates = []
        for _ in range(5):
            candidate = {name: np.clip(current_params[name] + np.random.normal(0, sigma), 0.0, 1.0) for name in param_names}
            candidates.append((candidate, evaluate_john_iii_fitness(candidate)))
        
        candidates.sort(key=lambda x: x[1]["fitness"], reverse=True)
        if candidates[0][1]["fitness"] > current_result["fitness"]:
            current_params, current_result = candidates[0]
            sigma *= 1.05
        else:
            sigma *= 0.95
            
        if (i+1) % 10 == 0:
            print(f"Iter {i+1:2d} | Fit: {current_result['fitness']:.4f} | ATP: {current_result['atp']:.4f} | MI: {current_result['mi']:.4f}")

    print("\n--- OPTIMIZED PROTOCOL FOR JOHN G. CRAMER III ---")
    for k, v in current_params.items():
        print(f"  {k:<20}: {v:.4f}")
    
    print(f"\nOUTCOME AT AGE 92.1:")
    print(f"  Brain ATP    : {current_result['atp']:.4f}")
    print(f"  Memory Index : {current_result['mi']:.4f}")
    
    # Save results
    res_dir = "how-to-live-much-longer/results/john_iii"
    os.makedirs(res_dir, exist_ok=True)
    with open(f"{res_dir}/john_iii_summary.txt", "w") as f:
        f.write("JOHN G. CRAMER III (AGE 62.1 -> 92.1) REPORT\n")
        f.write(f"Final ATP: {current_result['atp']:.4f}\n")
        f.write(f"Final MI: {current_result['mi']:.4f}\n")
        f.write(f"Red Light: {current_params['red_light']:.4f}\n")
        f.write(f"Yoga: {current_params['yoga']:.4f}\n")

if __name__ == "__main__":
    run_optimization()
