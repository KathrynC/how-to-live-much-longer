"""ea_john_jr_optimizer.py

Deep optimization for Subject H: John Cramer Jr (Age 91).
Investigates the impact of Red Light, Sauna, and Yoga on late-life 
mitochondrial stability.

NOTE: This file is ignored by git.
"""

import numpy as np
import os
import json
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def evaluate_john_fitness(params):
    # ── SUBJECT H: JOHN CRAMER JR (Age 91) ──
    h_p = dict(DEFAULT_PATIENT)
    h_p.update({
        "baseline_age": 91.0,
        "profile": "scholar", 
        "apoe_genotype": "apoe4_het", # Assume familial APOE4
        "sex": "male",
        "intellectual_engagement": 0.8,
        "baseline_heteroplasmy": 0.45, # High starting damage at 91
        "post_viral_load": 0.0 # Unknown, assume 0 for now
    })
    
    h_i = dict(DEFAULT_INTERVENTION)
    h_i.update({
        "diet_type": "mediterranean",
        "exercise_level": params.get("exercise", 0.3),
        "sleep_quality": params.get("sleep", 0.7),
        "nr_dose": 0.9,
        "magnesium_dose": 1.0,
        "red_light_therapy": params.get("red_light", 0.8),
        "sauna_use": params.get("sauna", 0.5),
        "restorative_yoga": params.get("yoga", 0.8),
        "transplant_rate": params.get("transplant", 0.0) # Investigate if transplant is needed
    })
    
    # Simulate 10 years (Age 91 -> 101)
    res = unified_simulate(patient=h_p, intervention=h_i, sim_years=10.0)
    
    atp = res["states"][-1, 2]
    mi = res["memory_index"][-1]
    ros = res["states"][-1, 3]
    
    # Fitness: Keep ATP > 0.70 and maximize Memory Index
    atp_penalty = max(0, 0.70 - atp) ** 2 * 100
    fitness = (atp * 10) + (mi * 5) - atp_penalty - (ros * 2)
    
    return {"fitness": fitness, "atp": atp, "mi": mi, "ros": ros}

def run_optimization(budget=30):
    print(f"Optimizing Protocol for John Cramer Jr (Age 91 -> 101)...")
    param_names = ["exercise", "sleep", "red_light", "sauna", "yoga", "transplant"]
    current_params = {name: 0.5 for name in param_names}
    current_params["transplant"] = 0.0 # Start without transplant
    current_result = evaluate_john_fitness(current_params)
    
    sigma = 0.1
    for i in range(budget):
        candidates = []
        for _ in range(5):
            candidate = {name: np.clip(current_params[name] + np.random.normal(0, sigma), 0.0, 1.0) for name in param_names}
            candidates.append((candidate, evaluate_john_fitness(candidate)))
        
        candidates.sort(key=lambda x: x[1]["fitness"], reverse=True)
        if candidates[0][1]["fitness"] > current_result["fitness"]:
            current_params, current_result = candidates[0]
            sigma *= 1.05
        else:
            sigma *= 0.95
            
        if (i+1) % 5 == 0:
            print(f"Iter {i+1:2d} | Fit: {current_result['fitness']:.4f} | ATP: {current_result['atp']:.4f} | MI: {current_result['mi']:.4f}")

    print("\n--- OPTIMIZED PROTOCOL FOR JOHN CRAMER JR ---")
    for k, v in current_params.items():
        print(f"  {k:<20}: {v:.4f}")
    
    print(f"\nOUTCOME AT AGE 101:")
    print(f"  Brain ATP    : {current_result['atp']:.4f}")
    print(f"  Memory Index : {current_result['mi']:.4f}")
    print(f"  ROS Level    : {current_result['ros']:.4f}")
    
    # Save results
    res_dir = "how-to-live-much-longer/results/john_jr"
    os.makedirs(res_dir, exist_ok=True)
    with open(f"{res_dir}/john_jr_summary.txt", "w") as f:
        f.write("JOHN CRAMER JR (AGE 91 -> 101) REPORT\n")
        f.write(f"Final ATP: {current_result['atp']:.4f}\n")
        f.write(f"Final MI: {current_result['mi']:.4f}\n")
        f.write(f"Transplant Rate: {current_params['transplant']:.4f}\n")
        f.write(f"Red Light: {current_params['red_light']:.4f}\n")
        f.write(f"Yoga: {current_params['yoga']:.4f}\n")

if __name__ == "__main__":
    run_optimization()
