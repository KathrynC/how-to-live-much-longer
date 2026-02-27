"""ea_household_optimizer_westport.py

Optimizes the Westport Household (User + Peter) as a single unit.
Uses the 35-state Omni-Twin model.
Goal: Maximize combined Memory Index while honoring shared resources.
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def evaluate_household_fitness(params):
    # ── SHARED PARAMETERS ──
    diet = "mediterranean" # Established winner
    soc = "integrated"     # Established winner
    ani = "full_farm"      # Established winner
    fructose = params.get("shared_fructose", 0.1)
    
    # ── 1. USER SIMULATION (Scholar, 63.83) ──
    user_p = dict(DEFAULT_PATIENT)
    user_p.update({
        "baseline_age": 63.83, "apoe_genotype": "apoe4_het", "sex": "female",
        "intellectual_engagement": 1.0, "grief_intensity": 0.5, "fructose_intake": fructose
    })
    user_i = {
        **DEFAULT_INTERVENTION,
        "diet_type": diet, "social_protocol": soc, "animal_protocol": ani,
        "exercise_level": params.get("user_exercise", 0.8),
        "sleep_quality": params.get("user_sleep", 0.8),
        "nr_dose": 0.8, "rapamycin_dose": 0.4
    }
    res_user = unified_simulate(patient=user_p, intervention=user_i, sim_years=20.0, transplant_protocol="rescue")
    
    # ── 2. PETER SIMULATION (Biologist, 28.0) ──
    peter_p = dict(DEFAULT_PATIENT)
    peter_p.update({
        "baseline_age": 28.0, "profile": "biologist", "apoe_genotype": "apoe4_het", "sex": "male",
        "intellectual_engagement": 0.8, "grief_intensity": 0.7, "fructose_intake": fructose,
        "deep_sleep_genetic_penalty": 0.7
    })
    peter_i = {
        **DEFAULT_INTERVENTION,
        "diet_type": diet, "social_protocol": soc, "animal_protocol": ani,
        "exercise_type": "resistance",
        "exercise_level": params.get("peter_exercise", 0.6),
        "sleep_quality": params.get("peter_sleep", 0.8),
        "nr_dose": 0.9, "magnesium_dose": 1.0
    }
    res_peter = unified_simulate(patient=peter_p, intervention=peter_i, sim_years=20.0)
    
    # ── AGGREGATE FITNESS ──
    user_mi = res_user["memory_index"][-1]
    peter_mi = res_peter["memory_index"][-1]
    
    # Equal weight for household harmony
    fitness = (user_mi + peter_mi) / 2.0
    
    return {
        "fitness": fitness,
        "user_mi": user_mi,
        "peter_mi": peter_mi,
        "user_ros": res_user["states"][-1, 3],
        "peter_ros": res_peter["states"][-1, 3]
    }

def run_optimization(budget=100):
    print(f"Optimizing Westport Household (Budget: {budget})...\n")
    param_names = ["shared_fructose", "user_exercise", "user_sleep", "peter_exercise", "peter_sleep"]
    current_params = {name: 0.5 for name in param_names}
    current_result = evaluate_household_fitness(current_params)
    
    sigma = 0.1
    for i in range(budget):
        candidates = []
        for _ in range(5):
            candidate = {name: np.clip(current_params[name] + np.random.normal(0, sigma), 0.0, 1.0) for name in param_names}
            candidates.append((candidate, evaluate_household_fitness(candidate)))
        
        candidates.sort(key=lambda x: x[1]["fitness"], reverse=True)
        if candidates[0][1]["fitness"] > current_result["fitness"]:
            current_params, current_result = candidates[0]
            sigma *= 1.1
        else:
            sigma *= 0.95
            
        if (i+1) % 20 == 0:
            print(f"Trial {i+1:3d} | Fitness: {current_result['fitness']:.4f} | User MI: {current_result['user_mi']:.4f} | Peter MI: {current_result['peter_mi']:.4f}")

    print("\nOPTIMAL WESTPORT CONFIGURATION:")
    for k, v in current_params.items():
        print(f"  {k:<20}: {v:.4f}")
    return current_params, current_result

if __name__ == "__main__":
    run_optimization(budget=100)
