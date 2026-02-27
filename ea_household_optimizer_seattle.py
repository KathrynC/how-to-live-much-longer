"""ea_household_optimizer_seattle.py

Optimizes the Seattle Household (John + Shea + John III).
Uses the 35-state Omni-Twin model.
Goal: Maximize primary Peak (John) while securing the Donor (Shea) and Rescue (John III).
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def evaluate_seattle_fitness(params):
    # ── SHARED PARAMETERS ──
    diet = "mediterranean"
    fructose = params.get("shared_fructose", 0.1)
    
    # ── 1. JOHN SIMULATION (Scholar, 91.0) ──
    john_p = dict(DEFAULT_PATIENT)
    john_p.update({
        "baseline_age": 91.0, "apoe_genotype": "apoe4_het", "sex": "male",
        "intellectual_engagement": 1.0, "grief_intensity": 0.5, "fructose_intake": fructose
    })
    john_i = {
        **DEFAULT_INTERVENTION,
        "diet_type": diet, "coffee_intake": 0.8,
        "exercise_level": 0.3, "sleep_quality": 0.8,
        "nr_dose": 0.8, "magnesium_dose": 0.8
    }
    res_john = unified_simulate(patient=john_p, intervention=john_i, sim_years=10.0, transplant_protocol="rescue")
    
    # ── 2. SHEA SIMULATION (Donor, 26.0) ──
    shea_p = dict(DEFAULT_PATIENT)
    shea_p.update({
        "baseline_age": 26.0, "profile": "biologist", "apoe_genotype": "apoe4_het", "sex": "female",
        "intellectual_engagement": 0.8, "grief_intensity": 0.3, "fructose_intake": fructose
    })
    shea_i = {
        **DEFAULT_INTERVENTION,
        "diet_type": diet, "exercise_level": params.get("shea_exercise", 0.6),
        "sleep_quality": params.get("shea_sleep", 0.8),
        "nr_dose": 0.5, "magnesium_dose": 0.8
    }
    res_shea = unified_simulate(patient=shea_p, intervention=shea_i, sim_years=10.0)
    
    # ── 3. JOHN III SIMULATION (Rescue, 61.0) ──
    john3_p = dict(DEFAULT_PATIENT)
    john3_p.update({
        "baseline_age": 61.0, "apoe_genotype": "apoe4_het", "sex": "male",
        "intellectual_engagement": 0.7, "grief_intensity": 0.8, "fructose_intake": fructose
    })
    alc_lvl = params.get("john3_alcohol", 0.8)
    john3_i = {
        **DEFAULT_INTERVENTION,
        "diet_type": diet, "alcohol_intake": alc_lvl,
        "exercise_level": params.get("john3_exercise", 0.4),
        "sleep_quality": 0.6,
        "nr_dose": 0.8, "magnesium_dose": 0.8
    }
    res_john3 = unified_simulate(patient=john3_p, intervention=john3_i, sim_years=10.0)
    
    # ── AGGREGATE FITNESS ──
    j_mi = res_john["memory_index"][-1]
    s_mi = res_shea["memory_index"][-1]
    j3_mi = res_john3["memory_index"][-1]
    
    fitness = 0.5 * j_mi + 0.25 * s_mi + 0.25 * j3_mi
    
    return {
        "fitness": fitness,
        "john_mi": j_mi,
        "shea_mi": s_mi,
        "john3_mi": j3_mi,
        "john3_ros": res_john3["states"][-1, 3]
    }

def run_optimization(budget=100):
    print(f"Optimizing Seattle Household (Budget: {budget})...\n")
    param_names = ["shared_fructose", "shea_exercise", "shea_sleep", "john3_exercise", "john3_alcohol"]
    current_params = {name: 0.5 for name in param_names}
    current_result = evaluate_seattle_fitness(current_params)
    
    sigma = 0.1
    for i in range(budget):
        candidates = []
        for _ in range(5):
            candidate = {name: np.clip(current_params[name] + np.random.normal(0, sigma), 0.0, 1.0) for name in param_names}
            candidates.append((candidate, evaluate_seattle_fitness(candidate)))
        
        candidates.sort(key=lambda x: x[1]["fitness"], reverse=True)
        if candidates[0][1]["fitness"] > current_result["fitness"]:
            current_params, current_result = candidates[0]
            sigma *= 1.1
        else:
            sigma *= 0.95
            
        if (i+1) % 20 == 0:
            print(f"Trial {i+1:3d} | Fitness: {current_result['fitness']:.4f} | John MI: {current_result['john_mi']:.4f} | John3 MI: {current_result['john3_mi']:.4f}")

    print("\nOPTIMAL SEATTLE CONFIGURATION:")
    for k, v in current_params.items():
        print(f"  {k:<20}: {v:.4f}")
    return current_params, current_result

if __name__ == "__main__":
    run_optimization(budget=100)
