"""ea_westport_household_optimizer.py

A local-only, deep optimization script for the Westport Household.
Models:
- Subject A: Kathryn (63.83, Scholar, APOE4 Het, Rapamycin)
- Subject B: Peter (28.0, Aspiring Biologist, EDS, APOE4 Het)

NOTE: This file is ignored by git.
"""

import numpy as np
import os
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def evaluate_westport_fitness(params):
    # ── SHARED PARAMETERS ──
    shared_fructose = params.get("shared_fructose", 0.0) # Assume optimal low-fructose
    diet_type = "mediterranean"
    
    # ── SUBJECT A: KATHRYN (Age 63.83, Scholar) ──
    a_p = dict(DEFAULT_PATIENT)
    a_p.update({
        "baseline_age": 63.83,
        "profile": "scholar", 
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "fructose_intake": shared_fructose,
        "post_viral_load": 0.0 # Never COVID
    })
    
    a_i = dict(DEFAULT_INTERVENTION)
    a_i.update({
        "diet_type": diet_type,
        "exercise_level": params.get("a_exercise", 0.5),
        "sleep_quality": params.get("a_sleep", 0.8),
        "rapamycin_dose": 0.4,
        "nad_supplement": 0.8,
        "hrt_therapy": 1.0,
        "red_light_therapy": params.get("shared_red_light", 0.5),
        "sauna_use": params.get("shared_sauna", 0.5),
        "restorative_yoga": params.get("shared_yoga", 0.5)
    })
    
    res_a = unified_simulate(patient=a_p, intervention=a_i, sim_years=30.0)
    
    # ── SUBJECT B: PETER (Age 28.0, Aspiring Biologist/EDS) ──
    b_p = dict(DEFAULT_PATIENT)
    b_p.update({
        "baseline_age": 28.0,
        "profile": "eds", # 1.4x EDS drag
        "apoe_genotype": "apoe4_het",
        "sex": "male",
        "fructose_intake": shared_fructose,
        "intellectual_engagement": 0.8, # Aspiring Biologist
        "post_viral_load": 0.0 # Never COVID
    })
    
    b_i = dict(DEFAULT_INTERVENTION)
    b_i.update({
        "diet_type": diet_type,
        "exercise_level": params.get("b_exercise", 0.5),
        "sleep_quality": params.get("b_sleep", 0.8),
        "nr_dose": 0.9, # To bridge EDS drag
        "magnesium_dose": 1.0,
        "animal_protocol": "full_farm", # Shared household resource
        "red_light_therapy": params.get("shared_red_light", 0.5),
        "sauna_use": params.get("shared_sauna", 0.5),
        "restorative_yoga": params.get("shared_yoga", 0.5)
    })
    
    res_b = unified_simulate(patient=b_p, intervention=b_i, sim_years=30.0)
    
    # ── AGGREGATE FITNESS ──
    mi_a = res_a["memory_index"][-1]
    mi_b = res_b["memory_index"][-1]
    
    a_atp = res_a["states"][-1, 2]
    b_atp = res_b["states"][-1, 2]
    
    # PENALTY: Energy Stability is Paramount
    a_atp_penalty = max(0, 0.90 - a_atp) ** 2 * 20
    b_atp_penalty = max(0, 0.90 - b_atp) ** 2 * 20
    
    fitness = (mi_a + mi_b) / 2.0 - a_atp_penalty - b_atp_penalty
    
    return {
        "fitness": fitness,
        "mi_a": mi_a,
        "mi_b": mi_b,
        "a_atp": a_atp,
        "b_atp": b_atp,
        "a_ros": res_a["states"][-1, 3],
        "b_ros": res_b["states"][-1, 3]
    }

import json
import pandas as pd

def run_optimization(budget=15):
    print(f"Deep Optimization: Westport Household (Kathryn & Peter)...")
    param_names = [
        "a_exercise", "a_sleep", "b_exercise", "b_sleep", 
        "shared_fructose", "shared_red_light", "shared_sauna", "shared_yoga"
    ]
    current_params = {name: 0.7 for name in param_names}
    current_result = evaluate_westport_fitness(current_params)
    
    sigma = 0.1
    for i in range(budget):
        candidates = []
        for _ in range(3):
            candidate = {name: np.clip(current_params[name] + np.random.normal(0, sigma), 0.05, 1.0) for name in param_names}
            candidates.append((candidate, evaluate_westport_fitness(candidate)))
        
        candidates.sort(key=lambda x: x[1]["fitness"], reverse=True)
        if candidates[0][1]["fitness"] > current_result["fitness"]:
            current_params, current_result = candidates[0]
            sigma *= 1.05
        else:
            sigma *= 0.95
            
        if (i+1) % 10 == 0:
            print(f"Iter {i+1:3d} | Fit: {current_result['fitness']:.4f} | K_ATP: {current_result['a_atp']:.4f} | P_ATP: {current_result['b_atp']:.4f}")

    print("\nOPTIMIZED WESTPORT PROTOCOL (30-Year Projection):")
    for k, v in current_params.items():
        print(f"  {k:<20}: {v:.4f}")
    
    # -- PERSISTENCE: Save Final Results --
    res_dir = "how-to-live-much-longer/results/westport"
    os.makedirs(res_dir, exist_ok=True)
    
    # Save Protocol
    with open(f"{res_dir}/westport_final_protocol.json", "w") as f:
        json.dump(current_params, f, indent=4)
    
    # Save a Summary Report
    with open(f"{res_dir}/westport_summary.txt", "w") as f:
        f.write(f"WESTPORT HOUSEHOLD OPTIMIZATION REPORT\n")
        f.write(f"Budget: {budget} iterations\n\n")
        f.write(f"Final Brain ATP (Kathryn): {current_result['a_atp']:.4f}\n")
        f.write(f"Final Memory Index (Kathryn): {current_result['mi_a']:.4f}\n")
        f.write(f"Final Brain ATP (Peter): {current_result['b_atp']:.4f}\n")
        f.write(f"Final Memory Index (Peter): {current_result['mi_b']:.4f}\n")
    
    print(f"\nResults saved to {res_dir}/")

if __name__ == "__main__":
    run_optimization()
