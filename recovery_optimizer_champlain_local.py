"""recovery_optimizer_champlain_local.py

Optimizes the 'Recovery Velocity' for the Champlain Household.
Goal: Find the protocol that returns subjects to baseline ATP fastest after a shock.

NOTE: This file is ignored by git.
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def evaluate_recovery_fitness(params, subject="Jasper"):
    # ── PATIENT SETUP ──
    if subject == "Ratio":
        p = dict(DEFAULT_PATIENT); p.update({"baseline_age": 23.0, "profile": "artist", "seizure_vulnerability": 0.8})
        shock_type = "sleep_crisis"
    else:
        p = dict(DEFAULT_PATIENT); p.update({"baseline_age": 24.0, "profile": "biologist", "sex": "male"})
        shock_type = "preschool_spike"
        
    # ── INTERVENTION SETUP (Baseline) ──
    i = dict(DEFAULT_INTERVENTION)
    i.update({
        "exercise_level": params.get("exercise", 0.5),
        "sleep_quality": params.get("sleep", 0.85),
        "nr_dose": params.get("nr", 0.5),
        "magnesium_dose": params.get("mag", 0.5),
        "rapamycin_dose": params.get("rapa", 0.2)
    })
    
    # 1. Measure shock depth (2-week shock)
    p_shock = dict(p)
    i_shock = dict(i)
    if shock_type == "preschool_spike":
        p_shock["inflammation_level"] = 0.8
    else:
        i_shock["sleep_quality"] = 0.2
        
    res_shock = unified_simulate(patient=p_shock, intervention=i_shock, sim_years=0.04)
    atp_at_shock_end = res_shock["states"][-1, 2]
    
    # 2. Measure recovery (3-month window)
    res_rec = unified_simulate(patient=p, intervention=i, sim_years=0.25)
    atp_at_rec_end = res_rec["states"][-1, 2]
    
    # 3. Fitness = Recovery Velocity (increase in ATP over 3 months)
    recovery_velocity = (atp_at_rec_end - atp_at_shock_end)
    
    # Penalty: If shock end ATP is too low (crisis), penalize
    penalty = 0.0
    if atp_at_shock_end < 0.80:
        penalty = (0.80 - atp_at_shock_end) * 10
        
    return recovery_velocity - penalty

def run_recovery_ea(subject="Jasper"):
    print(f"Optimizing Recovery Window for {subject}...")
    param_names = ["exercise", "sleep", "nr", "mag", "rapa"]
    current_params = {name: 0.5 for name in param_names}
    
    sigma = 0.1
    best_fitness = -999
    
    for i in range(30):
        best_cand = None
        for _ in range(5):
            candidate = {name: np.clip(current_params[name] + np.random.normal(0, sigma), 0.0, 1.0) for name in param_names}
            fitness = evaluate_recovery_fitness(candidate, subject=subject)
            if best_cand is None or fitness > best_cand[1]:
                best_cand = (candidate, fitness)
        
        if best_cand[1] > best_fitness:
            current_params, best_fitness = best_cand
            sigma *= 1.05
        else:
            sigma *= 0.95
            
    print(f"  Optimized {subject} Protocol:")
    for k, v in current_params.items():
        print(f"    {k:<10}: {v:.4f}")
    print(f"  Recovery Fitness (dV/dt): {best_fitness:.4f}\n")
    return current_params

if __name__ == "__main__":
    p_ratio = run_recovery_ea("Ratio")
    p_jasper = run_recovery_ea("Jasper")
