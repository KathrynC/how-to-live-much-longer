"""long_term_sensitivity_audit.py

Performs a 20-year sensitivity analysis on the 33-state model 
specifically for the 63.83yo female scholar profile.
Compares: Sleep Quality, Mitochondrial Transplant, and Exercise Level.
"""

import numpy as np
from unified_brain_model import unified_derivatives, initial_unified_state, memory_index, DEFAULT_PATIENT, DEFAULT_INTERVENTION, N_UNIFIED_STATES

def run_sim(sleep_q, has_transplant, exercise_lvl):
    patient = dict(DEFAULT_PATIENT)
    patient.update({"baseline_age": 63.83, "apoe_genotype": "apoe4_het", "sex": "female", "intellectual_engagement": 1.0, "grief_intensity": 0.5})
    
    base_intervention = dict(DEFAULT_INTERVENTION)
    base_intervention.update({"fasting_regimen": 0.8, "nad_supplement": 0.8, "therapy_intensity": 0.8, "alcohol_intake": 0.1})
    
    sim_years = 20.0
    dt = 0.05 # Faster step for audit
    n_steps = int(sim_years / dt)
    state = initial_unified_state(patient)
    
    for i in range(n_steps):
        t = i * dt
        intervention = dict(base_intervention)
        intervention["exercise_level"] = exercise_lvl
        if has_transplant and 5.0 <= t < 6.0:
            intervention["transplant_rate"] = 0.8
        else:
            intervention["transplant_rate"] = 0.0
            
        # Integration
        k1 = unified_derivatives(state, t, intervention, patient, sleep_quality=sleep_q)
        state = state + dt * k1
        state = np.maximum(state, 0.0)
        state[5] = min(state[5], 1.0)
        
    return memory_index(state[10], state[8], state[11], state[12], state[13])

def run_audit():
    print("Running Long-Term Sensitivity Audit (20-Year Horizon)...")
    
    # Baseline: The "Sustainable Scholar" we just ran
    base_mi = run_sim(sleep_q=0.8, has_transplant=True, exercise_lvl=1.0)
    
    # 1. Effect of Sleep (Drop sleep to 0.5)
    sleep_impact = base_mi - run_sim(sleep_q=0.5, has_transplant=True, exercise_lvl=1.0)
    
    # 2. Effect of Rejuvenation (Remove Transplant)
    rejuv_impact = base_mi - run_sim(sleep_q=0.8, has_transplant=False, exercise_lvl=1.0)
    
    # 3. Effect of Exercise (Drop exercise to 0.2)
    exercise_impact = base_mi - run_sim(sleep_q=0.8, has_transplant=True, exercise_lvl=0.2)
    
    print(f"\nResults (Delta in 20-Year Memory Index):")
    print(f"{'Factor':<25} | {'Impact (Δ MI)':<15}")
    print("-" * 45)
    print(f"{'Sleep Quality':<25} | {sleep_impact:.4f}")
    print(f"{'Mito Transplant':<25} | {rejuv_impact:.4f}")
    print(f"{'Exercise Level':<25} | {exercise_impact:.4f}")

if __name__ == "__main__":
    run_audit()
