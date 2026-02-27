"""scholar_cliff_rejuvenation_sim.py

Definitive 20-year simulation for a 63.83yo female scholar.
Features:
1. Long-term forecast (20 years)
2. Mitochondrial Transplant Rejuvenation (Year 5)
3. Lifestyle Step-up (Exercise 0.8 -> 1.0 at Year 1)
"""

import numpy as np
from unified_brain_model import unified_derivatives, initial_unified_state, memory_index, DEFAULT_PATIENT, DEFAULT_INTERVENTION, N_UNIFIED_STATES

def run_definitive_sim():
    print("Running 20-Year 'Cliff Forecast & Rejuvenation' Simulation...")
    
    # 1. Setup Patient
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.5 # Assuming stress is partially managed
    })
    
    # 2. Setup Base Intervention
    base_intervention = dict(DEFAULT_INTERVENTION)
    base_intervention.update({
        "fasting_regimen": 0.8,
        "nad_supplement": 0.8,
        "therapy_intensity": 0.8,
        "alcohol_intake": 0.1 # Post-taper level
    })
    
    # 3. Integration Loop
    sim_years = 20.0 
    dt = 0.01 
    n_steps = int(sim_years / dt)
    
    state = initial_unified_state(patient)
    time_arr = np.linspace(0, sim_years, n_steps + 1)
    states = np.zeros((n_steps + 1, N_UNIFIED_STATES))
    states[0] = state
    
    mi_trace = np.zeros(n_steps + 1)
    mi_trace[0] = memory_index(state[10], state[8], state[11], state[12], state[13])
    
    for i in range(n_steps):
        t = time_arr[i]
        intervention = dict(base_intervention)
        
        # FEATURE 3: Exercise Step-up at Year 1
        if t < 1.0:
            intervention["exercise_level"] = 0.8
        else:
            intervention["exercise_level"] = 1.0
            
        # FEATURE 2: Mitochondrial Transplant at Year 5
        # (Transplant is a 'Rate' intervention in the ODE)
        if 5.0 <= t < 6.0: # 1-year intensive rejuvenation window
            intervention["transplant_rate"] = 0.8
        else:
            intervention["transplant_rate"] = 0.0
            
        # RK4 Step
        k1 = unified_derivatives(state, t, intervention, patient, sleep_quality=0.8)
        k2 = unified_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, intervention, patient, sleep_quality=0.8)
        k3 = unified_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, intervention, patient, sleep_quality=0.8)
        k4 = unified_derivatives(state + dt * k3, t + dt, intervention, patient, sleep_quality=0.8)
        
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Clamping
        state = np.maximum(state, 0.0)
        state[5] = min(state[5], 1.0)
        
        states[i+1] = state
        mi_trace[i+1] = memory_index(state[10], state[8], state[11], state[12], state[13])
        
    print(f"\n20-Year Forecast (Age {63.83:.1f} -> {63.83+20:.1f}):")
    print(f"{'Age':<10} | {'ATP':<10} | {'Brain ROS':<10} | {'Heteroplasmy':<15} | {'Memory Index'}")
    print("-" * 75)
    
    # Subsample results every 2.5 years
    indices_to_show = np.linspace(0, n_steps, 9, dtype=int)
    for idx in indices_to_show:
        age = 63.83 + time_arr[idx]
        atp = states[idx, 2]
        ros = states[idx, 3]
        # Total het = (N_del + N_pt) / (N_h + N_del + N_pt)
        total_copies = states[idx, 0] + states[idx, 1] + states[idx, 7]
        het = (states[idx, 1] + states[idx, 7]) / max(total_copies, 1e-12)
        
        rejuv_note = " [REJUV]" if 5.0 <= time_arr[idx] <= 6.0 else ""
        print(f"{age:<10.1f} | {atp:<10.4f} | {ros:<10.4f} | {het:<15.4f} | {mi_trace[idx]:.4f}{rejuv_note}")

if __name__ == "__main__":
    run_definitive_sim()
