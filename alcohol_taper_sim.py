"""alcohol_taper_sim.py

Simulates a 1-month rapid alcohol taper for a 63.83yo female scholar.
Uses the 33-state Universal Human Digital Twin to track recovery of 
Liver, GSH, and Brain ROS.
"""

import numpy as np
from unified_brain_model import unified_derivatives, initial_unified_state, memory_index, DEFAULT_PATIENT, DEFAULT_INTERVENTION, N_UNIFIED_STATES

def run_taper_sim():
    print("Running Rapid 1-Month Alcohol Taper Simulation...")
    
    # 1. Setup Patient
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.8
    })
    
    # 2. Setup Intervention (High initial stress, good recovery habits)
    base_intervention = dict(DEFAULT_INTERVENTION)
    base_intervention.update({
        "exercise_level": 0.8,
        "fasting_regimen": 0.8,
        "nad_supplement": 0.8,
        "therapy_intensity": 0.8
    })
    
    sleep_q = 0.8
    
    # 3. Integration Loop
    sim_years = 1.0 # Simulate 1 full year to see the 'tail' of recovery
    dt = 0.01 # ~3.65 days
    n_steps = int(sim_years / dt)
    
    state = initial_unified_state(patient)
    # Force initial state to high alcohol burden (if it were an ODE state, 
    # but it's an intervention parameter, so we handle it in the loop)
    
    time_arr = np.linspace(0, sim_years, n_steps + 1)
    states = np.zeros((n_steps + 1, N_UNIFIED_STATES))
    states[0] = state
    
    mi_trace = np.zeros(n_steps + 1)
    mi_trace[0] = memory_index(state[10], state[8], state[11], state[12], state[13])
    
    for i in range(n_steps):
        t = time_arr[i]
        
        # Rapid 1-month Taper (1 month = 1/12 of a year ~ 0.083 years)
        taper_horizon = 1.0 / 12.0
        if t < taper_horizon:
            # Linear taper from 0.8 to 0.1
            current_alc = 0.8 - (0.7 * (t / taper_horizon))
        else:
            current_alc = 0.1 # Maintenance level
            
        intervention = dict(base_intervention)
        intervention["alcohol_intake"] = current_alc
        
        # RK4 Step
        k1 = unified_derivatives(state, t, intervention, patient, sleep_q)
        k2 = unified_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, intervention, patient, sleep_q)
        k3 = unified_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, intervention, patient, sleep_q)
        k4 = unified_derivatives(state + dt * k3, t + dt, intervention, patient, sleep_q)
        
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Clamping
        state = np.maximum(state, 0.0)
        state[5] = min(state[5], 1.0)
        # ... (rest of clamping same as unified_simulate)
        
        states[i+1] = state
        mi_trace[i+1] = memory_index(state[10], state[8], state[11], state[12], state[13])
        
    print(f"\nTaper Results (Time: 0.0 -> 1.0 year):")
    print(f"{'Time (Months)':<15} | {'Alc Intake':<15} | {'Liver Health':<15} | {'Brain ROS':<15} | {'Memory Index'}")
    print("-" * 85)
    
    indices_to_show = [0, 2, 4, 8, 12, 24, 52, 100] # Subsample throughout the year
    for idx in indices_to_show:
        if idx >= len(time_arr): break
        t_months = time_arr[idx] * 12.0
        
        if time_arr[idx] < taper_horizon:
            alc = 0.8 - (0.7 * (time_arr[idx] / taper_horizon))
        else:
            alc = 0.1
            
        print(f"{t_months:<15.1f} | {alc:<15.2f} | {states[idx, 22]:<15.4f} | {states[idx, 3]:<15.4f} | {mi_trace[idx]:.4f}")

if __name__ == "__main__":
    run_taper_sim()
