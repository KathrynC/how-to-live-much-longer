"""selena_rescue_sim_20yr.py

Evaluates the 'Donor Rescue' protocol for Selena Shea:
- 26yo Female, APOE4 Het, probable EDS
- Starting condition: Heavy Drinker (0.8), Intensive Cannabis (0.8)
- Rescue path: 1-month Taper + Sheltie Pulse + High-dose Magnesium/NR.
"""

import numpy as np
from unified_brain_model import unified_derivatives, initial_unified_state, memory_index, DEFAULT_PATIENT, DEFAULT_INTERVENTION, N_UNIFIED_STATES

def run_selena_rescue():
    print("Running 20-Year 'Donor Rescue' Simulation (Selena Shea)...")
    
    # 1. Selena's Profile
    p = dict(DEFAULT_PATIENT)
    p.update({
        "baseline_age": 26.0,
        "profile": "biologist",
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 0.8,
        "grief_intensity": 0.5,
        "deep_sleep_genetic_penalty": 0.8 # Conservatively high
    })
    
    # 2. Intervention Base (The Rescue Plan)
    # High-intensity habits + Sheltie Agility
    base_i = dict(DEFAULT_INTERVENTION)
    base_i.update({
        "diet_type": "mediterranean",
        "exercise_type": "hiit", # Agility proxy
        "exercise_level": 0.9,
        "social_protocol": "integrated",
        "animal_protocol": "active",
        "nr_dose": 0.9,
        "magnesium_dose": 1.0,
        "hrt_therapy": 0.0 # Too young for HRT
    })
    
    sleep_q = 0.85
    sim_years = 20.0
    dt = 0.01
    n_steps = int(sim_years / dt)
    
    state = initial_unified_state(p)
    time_arr = np.linspace(0, sim_years, n_steps + 1)
    states = np.zeros((n_steps + 1, N_UNIFIED_STATES))
    states[0] = state
    
    mi_trace = np.zeros(n_steps + 1)
    mi_trace[0] = memory_index(state[10], state[8], state[11], state[12], state[13])
    
    for i in range(n_steps):
        t = time_arr[i]
        intervention = dict(base_i)
        
        # RAPID TAPER (1 Month)
        taper_h = 1.0/12.0
        if t < taper_h:
            # Linear Taper from 0.8 down
            current_burden = 0.8 - (0.7 * (t / taper_h))
        else:
            current_burden = 0.1 # Sustainable maintenance
            
        intervention["alcohol_intake"] = current_burden
        intervention["cannabis_use"] = current_burden
        
        # RK4
        k1 = unified_derivatives(state, t, intervention, p, sleep_q)
        k2 = unified_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, intervention, p, sleep_q)
        k3 = unified_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, intervention, p, sleep_q)
        k4 = unified_derivatives(state + dt * k3, t + dt, intervention, p, sleep_q)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        state = np.maximum(state, 0.0)
        state[5] = min(state[5], 1.0)
        
        states[i+1] = state
        mi_trace[i+1] = memory_index(state[10], state[8], state[11], state[12], state[13])
        
    print(f"\nSelena's Year 20 Rescue Results (Age 46.0):")
    print(f"{'Metric':<25} | {'Value'}")
    print("-" * 45)
    
    # 3. Report
    def get_het(s):
        total = s[0] + s[1] + s[7]
        return (s[1] + s[7]) / max(total, 1e-12)

    print(f"{'Memory Index':<25} | {mi_trace[-1]:.4f}")
    print(f"{'Heteroplasmy':<25} | {get_het(states[-1]):.4f}")
    print(f"{'Brain ROS':<25} | {states[-1, 3]:.4f}")
    print(f"{'Synaptic Strength':<25} | {states[-1, 10]:.4f}")
    print(f"{'Liver GSH Pool':<25} | {states[-1, 23]:.4f}")

if __name__ == "__main__":
    run_selena_rescue()
