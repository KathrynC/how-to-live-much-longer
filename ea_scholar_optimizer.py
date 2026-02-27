"""ea_scholar_optimizer.py

Uses a OnePlusLambda Evolutionary Strategy to find the optimal intervention 
protocol for a 'High-Engagement Scholar' (Graduate Student) using the 
33-state Universal Human Digital Twin.

Goal: Maximize Terminal Memory Index (100-year horizon) while 
      minimizing the "Biological Cost" (Brain ROS).
"""

import numpy as np
import time
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

# ── Fitness Function ─────────────────────────────────────────────────────────

def evaluate_scholar_fitness(params):
    """
    Evaluates an intervention protocol for a high-intensity scholar.
    Profile: 63.83yo Female, APOE4 Het.
    """
    
    # 1. Setup Patient
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83, # 63 years, 10 months
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.8,
        "fructose_intake": params.get("fructose_intake", 0.5)
    })
    
    # 2. Setup Intervention
    intervention = dict(DEFAULT_INTERVENTION)
    intervention.update({
        "exercise_level": params.get("exercise_level", 0.0),
        "fasting_regimen": params.get("fasting_regimen", 0.0),
        "nad_supplement": params.get("nad_supplement", 0.0),
        "probiotic_intensity": params.get("probiotic_intensity", 0.0),
        "therapy_intensity": params.get("therapy_intensity", 0.0),
        "alcohol_intake": 0.8 # Added: High initial alcohol burden
    })
    
    sleep_q = params.get("sleep_quality", 0.5)
    
    # 3. Simulate (30-year window)
    res = unified_simulate(patient=patient, intervention=intervention, sleep_quality=sleep_q, sim_years=30.0)
    
    # 4. Compute Fitness
    # Primary: Memory Index (Goal: Maximize)
    final_mi = res["memory_index"][-1]
    
    # Penalty: Brain ROS (Goal: Minimize)
    final_ros = res["states"][-1, 3]
    
    # Weighted Fitness: Maximize MI, Penalize ROS exceeding baseline (approx 0.5)
    ros_penalty = 0.1 * max(0, final_ros - 0.5)
    fitness = final_mi - ros_penalty
    
    return {
        "fitness": fitness,
        "memory_index": final_mi,
        "brain_ros": final_ros,
        "insulin_sens": res["states"][-1, 32]
    }

# ── Evolutionary Strategy (1 + 5) ───────────────────────────────────────────

def run_optimization(budget=100):
    print(f"Starting 'Perfect Scholar' Optimization (Budget: {budget})...")
    print()
    
    # Search Space
    param_names = ["sleep_quality", "exercise_level", "fasting_regimen", 
                   "nad_supplement", "probiotic_intensity", "therapy_intensity", 
                   "fructose_intake"]
    
    # Initial "Average" Protocol
    current_params = {name: 0.5 for name in param_names}
    current_result = evaluate_scholar_fitness(current_params)
    
    sigma = 0.15 # Step size
    
    for i in range(budget):
        # Generate 5 candidates (Lambda=5)
        candidates = []
        for _ in range(5):
            candidate = {}
            for name in param_names:
                # Mutate: add gaussian noise
                val = current_params[name] + np.random.normal(0, sigma)
                candidate[name] = np.clip(val, 0.0, 1.0)
            
            # Diet: Fructose intake is better when LOW
            # No special logic needed, EA will find it.
            
            candidates.append((candidate, evaluate_scholar_fitness(candidate)))
            
        # Select best candidate
        candidates.sort(key=lambda x: x[1]["fitness"], reverse=True)
        best_candidate, best_res = candidates[0]
        
        if best_res["fitness"] > current_result["fitness"]:
            current_params = best_candidate
            current_result = best_res
            # Success! Increase sigma slightly
            sigma *= 1.1
        else:
            # Failure. Decrease sigma
            sigma *= 0.95
            
        sigma = np.clip(sigma, 0.01, 0.3)
        
        if (i+1) % 10 == 0:
            print(f"Gen {i+1:3d} | Fitness: {current_result['fitness']:.4f} | MI: {current_result['memory_index']:.4f} | ROS: {current_result['brain_ros']:.4f}")

    print()
    print("="*60)
    print("OPTIMAL 'PERFECT SCHOLAR' PROTOCOL FOUND")
    print("="*60)
    for name in param_names:
        print(f"  {name:<20}: {current_params[name]:.4f}")
    
    print("-" * 60)
    print(f"  Final Memory Index: {current_result['memory_index']:.4f}")
    print(f"  Final Brain ROS:    {current_result['brain_ros']:.4f}")
    print(f"  Insulin Sensitivity: {current_result['insulin_sens']:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_optimization(budget=100)
