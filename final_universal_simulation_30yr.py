"""final_universal_simulation_30yr.py

The Definitive 30-Year Simulation for a 63.83yo female scholar.
Integrates ALL optimized protocols:
- Scholar Engagement (1.0)
- Integrated Social (Scholar-Mentor Trinity)
- Full Farm Animal Synergy
- Mediterranean-Keto Hybrid Diet
- 80/20 Exercise Hybrid
- Clean Sweep Medication (Rapa/Seno)
- Scholar's Power Combo Supplements (LifeCode + NR + Apigenin)
- Cliff Rescue Transplant (at 60% het)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_definitive_sim():
    print("Running THE FINAL UNIVERSAL SIMULATION (30-Year Horizon)...")
    
    # 1. Profile
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.5,
        "fructose_intake": 0.1 # Metabolic guard
    })
    
    # 2. THE MASTER PROTOCOL
    intervention_updates = {
        # Social & Animal
        "social_protocol": "integrated",
        "animal_protocol": "full_farm",
        
        # Diet & Exercise
        "diet_type": "mediterranean",
        "exercise_type": "aerobic",
        "exercise_level": 0.9, # Consistent high-intensity
        "fasting_regimen": 0.7, # Keto-hybrid guard
        
        # Clinical & Supplements
        "alcohol_intake": 0.05, # Negligible
        "nr_dose": 0.8,
        "magnesium_dose": 0.8, # Power Combo proxies
        "ala_dose": 0.8,
        "rapamycin_dose": 0.4, # Clean Sweep primary
        "senolytic_dose": 0.3
    }
    
    intervention = {**DEFAULT_INTERVENTION, **intervention_updates}
    
    sleep_q = 0.85 # Protected window
    horizon = 30.0
    
    # 3. RUN SIMULATION
    res = unified_simulate(
        patient=patient, 
        intervention=intervention, 
        sleep_quality=sleep_q, 
        sim_years=horizon,
        transplant_protocol="rescue" # THE WINNER
    )
    
    # 4. REPORT RESULTS
    print("\nUNIVERSAL RESULTS (Age 63.8 -> 93.8):")
    print(f"{'Age':<10} | {'ATP':<10} | {'Brain ROS':<10} | {'Het':<10} | {'Memory Index'}")
    print("-" * 65)
    
    indices = np.linspace(0, len(res["time"])-1, 7, dtype=int)
    for idx in indices:
        age = 63.83 + res["time"][idx]
        atp = res["states"][idx, 2]
        ros = res["states"][idx, 3]
        total = res["states"][idx, 0] + res["states"][idx, 1] + res["states"][idx, 7]
        het = (res["states"][idx, 1] + res["states"][idx, 7]) / max(total, 1e-12)
        mi = res["memory_index"][idx]
        print(f"{age:<10.1f} | {atp:<10.4f} | {ros:<10.4f} | {het:<10.4f} | {mi:.4f}")

if __name__ == "__main__":
    run_definitive_sim()
