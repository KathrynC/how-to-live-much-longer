"""household_omni_twin_30yr.py

THE DEFINITIVE 30-YEAR FORECAST for the User and Peter.
Integrates ALL optimized protocols using the 35-state Omni-Twin model.
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_household_forecast():
    print("Running THE DEFINITIVE 30-YEAR HOUSEHOLD FORECAST (Omni-Twin)...")
    
    # ── 1. USER PROFILE (The Sustainable Scholar) ──
    user_patient = dict(DEFAULT_PATIENT)
    user_patient.update({
        "baseline_age": 63.83, "apoe_genotype": "apoe4_het", "sex": "female",
        "intellectual_engagement": 1.0, "grief_intensity": 0.5, "fructose_intake": 0.1
    })
    user_intervention = {
        **DEFAULT_INTERVENTION,
        "social_protocol": "integrated", "animal_protocol": "full_farm",
        "diet_type": "mediterranean", "exercise_type": "aerobic", "exercise_level": 0.9,
        "fasting_regimen": 0.7, "alcohol_intake": 0.05, "nr_dose": 0.8,
        "magnesium_dose": 0.8, "ala_dose": 0.8, "rapamycin_dose": 0.4, 
        "senolytic_dose": 0.3, "coffee_intake": 0.8, "hrt_therapy": 0.8 # PHASE 7 additions
    }
    
    # ── 2. PETER PROFILE (The Wildlife Biologist) ──
    peter_patient = dict(DEFAULT_PATIENT)
    peter_patient.update({
        "baseline_age": 28.0, "profile": "biologist", "apoe_genotype": "apoe4_het", "sex": "male",
        "intellectual_engagement": 0.8, "grief_intensity": 0.7, "social_support": 0.8,
        "deep_sleep_genetic_penalty": 0.7 # 23andMe recalibration
    })
    peter_intervention = {
        **DEFAULT_INTERVENTION,
        "social_protocol": "integrated", "animal_protocol": "full_farm",
        "diet_type": "mediterranean", "exercise_type": "resistance", "exercise_level": 0.6,
        "fasting_regimen": 0.5, "alcohol_intake": 0.0, "nr_dose": 0.9,
        "magnesium_dose": 1.0, # High dose for sleep penalty
        "coffee_intake": 0.5
    }
    
    horizon = 30.0
    
    # ── 3. SIMULATE ──
    print("Simulating User Trajectory...")
    res_user = unified_simulate(patient=user_patient, intervention=user_intervention, sleep_quality=0.85, sim_years=horizon, transplant_protocol="rescue")
    
    print("Simulating Peter Trajectory...")
    res_peter = unified_simulate(patient=peter_patient, intervention=peter_intervention, sleep_quality=0.9, sim_years=horizon, transplant_protocol="decadal")
    
    # ── 4. REPORT ──
    print(f"\n30-YEAR HOUSEHOLD CEILING (Year 30):")
    print(f"{'Metric':<25} | {'User (Age 93.8)':<20} | {'Peter (Age 58.0)'}")
    print("-" * 75)
    
    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("ATP Production", 2, "state"),
        ("Brain ROS", 3, "state"),
        ("Heteroplasmy", None, "het"),
        ("Insulin Sensitivity", 32, "state"),
        ("Hormone Shield", 33, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_u, v_p = res_user["memory_index"][-1], res_peter["memory_index"][-1]
        elif mtype == "het":
            def get_het(r):
                total = r["states"][-1, 0] + r["states"][-1, 1] + r["states"][-1, 7]
                return (r["states"][-1, 1] + r["states"][-1, 7]) / max(total, 1e-12)
            v_u, v_p = get_het(res_user), get_het(res_peter)
        else:
            v_u, v_p = res_user["states"][-1, idx], res_peter["states"][-1, idx]
        print(f"{label:<25} | {v_u:20.4f} | {v_p:.4f}")

if __name__ == "__main__":
    run_household_forecast()
