"""rejuvenation_comparison_30yr.py

Compares three mitochondrial transplant protocols over 30 years for a 63.83yo:
1. Early Prevention (Age 65)
2. Cliff Rescue (Dynamic, triggers at 60% het)
3. Decadal Maintenance (Every 5 years)
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_rejuv_comparison():
    print("Evaluating Rejuvenation (Transplant) Protocols (30-Year Horizon)...")
    
    # Base Profile
    patient = dict(DEFAULT_PATIENT)
    patient.update({
        "baseline_age": 63.83,
        "apoe_genotype": "apoe4_het",
        "sex": "female",
        "intellectual_engagement": 1.0,
        "grief_intensity": 0.5
    })
    
    # Baseline intervention (Sustainable Scholar Full Stack)
    intervention_base = dict(DEFAULT_INTERVENTION)
    intervention_base.update({
        "diet_type": "mediterranean",
        "exercise_type": "aerobic",
        "exercise_level": 0.8,
        "social_protocol": "integrated",
        "animal_protocol": "full_farm",
        "alcohol_intake": 0.1,
        "nr_dose": 0.8,
        "rapamycin_dose": 0.3
    })
    
    sleep_q = 0.8
    horizon = 30.0 # Focus on the very long term (Age 93.8)
    
    # Run Scenarios
    print("Simulating Early Prevention (Age 65)...")
    res_early = unified_simulate(patient=patient, intervention=intervention_base, sleep_quality=sleep_q, sim_years=horizon, transplant_protocol="early")
    
    print("Simulating Cliff Rescue (at 60% het)...")
    res_rescue = unified_simulate(patient=patient, intervention=intervention_base, sleep_quality=sleep_q, sim_years=horizon, transplant_protocol="rescue")
    
    print("Simulating Decadal Maintenance (Every 5 yrs)...")
    res_decad = unified_simulate(patient=patient, intervention=intervention_base, sleep_quality=sleep_q, sim_years=horizon, transplant_protocol="decadal")
    
    print(f"\nRejuvenation Results at Year 30 (Age 93.8):")
    print(f"{'Metric':<25} | {'Early':<15} | {'Rescue':<15} | {'Decadal'}")
    print("-" * 80)
    
    # Metrics
    def get_het(res, idx):
        total = res["states"][idx, 0] + res["states"][idx, 1] + res["states"][idx, 7]
        return (res["states"][idx, 1] + res["states"][idx, 7]) / max(total, 1e-12)

    metrics = [
        ("Memory Index", -1, "memory_index"),
        ("ATP Production", 2, "state"),
        ("Brain ROS", 3, "state")
    ]
    
    for label, idx, mtype in metrics:
        if mtype == "memory_index":
            v_e, v_r, v_d = res_early["memory_index"][-1], res_rescue["memory_index"][-1], res_decad["memory_index"][-1]
        else:
            v_e, v_r, v_d = res_early["states"][-1, idx], res_rescue["states"][-1, idx], res_decad["states"][-1, idx]
        print(f"{label:<25} | {v_e:<15.4f} | {v_r:<15.4f} | {v_d:.4f}")
        
    print(f"{'Heteroplasmy':<25} | {get_het(res_early, -1):<15.4f} | {get_het(res_rescue, -1):<15.4f} | {get_het(res_decad, -1):.4f}")

if __name__ == "__main__":
    run_rejuv_comparison()
