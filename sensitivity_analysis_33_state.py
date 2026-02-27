"""sensitivity_analysis_33_state.py

Performs a One-At-a-Time (OAT) sensitivity analysis on the 33-state 
Universal Human Digital Twin to identify which lifestyle and intervention 
parameters provide the most "Bang for the Buck" for cognitive preservation 
(Memory Index).
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_sensitivity_analysis():
    print("Starting Sensitivity Analysis on 33-State Model...")
    
    # Define the baseline 70-year-old APOE4 carrier
    baseline_patient = dict(DEFAULT_PATIENT)
    baseline_patient.update({
        "baseline_age": 70.0,
        "apoe_genotype": "apoe4_het",
        "intellectual_engagement": 0.5,
        "grief_intensity": 0.5,
        "fructose_intake": 0.5,
        "salt_intake": 0.5,
        "protein_intake": 0.5,
        "pollution_exposure": 0.5
    })
    
    baseline_intervention = dict(DEFAULT_INTERVENTION)
    baseline_intervention.update({
        "exercise_level": 0.5,
        "alcohol_intake": 0.2,
        "fasting_regimen": 0.2,
        "therapy_intensity": 0.2,
        "probiotic_intensity": 0.2,
        "nad_supplement": 0.2,
        "transplant_rate": 0.0
    })
    
    baseline_sleep = 0.5
    
    # Run baseline to get a reference point
    res_baseline = unified_simulate(
        patient=baseline_patient, 
        intervention=baseline_intervention, 
        sleep_quality=baseline_sleep,
        sim_years=30.0
    )
    base_mi = res_baseline['memory_index'][-1]
    print(f"Baseline Memory Index: {base_mi:.4f}")
    print()
    
    # Define parameters to sweep and their ranges
    params_to_sweep = {
        # Interventions / Lifestyle (0.0 to 1.0)
        "sleep_quality": ("sleep", 0.1, 0.9),
        "exercise_level": ("intervention", 0.0, 1.0),
        "fasting_regimen": ("intervention", 0.0, 1.0),
        "nad_supplement": ("intervention", 0.0, 1.0),
        "transplant_rate": ("intervention", 0.0, 1.0),
        "alcohol_intake": ("intervention", 0.0, 1.0),
        "therapy_intensity": ("intervention", 0.0, 1.0),
        
        # Patient Environment / Diet (0.0 to 1.0)
        "fructose_intake": ("patient", 0.0, 1.0),
        "salt_intake": ("patient", 0.0, 1.0),
        "pollution_exposure": ("patient", 0.0, 1.0),
        "intellectual_engagement": ("patient", 0.0, 1.0)
    }
    
    results = []
    
    for param, (param_type, min_val, max_val) in params_to_sweep.items():
        # Test Min
        p_min = dict(baseline_patient)
        i_min = dict(baseline_intervention)
        s_min = baseline_sleep
        
        if param_type == "patient": p_min[param] = min_val
        elif param_type == "intervention": i_min[param] = min_val
        elif param_type == "sleep": s_min = min_val
            
        res_min = unified_simulate(patient=p_min, intervention=i_min, sleep_quality=s_min)
        mi_min = res_min['memory_index'][-1]
        
        # Test Max
        p_max = dict(baseline_patient)
        i_max = dict(baseline_intervention)
        s_max = baseline_sleep
        
        if param_type == "patient": p_max[param] = max_val
        elif param_type == "intervention": i_max[param] = max_val
        elif param_type == "sleep": s_max = max_val
            
        res_max = unified_simulate(patient=p_max, intervention=i_max, sleep_quality=s_max)
        mi_max = res_max['memory_index'][-1]
        
        # Calculate impact
        impact = abs(mi_max - mi_min)
        direction = "+" if mi_max > mi_min else "-"
        
        results.append({
            "parameter": param,
            "min_mi": mi_min,
            "max_mi": mi_max,
            "impact": impact,
            "direction": direction
        })
        
    # Sort by impact
    results.sort(key=lambda x: x["impact"], reverse=True)
    
    print(f"{'Parameter':<25} | {'Impact (Δ MI)':<15} | {'Direction':<10} | {'Range (Min -> Max)'}")
    print("-" * 75)
    for r in results:
        print(f"{r['parameter']:<25} | {r['impact']:.4f}          | {r['direction']:<10} | {r['min_mi']:.3f} -> {r['max_mi']:.3f}")
        
if __name__ == "__main__":
    run_sensitivity_analysis()
