"""telemetry_bridge_local.py

Phase 10 Prototype: Converts wearable telemetry into ODE parameters.
Demonstrates a 'Morning Sync' for Ratio and Kathryn.

NOTE: This file is ignored by git.
"""

import numpy as np
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def map_telemetry_to_params(telemetry_data):
    """
    Converts raw device data into simulator-ready overrides.
    
    Mapping Logic:
    - HRV (Oura/Whoop) -> Inflammation: (100 - HRV) / 100
    - Deep Sleep (min) -> Sleep Quality: min(deep_min / 90.0, 1.0)
    - Avg Glucose (CGM) -> Insulin Sensitivity: (120 - Glucose) / 50
    """
    overrides = {"patient": {}, "intervention": {}}
    
    # 1. Autonomic Load (Inflammation)
    if "hrv" in telemetry_data:
        # High HRV = Low Inflammation. 
        # Baseline normal ~ 50ms. Scale 0 to 100.
        overrides["patient"]["inflammation_level"] = max(0.05, (100 - telemetry_data["hrv"]) / 150.0)
        
    # 2. Repair Window (Sleep Quality)
    if "deep_sleep_min" in telemetry_data:
        # 90 mins of deep sleep is the '1.0' target.
        overrides["intervention"]["sleep_quality"] = min(telemetry_data["deep_sleep_min"] / 90.0, 1.0)
        
    # 3. Metabolic Tax (Insulin)
    if "avg_glucose" in telemetry_data:
        # Healthy target < 90. Spikes reduce sensitivity.
        overrides["patient"]["insulin_sensitivity"] = max(0.1, (140 - telemetry_data["avg_glucose"]) / 100.0)
        
    return overrides

if __name__ == "__main__":
    print("PHASE 10 TELEMETRY BRIDGE: MORNING SYNC DEMO\n")
    
    # Example: A 'Bad Night' for Ratio (Low sleep, Low HRV)
    ratio_bad_night = {"hrv": 35, "deep_sleep_min": 40, "avg_glucose": 105}
    
    # Example: A 'Perfect Night' for Kathryn (High sleep, High HRV)
    kathryn_good_night = {"hrv": 75, "deep_sleep_min": 95, "avg_glucose": 88}
    
    for name, tel in [("Ratio", ratio_bad_night), ("Kathryn", kathryn_good_night)]:
        p_over = map_telemetry_to_params(tel)
        
        print(f"Syncing {name} with telemetry data: {tel}")
        print(f"  Resulting Inflammation: {p_over['patient']['inflammation_level']:.3f}")
        print(f"  Resulting Sleep Quality: {p_over['intervention']['sleep_quality']:.3f}")
        
        # Run a 1-day 'Forecast' based on this telemetry
        p_final = dict(DEFAULT_PATIENT); p_final.update(p_over["patient"])
        i_final = dict(DEFAULT_INTERVENTION); i_final.update(p_over["intervention"])
        
        res = unified_simulate(patient=p_final, intervention=i_final, sim_years=0.01) # 3.6 days
        atp = res["states"][-1, 2]
        ros = res["states"][-1, 3]
        
        print(f"  Forecasting Next 72 Hours:")
        print(f"    Energy Reserve (ATP): {atp:.4f}")
        print(f"    Oxidative Stress (ROS): {ros:.4f}")
        if name == "Ratio":
            risk = "ELEVATED" if ros > 0.25 else "LOW"
            print(f"    Daily Seizure Risk: {risk}")
        print("-" * 50)
