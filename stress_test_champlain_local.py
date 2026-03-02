"""stress_test_champlain_local.py

A transient disturbance simulation for the Champlain Household.
Measures 'Recovery Velocity' after acute metabolic shocks.

NOTE: This file is ignored by git.
"""

import numpy as np
import os
from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION

def run_stress_test(name, subject, p_params, i_params, shock_duration=0.1, shock_type="inflammation"):
    # Baseline Simulation (1 year)
    res_base = unified_simulate(patient=p_params, intervention=i_params, sim_years=1.0)
    baseline_atp = res_base["states"][-1, 2]
    
    # Shock Simulation
    p_shock = dict(p_params)
    i_shock = dict(i_params)
    
    if shock_type == "preschool_spike":
        p_shock["inflammation_level"] = 0.8 # Viral load
        p_shock["gut_health"] = 0.3 # Gut/Liver drain
    elif shock_type == "sleep_crisis":
        i_shock["sleep_quality"] = 0.2 # Extreme deprivation
    elif shock_type == "gallery_crunch":
        p_shock["intellectual_engagement"] = 1.5
        i_shock["sleep_quality"] = 0.4
        i_shock["coffee_intake"] = 0.8
        
    res_shock = unified_simulate(patient=p_shock, intervention=i_shock, sim_years=shock_duration)
    shock_atp = res_shock["states"][-1, 2]
    shock_ros = res_shock["states"][-1, 3]
    
    # Recovery Simulation (1 year post-shock)
    res_rec = unified_simulate(patient=p_params, intervention=i_params, sim_years=1.0)
    final_atp = res_rec["states"][-1, 2]
    
    # Calculate Recovery Velocity
    recovery_velocity = (final_atp - shock_atp) / 1.0
    
    return {
        "name": name,
        "subject": subject,
        "baseline_atp": baseline_atp,
        "shock_atp": shock_atp,
        "shock_ros": shock_ros,
        "recovery_velocity": recovery_velocity,
        "seizure_risk": "HIGH" if (subject == "Ratio" and (shock_atp < 0.7 or shock_ros > 1.2)) else "LOW"
    }

if __name__ == "__main__":
    # --- SUBJECT F: RATIO (23) ---
    f_p = dict(DEFAULT_PATIENT); f_p.update({"baseline_age": 23.0, "profile": "artist", "intellectual_engagement": 1.0, "seizure_vulnerability": 0.8})
    f_i = dict(DEFAULT_INTERVENTION); f_i.update({"diet_type": "mediterranean", "cannabis_dose": 0.6, "sleep_quality": 0.85, "magnesium_dose": 0.8})
    
    # --- SUBJECT G: JASPER (24) ---
    g_p = dict(DEFAULT_PATIENT); g_p.update({"baseline_age": 24.0, "profile": "biologist", "sex": "male"})
    g_i = dict(DEFAULT_INTERVENTION); g_i.update({"diet_type": "mediterranean", "social_protocol": "teaching", "nr_dose": 0.9, "magnesium_dose": 1.0})
    
    print("CHAMPLAIN HOUSEHOLD: METABOLIC STORM STRESS TEST\n")
    
    # Test 1: Jasper Preschool Spike
    t1 = run_stress_test("Preschool Spike", "Jasper", g_p, g_i, shock_duration=0.08, shock_type="preschool_spike")
    print(f"Scenario: {t1['name']:<15} | Subject: {t1['subject']:<8}")
    print(f"  ATP Drop: {t1['baseline_atp']:.4f} -> {t1['shock_atp']:.4f}")
    print(f"  Shock ROS: {t1['shock_ros']:.4f} | Recovery Vel: {t1['recovery_velocity']:.4f}")
    print("-" * 50)
    
    # Test 2: Ratio Sleep Crisis
    t2 = run_stress_test("Sleep Crisis", "Ratio", f_p, f_i, shock_duration=0.04, shock_type="sleep_crisis")
    print(f"Scenario: {t2['name']:<15} | Subject: {t2['subject']:<8}")
    print(f"  ATP Drop: {t2['baseline_atp']:.4f} -> {t2['shock_atp']:.4f}")
    print(f"  Shock ROS: {t2['shock_ros']:.4f} | Seizure Risk: {t2['seizure_risk']}")
    print("-" * 50)
    
    # Test 3: Gallery Crunch (Ratio)
    t3 = run_stress_test("Gallery Crunch", "Ratio", f_p, f_i, shock_duration=0.04, shock_type="gallery_crunch")
    print(f"Scenario: {t3['name']:<15} | Subject: {t3['subject']:<8}")
    print(f"  ATP Drop: {t3['baseline_atp']:.4f} -> {t3['shock_atp']:.4f}")
    print(f"  Shock ROS: {t3['shock_ros']:.4f} | Seizure Risk: {t3['seizure_risk']}")
    print("-" * 50)
    
    # Test 4: Gallery Crunch (Jasper)
    t4 = run_stress_test("Gallery Crunch", "Jasper", g_p, g_i, shock_duration=0.04, shock_type="gallery_crunch")
    print(f"Scenario: {t4['name']:<15} | Subject: {t4['subject']:<8}")
    print(f"  ATP Drop: {t4['baseline_atp']:.4f} -> {t4['shock_atp']:.4f}")
    print(f"  Shock ROS: {t4['shock_ros']:.4f} | Recovery Vel: {t4['recovery_velocity']:.4f}")
    print("-" * 50)
