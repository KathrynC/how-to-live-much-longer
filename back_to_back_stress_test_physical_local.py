"""back_to_back_stress_test_physical_local.py

Simulates two stressors with the Physical Recovery Suite (RLT, Sauna, Yoga).
Subjects: Peter (1.35x) and Jasper (1.45x).

NOTE: This file is ignored by git.
"""

from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION
import numpy as np

def run_double_shock_physical(name, age, drag):
    p = dict(DEFAULT_PATIENT); p.update({"baseline_age": age, "structural_drag_override": drag})
    i = dict(DEFAULT_INTERVENTION); i.update({
        "nr_dose": 0.95, 
        "sleep_quality": 0.85,
        "red_light_therapy": 0.8,
        "sauna_use": 0.5,
        "restorative_yoga": 0.9
    })
    
    # 1. Baseline
    res_base = unified_simulate(patient=p, intervention=i, sim_years=0.1)
    atp_0 = res_base["states"][-1, 2]
    
    # 2. Shock 1 (Viral)
    p_s1 = dict(p); p_s1["inflammation_level"] = 0.8
    res_s1 = unified_simulate(patient=p_s1, intervention=i, sim_years=0.08)
    atp_s1 = res_s1["states"][-1, 2]
    
    # 3. Brief Recovery (1 month)
    res_r1 = unified_simulate(patient=p, intervention=i, sim_years=0.08)
    atp_r1 = res_r1["states"][-1, 2]
    
    # 4. Shock 2 (Work Stress)
    i_s2 = dict(i); i_s2["sleep_quality"] = 0.4
    res_s2 = unified_simulate(patient=p, intervention=i_s2, sim_years=0.08)
    atp_s2 = res_s2["states"][-1, 2]
    
    return {
        "name": name, "atp_baseline": atp_0, "atp_shock1": atp_s1, 
        "atp_recovery": atp_r1, "atp_shock2": atp_s2,
        "critical_failure": "YES" if atp_s2 < 0.75 else "NO"
    }

if __name__ == "__main__":
    print("BACK-TO-BACK STRESS TEST: PHYSICAL RECOVERY SUITE\n")
    p_data = run_double_shock_physical("Peter (1.35x)", 28, 1.35)
    j_data = run_double_shock_physical("Jasper (1.45x)", 24, 1.45)
    
    for d in [p_data, j_data]:
        print(f"Subject: {d['name']}")
        print(f"  Baseline ATP: {d['atp_baseline']:.4f}")
        print(f"  After Shock 1 (Viral): {d['atp_shock1']:.4f}")
        print(f"  Mid-Recovery ATP: {d['atp_recovery']:.4f}")
        print(f"  After Shock 2 (Work):  {d['atp_shock2']:.4f}")
        print(f"  Critical Failure Risk: {d['critical_failure']}")
        print("-" * 40)
