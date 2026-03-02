"""westport_full_farm_roi_local.py

Calculates the 'Biological ROI' of the Full Farm environment for Kathryn and Peter.
Compares Urban Baseline vs. Full Farm over 20 years.

NOTE: This file is ignored by git.
"""

from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION
import numpy as np

def run_roi_analysis(name, age, drag):
    # Life-Path A: Urban Baseline
    p_a = dict(DEFAULT_PATIENT); p_a.update({"baseline_age": age, "structural_drag_override": drag})
    i_a = dict(DEFAULT_INTERVENTION); i_a.update({"animal_protocol": "none", "nr_dose": 0.5, "sleep_quality": 0.7})
    res_a = unified_simulate(patient=p_a, intervention=i_a, sim_years=20.0)
    
    # Life-Path B: Full Farm
    i_b = dict(i_a); i_b.update({"animal_protocol": "full_farm", "nr_dose": 0.9, "sleep_quality": 0.85})
    res_b = unified_simulate(patient=p_a, intervention=i_b, sim_years=20.0)
    
    # Metrics
    final_het_a = res_a["states"][-1, 1] + res_a["states"][-1, 7]
    final_het_b = res_b["states"][-1, 1] + res_b["states"][-1, 7]
    
    # Life-Gain (Years) - Crude proxy: each 0.01 reduction in het ~ 1 year of runway
    life_gain = (final_het_a - final_het_b) * 100
    
    mi_boost = (res_b["memory_index"][-1] - res_a["memory_index"][-1]) / max(res_a["memory_index"][-1], 1e-12) * 100
    atp_diff = (res_b["states"][-1, 2] - res_a["states"][-1, 2])
    
    return {
        "name": name, "life_gain": life_gain, "mi_boost": mi_boost, "atp_diff": atp_diff,
        "base_mi": res_a["memory_index"][-1], "farm_mi": res_b["memory_index"][-1]
    }

if __name__ == "__main__":
    print("WESTPORT FULL FARM: BIOLOGICAL ROI ANALYSIS (20-Year Horizon)\n")
    k_roi = run_roi_analysis("Kathryn (Age 64-84)", 64, 1.0)
    p_roi = run_roi_analysis("Peter (Age 28-48)", 28, 1.35)
    
    for r in [k_roi, p_roi]:
        print(f"Subject: {r['name']}")
        print(f"  Mitochondrial Life-Gain : {r['life_gain']:.1f} Additional Years")
        print(f"  Memory Index Boost      : {r['mi_boost']:.1f}%")
        print(f"  Net ATP Increase        : +{r['atp_diff']:.4f} MU/day")
        print(f"  Final Memory Index (Farm): {r['farm_mi']:.4f}")
        print("-" * 50)
