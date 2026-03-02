"""champlain_enriched_environment_roi_local.py

Calculates the 'Biological ROI' of an Enriched Environment for Ratio and Jasper.
Models the Artist Information Ecosystem as a 'Full Farm' equivalent EE.

NOTE: This file is ignored by git.
"""

from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION
import numpy as np

def run_roi_analysis(name, age, drag, p_overrides):
    # Life-Path A: Baseline
    p_a = dict(DEFAULT_PATIENT); p_a.update({"baseline_age": age, "structural_drag_override": drag})
    p_a.update(p_overrides)
    i_a = dict(DEFAULT_INTERVENTION); i_a.update({"animal_protocol": "none", "nr_dose": 0.8, "sleep_quality": 0.85})
    
    res_a = unified_simulate(patient=p_a, intervention=i_a, sim_years=20.0)
    
    # Life-Path B: Enriched Artist Ecosystem
    i_b = dict(i_a); i_b.update({"animal_protocol": "full_farm"}) # Using farm as EE proxy
    res_b = unified_simulate(patient=p_a, intervention=i_b, sim_years=20.0)
    
    # Metrics
    final_het_a = res_a["states"][-1, 1] + res_a["states"][-1, 7]
    final_het_b = res_b["states"][-1, 1] + res_b["states"][-1, 7]
    
    # Life-Gain (Years)
    life_gain = (final_het_a - final_het_b) * 100
    atp_diff = (res_b["states"][-1, 2] - res_a["states"][-1, 2])
    
    return {
        "name": name, "life_gain": life_gain, "atp_diff": atp_diff,
        "base_het": final_het_a, "env_het": final_het_b
    }

if __name__ == "__main__":
    print("10 CHAMPLAIN: ENRICHED ENVIRONMENT ROI (20-Year Horizon)\n")
    r_roi = run_roi_analysis("Ratio (Age 23-43)", 23, 1.15, {"seizure_vulnerability": 0.8})
    j_roi = run_roi_analysis("Jasper (Age 24-44)", 24, 1.45, {})
    
    for r in [r_roi, j_roi]:
        print(f"Subject: {r['name']}")
        print(f"  Mitochondrial Life-Gain : {r['life_gain']:.1f} Additional Years")
        print(f"  Net ATP Increase        : +{r['atp_diff']:.4f} MU/day")
        print(f"  Final Heteroplasmy (EE): {r['env_het']:.4f}")
        print("-" * 50)
