"""westport_synergy_test_local.py

Quantifies the biological value of the 'Full Farm' environment for Westport.
Models: Kathryn (64) and Peter (28).

NOTE: This file is ignored by git.
"""

from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION
import numpy as np

def run_comparison(name, age, profile, p_overrides):
    # Base Case: No Farm
    p_base = dict(DEFAULT_PATIENT); p_base.update({"baseline_age": age, "profile": profile})
    p_base.update(p_overrides)
    i_base = dict(DEFAULT_INTERVENTION); i_base.update({"animal_protocol": "none", "nr_dose": 0.8, "sleep_quality": 0.85})
    
    res_base = unified_simulate(patient=p_base, intervention=i_base, sim_years=20.0)
    
    # Synergy Case: Full Farm
    i_farm = dict(i_base); i_farm.update({"animal_protocol": "full_farm"})
    res_farm = unified_simulate(patient=p_base, intervention=i_farm, sim_years=20.0)
    
    return {
        "name": name,
        "base_atp": res_base["states"][-1, 2],
        "farm_atp": res_farm["states"][-1, 2],
        "base_mi": res_base["memory_index"][-1],
        "farm_mi": res_farm["memory_index"][-1],
        "base_bdnf": res_base["states"][-1, 27],
        "farm_bdnf": res_farm["states"][-1, 27]
    }

if __name__ == "__main__":
    print("WESTPORT FULL FARM SYNERGY TEST\n")
    
    k_data = run_comparison("Kathryn", 64, "scholar", {"sex": "female", "osteopenia": True})
    p_data = run_comparison("Peter", 28, "biologist", {"sex": "male"})
    
    for d in [k_data, p_data]:
        print(f"Subject: {d['name']}")
        print(f"  ATP:   {d['base_atp']:.4f} -> {d['farm_atp']:.4f} (Δ {d['farm_atp'] - d['base_atp']:.4f})")
        print(f"  MemIdx: {d['base_mi']:.4f} -> {d['farm_mi']:.4f} (Δ {d['farm_mi'] - d['base_mi']:.4f})")
        print(f"  BDNF:   {d['base_bdnf']:.4f} -> {d['farm_bdnf']:.4f} (Δ {d['farm_bdnf'] - d['base_bdnf']:.4f})")
        print("-" * 40)
