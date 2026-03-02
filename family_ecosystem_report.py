"""family_ecosystem_report.py

The primary reporting script for the Family Ecosystem Reporting Module (FERM).
Generates a unified Mitochondrial Maturity Matrix for all seven family members.

NOTE: This file is ignored by git.
"""

from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION
import numpy as np
import os

def calculate_maturity_metrics(name, age, profile, p_overrides=None, i_overrides=None):
    p = dict(DEFAULT_PATIENT); p.update({"baseline_age": age, "profile": profile})
    if p_overrides: p.update(p_overrides)
    
    # Setup Optimized Intervention (Surge + Physical + EE + Precision)
    i = dict(DEFAULT_INTERVENTION)
    i.update({
        "nr_dose": 1.0, 
        "sleep_quality": 0.85,
        "red_light_therapy": 0.8,
        "restorative_yoga": 0.9,
        "animal_protocol": "full_farm",
        "diet_type": "mediterranean",
        "side_sleeping": 1.0,           # Phase 11
        "akkermansia_probiotic": 1.0    # Phase 11
    })
    
    # Subject-Specific Precision Tiers
    if profile == "eds":
        i.update({"urolithin_a": 1.0, "vns_intensity": 1.0})
    if name == "Ratio":
        i.update({"molecular_hydrogen": 1.0})
    if age > 60:
        i.update({"spermidine": 1.0})
        
    if i_overrides: i.update(i_overrides)
    
    res = unified_simulate(patient=p, intervention=i, sim_years=20.0)
    atp = res["states"][-1, 2]
    het = res["states"][-1, 1] + res["states"][-1, 7]
    mi = res["memory_index"][-1]
    
    m_age = age + (het * 100 - 30)
    het_delta_per_year = (het - (res["states"][0, 1] + res["states"][0, 7])) / 20.0
    runway = max(0, (0.50 - het) / max(het_delta_per_year, 1e-6))
    
    return {"name": name, "chrono_age": age, "m_age": m_age, "atp": atp, "het": het, "mi": mi, "runway": runway}

if __name__ == "__main__":
    print("FAMILY ECOSYSTEM REPORTING MODULE (FERM): MASTER MATRIX v1.0\n")
    
    family = [
        calculate_maturity_metrics("Ratio", 23, "artist", {"structural_drag_override": 1.15, "seizure_vulnerability": 0.8}, {"magnesium_dose": 1.0}),
        calculate_maturity_metrics("Jasper", 24, "eds", {"structural_drag_override": 1.45}, {"nr_dose": 1.0}),
        calculate_maturity_metrics("Kathryn", 64, "scholar", {"sex": "female", "osteopenia": True}, {"rapamycin_dose": 0.4}),
        calculate_maturity_metrics("Peter", 28, "eds", {"structural_drag_override": 1.35}, {"nr_dose": 1.0}),
        calculate_maturity_metrics("John Jr.", 91, "scholar", {"baseline_heteroplasmy": 0.65}, {"transplant_rate": 0.9}),
        calculate_maturity_metrics("John III", 62, "scholar", {"sex": "male"}, {"rapamycin_dose": 0.4}),
        calculate_maturity_metrics("Selena Shea", 26, "eds", {"structural_drag_override": 1.30}, {"nr_dose": 1.0})
    ]
    
    family.sort(key=lambda x: x["m_age"], reverse=True)
    
    print(f"{'Name':<15} | {'Age':<4} | {'M-Age':<6} | {'ATP':<6} | {'Het':<6} | {'MemIdx':<6} | {'Runway':<6}")
    print("-" * 80)
    for f in family:
        print(f"{f['name']:<15} | {f['chrono_age']:<4} | {f['m_age']:<6.1f} | {f['atp']:<6.3f} | {f['het']:<6.3f} | {f['mi']:<6.3f} | {f['runway']:<6.1f} yrs")
    
    print("\n--- CLINICAL STATUS SUMMARY ---")
    for f in family:
        zone = "GREEN" if f['atp'] > 0.85 else ("YELLOW" if f['atp'] > 0.75 else "RED")
        print(f"  {f['name']:<15}: {zone} ZONE | M-Age {f['m_age']:>5.1f} | Projected Memory Index: {f['mi']:.3f}")
