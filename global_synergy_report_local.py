"""global_synergy_report_local.py

Named Comparative Health Matrix using the refined EDS Gradient Model.
- Jasper: 1.45x (Diagnosed)
- Peter: 1.35x (Loosely Diagnosed)
- Selena: 1.30x (High Symptom Profile)
- Ratio: 1.15x (Mild Symptomatic)
"""

from unified_brain_model import unified_simulate, DEFAULT_PATIENT, DEFAULT_INTERVENTION
import numpy as np

def get_subject_metrics(name, age, profile, p_overrides=None, i_overrides=None):
    p = dict(DEFAULT_PATIENT); p.update({"baseline_age": age, "profile": profile})
    if p_overrides: p.update(p_overrides)
    
    i = dict(DEFAULT_INTERVENTION); i.update({"nr_dose": 0.8, "sleep_quality": 0.85, "exercise_level": 0.6})
    if i_overrides: i.update(i_overrides)
    
    res = unified_simulate(patient=p, intervention=i, sim_years=20.0)
    final_atp = res["states"][-1, 2]
    final_het = res["states"][-1, 1] + res["states"][-1, 7]
    final_mi = res["memory_index"][-1]
    m_age = age + (final_het * 100 - 30)
    
    return {"name": name, "age": age, "atp": final_atp, "het": final_het, "mi": final_mi, "m_age": m_age}

if __name__ == "__main__":
    subjects = [
        # Champlain
        get_subject_metrics("Ratio", 23, "artist", {"seizure_vulnerability": 0.8, "structural_drag_override": 1.15}, {"magnesium_dose": 0.8}),
        get_subject_metrics("Jasper", 24, "eds", {"structural_drag_override": 1.45}, {"nr_dose": 1.0, "social_protocol": "teaching"}),
        # Westport
        get_subject_metrics("Kathryn", 64, "scholar", {"sex": "female", "osteopenia": True}, {"rapamycin_dose": 0.4, "animal_protocol": "full_farm"}),
        get_subject_metrics("Peter", 28, "eds", {"structural_drag_override": 1.35}, {"nr_dose": 0.9, "animal_protocol": "full_farm"}),
        # Seattle
        get_subject_metrics("John Jr.", 91, "scholar", {"baseline_heteroplasmy": 0.65}, {"transplant_rate": 0.9}),
        get_subject_metrics("John III", 62, "scholar", {"sex": "male"}, {"rapamycin_dose": 0.4}),
        get_subject_metrics("Selena Shea", 26, "eds", {"structural_drag_override": 1.30}, {"nr_dose": 0.8})
    ]
    
    print(f"{'Name':<15} | {'Age':<4} | {'M-Age':<6} | {'ATP':<6} | {'Het':<6} | {'MemIdx':<6}")
    print("-" * 65)
    for s in subjects:
        print(f"{s['name']:<15} | {s['age']:<4} | {s['m_age']:<6.1f} | {s['atp']:<6.3f} | {s['het']:<6.3f} | {s['mi']:<6.3f}")
