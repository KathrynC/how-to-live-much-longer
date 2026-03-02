"""
Longevity Spelunker: Expanded to 12 Medical Motifs.
Maps 12 diverse medical and archetypal seeds to mitochondrial ODE vectors.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np

# Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent
EA_TOOLKIT_PATH = PROJECT_ROOT.parent / "ea-toolkit"

if str(EA_TOOLKIT_PATH) not in sys.path:
    sys.path.insert(0, str(EA_TOOLKIT_PATH))

import simulator
import analytics
import constants

# --- THE EXPANDED MOTIF LIBRARY ---
MOTIFS = {
    "The Blue Zone": {
        "intervention": {"rapamycin_dose": 0.25, "nad_supplement": 0.5, "senolytic_dose": 0.1, "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.75},
        "patient": {"baseline_age": 80.0, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.6, "genetic_vulnerability": 0.8, "metabolic_demand": 0.8, "inflammation_level": 0.2}
    },
    "The Icarus Trap": {
        "intervention": {"rapamycin_dose": 1.0, "nad_supplement": 1.0, "senolytic_dose": 1.0, "yamanaka_intensity": 0.5, "transplant_rate": 0.0, "exercise_level": 1.0},
        "patient": {"baseline_age": 60.0, "baseline_heteroplasmy": 0.45, "baseline_nad_level": 0.4, "genetic_vulnerability": 1.2, "metabolic_demand": 1.5, "inflammation_level": 0.8}
    },
    "The Stoic": {
        "intervention": {"rapamycin_dose": 0.75, "nad_supplement": 0.1, "senolytic_dose": 0.5, "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.25},
        "patient": {"baseline_age": 70.0, "baseline_heteroplasmy": 0.4, "baseline_nad_level": 0.5, "genetic_vulnerability": 1.0, "metabolic_demand": 1.0, "inflammation_level": 0.4}
    },
    "The Mitrix Renaissance": {
        "intervention": {"rapamycin_dose": 0.25, "nad_supplement": 0.75, "senolytic_dose": 0.5, "yamanaka_intensity": 0.0, "transplant_rate": 1.0, "exercise_level": 0.5},
        "patient": {"baseline_age": 90.0, "baseline_heteroplasmy": 0.6, "baseline_nad_level": 0.3, "genetic_vulnerability": 1.5, "metabolic_demand": 1.2, "inflammation_level": 0.9}
    },
    "The Fasting Monk": {
        "intervention": {"rapamycin_dose": 1.0, "nad_supplement": 0.0, "senolytic_dose": 0.25, "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.5},
        "patient": {"baseline_age": 50.0, "baseline_heteroplasmy": 0.15, "baseline_nad_level": 0.8, "genetic_vulnerability": 0.9, "metabolic_demand": 0.7, "inflammation_level": 0.1}
    },
    "The Biohacker": {
        "intervention": {"rapamycin_dose": 0.25, "nad_supplement": 1.0, "senolytic_dose": 0.5, "yamanaka_intensity": 0.25, "transplant_rate": 0.1, "exercise_level": 0.5},
        "patient": {"baseline_age": 40.0, "baseline_heteroplasmy": 0.1, "baseline_nad_level": 0.9, "genetic_vulnerability": 1.1, "metabolic_demand": 1.3, "inflammation_level": 0.3}
    },
    "The Olympian": {
        "intervention": {"rapamycin_dose": 0.0, "nad_supplement": 0.0, "senolytic_dose": 0.0, "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 1.0},
        "patient": {"baseline_age": 30.0, "baseline_heteroplasmy": 0.05, "baseline_nad_level": 1.0, "genetic_vulnerability": 1.0, "metabolic_demand": 2.0, "inflammation_level": 0.1}
    },
    "The Fragile APOE4": {
        "intervention": {"rapamycin_dose": 0.5, "nad_supplement": 0.5, "senolytic_dose": 1.0, "yamanaka_intensity": 0.0, "transplant_rate": 0.25, "exercise_level": 0.25},
        "patient": {"baseline_age": 60.0, "baseline_heteroplasmy": 0.4, "baseline_nad_level": 0.5, "genetic_vulnerability": 2.0, "metabolic_demand": 1.5, "inflammation_level": 0.7}
    },
    "The Brain-on-Fire": {
        "intervention": {"rapamycin_dose": 0.5, "nad_supplement": 1.0, "senolytic_dose": 0.5, "yamanaka_intensity": 0.0, "transplant_rate": 0.5, "exercise_level": 0.25},
        "patient": {"baseline_age": 65.0, "baseline_heteroplasmy": 0.4, "baseline_nad_level": 0.4, "genetic_vulnerability": 1.3, "metabolic_demand": 2.5, "inflammation_level": 0.8}
    },
    "The Sleeping Giant": {
        "intervention": {"rapamycin_dose": 0.1, "nad_supplement": 0.1, "senolytic_dose": 0.1, "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.1},
        "patient": {"baseline_age": 75.0, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.5, "genetic_vulnerability": 0.7, "metabolic_demand": 0.5, "inflammation_level": 0.2}
    },
    "The Stem Cell Scion": {
        "intervention": {"rapamycin_dose": 0.25, "nad_supplement": 0.5, "senolytic_dose": 0.5, "yamanaka_intensity": 0.25, "transplant_rate": 1.0, "exercise_level": 0.25},
        "patient": {"baseline_age": 70.0, "baseline_heteroplasmy": 0.5, "baseline_nad_level": 0.4, "genetic_vulnerability": 1.2, "metabolic_demand": 1.2, "inflammation_level": 0.6}
    },
    "The Baseline": {
        "intervention": {"rapamycin_dose": 0.0, "nad_supplement": 0.0, "senolytic_dose": 0.0, "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.0},
        "patient": {"baseline_age": 70.0, "baseline_heteroplasmy": 0.4, "baseline_nad_level": 0.5, "genetic_vulnerability": 1.0, "metabolic_demand": 1.0, "inflammation_level": 0.4}
    }
}

def run_longevity_spelunk():
    print(f"--- Launching the Expanded Longevity Spelunker (12 Motifs) ---")
    results = []
    
    for name, config in MOTIFS.items():
        print(f"Pelted: {name}...")
        res = simulator.simulate(intervention=config["intervention"], patient=config["patient"])
        stats = analytics.compute_all(res)
        results.append({
            "name": name,
            "final_atp": stats["energy"]["atp_final"],
            "survival_score": stats["energy"]["reserve_ratio"],
            "cliff_distance": stats["damage"]["cliff_distance_final"],
            "outcome": stats["dynamics"].get("regime", "UNKNOWN")
        })
        
    results.sort(key=lambda x: x["final_atp"], reverse=True)
    out_path = PROJECT_ROOT / "docs" / "LONGEVITY_PELT_REPORT.md"
    with open(out_path, "w") as f:
        f.write("# The Longevity Pelt: 12 Medical Motifs\n\n")
        f.write("| Motif | Outcome | Final ATP | Cliff Distance | Survival Vibe |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| **{r['name']}** | {r['outcome']} | {r['final_atp']:.3f} | {r['cliff_distance']:.3f} | {r['survival_score']:.2f} |\n")
            
    print(f"Longevity Pelt Report saved to: {out_path}")

if __name__ == "__main__":
    run_longevity_spelunk()
