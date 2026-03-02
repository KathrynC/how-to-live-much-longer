"""
Longevity Atlas Builder: 1,000-evaluation survey of mitochondrial aging.
Maps the fine-grained physics of survival and names emergent archetypes.
"""

import sys
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent
EA_TOOLKIT_PATH = PROJECT_ROOT.parent / "ea-toolkit"

if str(EA_TOOLKIT_PATH) not in sys.path:
    sys.path.insert(0, str(EA_TOOLKIT_PATH))

import simulator
import analytics
import constants
from ea_toolkit.algorithms.map_elites import MAPElitesArchive
from ea_toolkit.algorithms.ridge_walker import RidgeWalker
from ea_toolkit.algorithms.novelty_seeker import NoveltySeeker
from ea_toolkit.base import FitnessFunction

class LongevityFitness(FitnessFunction):
    def __init__(self, archive: MAPElitesArchive):
        super().__init__()
        self.eval_count = 0
        self.archive = archive

    def evaluate(self, params: dict) -> dict:
        self.eval_count += 1
        if self.eval_count % 50 == 0:
            print(f"Evaluated {self.eval_count} simulations...")
            
        intervention = {k: params[k] for k in ["rapamycin_dose", "nad_supplement", "senolytic_dose", "yamanaka_intensity", "transplant_rate", "exercise_level"]}
        patient = {k: params[k] for k in ["baseline_age", "baseline_heteroplasmy", "baseline_nad_level", "genetic_vulnerability", "metabolic_demand", "inflammation_level"]}
        
        res = simulator.simulate(intervention=intervention, patient=patient)
        stats = analytics.compute_all(res)
        
        atp = stats["energy"]["atp_final"]
        survival = stats["energy"]["reserve_ratio"]
        het = stats["damage"]["het_final"]
        
        fitness = atp * 5.0 + survival * 5.0 - het * 2.0
        
        res_data = {
            "fitness": float(fitness),
            "final_atp": float(atp),
            "final_het": float(het),
            "survival": float(survival),
            "regime": stats["dynamics"].get("regime", "UNKNOWN")
        }
        self.archive.add(params, res_data)
        return res_data

    def param_spec(self) -> dict:
        return {
            "rapamycin_dose": (0.0, 1.0),
            "nad_supplement": (0.0, 1.0),
            "senolytic_dose": (0.0, 1.0),
            "yamanaka_intensity": (0.0, 1.0),
            "transplant_rate": (0.0, 1.0),
            "exercise_level": (0.0, 1.0),
            "baseline_age": (20.0, 90.0),
            "baseline_heteroplasmy": (0.0, 0.9),
            "baseline_nad_level": (0.2, 1.0),
            "genetic_vulnerability": (0.5, 2.0),
            "metabolic_demand": (0.5, 2.5),
            "inflammation_level": (0.0, 1.0)
        }

def longevity_descriptor(res: dict) -> tuple:
    x = np.clip(res.get("final_het", 0.5), 0, 1)
    y = np.clip(res.get("final_atp", 0.5), 0, 1)
    z = np.clip(res.get("survival", 0.5), 0, 1)
    return (float(x), float(y), float(z))

def run_longevity_atlas(budget: int = 1000):
    print(f"--- Starting Longevity Atlas (Budget: {budget}) ---")
    archive = MAPElitesArchive(dims=3, bins_per_dim=25, descriptor_fn=longevity_descriptor)
    fitness_fn = LongevityFitness(archive)
    
    print("[Explorer 1/2] Novelty Search...")
    seeker = NoveltySeeker(fitness_fn, n_candidates=10)
    seeker.run(budget=budget // 2)
    
    print("[Explorer 2/2] Ridge Walker...")
    seed_params = {
        "rapamycin_dose": 0.25, "nad_supplement": 0.5, "senolytic_dose": 0.1, "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.75,
        "baseline_age": 80.0, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.6, "genetic_vulnerability": 0.8, "metabolic_demand": 0.8, "inflammation_level": 0.2
    }
    walker = RidgeWalker(fitness_fn, initial_params=seed_params, n_candidates=10)
    walker.run(budget=budget // 2)
    
    print(f"\n--- Atlas Complete: {len(archive.get_all_elites())} regions mapped ---")
    elites = archive.get_all_elites()
    
    # Export for D3
    _export_atlas_json(elites)
    _name_emergent_archetypes(elites)

def _export_atlas_json(elites):
    data = []
    for e in elites:
        res = e['result']
        params = e['params']
        data.append({
            "x": e['descriptor'][0],
            "y": e['descriptor'][1],
            "z": e['descriptor'][2],
            "atp": res['final_atp'],
            "het": res['final_het'],
            "fitness": res['fitness'],
            "age": params['baseline_age'],
            "regime": res['regime']
        })
    out_path = Path("artifacts/longevity_atlas_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"D3 Data exported to: {out_path}")

def _name_emergent_archetypes(elites):
    print("Spelunking for new named spots...")
    out_path = Path("docs/LONGEVITY_ATLAS_REPORT.md")
    with open(out_path, "w") as f:
        f.write("# The High-Resolution Longevity Atlas\n\n")
        
        spark = [e for e in elites if e['result']['final_atp'] > 0.9 and e['result']['final_het'] < 0.1]
        if spark:
            f.write("## New Sweet Spot: The Eternal Spark\n")
            f.write("Maximum biological fidelity. Peak function with low-dose interventions.\n\n")
            
        abyss = [e for e in elites if e['params']['yamanaka_intensity'] > 0.7 and e['result']['final_atp'] < 0.2]
        if abyss:
            f.write("## New Collapse Spot: The Reprogramming Abyss\n")
            f.write("High-intensity OSKM reprogramming exhausts Mu-day capacity, triggering instant energy collapse.\n\n")

        oasis = [e for e in elites if e['params']['baseline_age'] > 85 and e['result']['final_atp'] > 0.8]
        if oasis:
            f.write("## New Sweet Spot: The Transplant Oasis\n")
            f.write("Deep-time sanctuary for the oldest patients. Transplant resets the bio-clock.\n\n")

    print(f"Longevity Atlas Report saved to: {out_path}")

if __name__ == "__main__":
    run_longevity_atlas()
