#!/usr/bin/env python3
"""
protocol_interpolation.py

Linear interpolation between champion intervention protocols in 12D space.

Adapted from gait_interpolation.py in the parent Evolutionary-Robotics project,
which interpolated between champion gaits in 6D weight space and found
super-gaits at midpoints. Here we interpolate between known-effective
intervention protocols to map the fitness landscape and find synergistic
combinations.

Experiments:
  1. Pairwise interpolation: 5 champions × C(5,2)=10 pairs × 21 alpha steps
  2. Radial sweep from no-treatment center to each champion (5 × 21 steps)
  3. 3D grid through (rapamycin, NAD, exercise) subspace (10×10×10)

Scale: ~1325 sims
Estimated time: ~3 minutes
"""

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from constants import (
    DEFAULT_INTERVENTION,
    INTERVENTION_NAMES,
)
from simulator import simulate
from analytics import NumpyEncoder

PROJECT = Path(__file__).resolve().parent


# ── Champion protocols ──────────────────────────────────────────────────────

CHAMPIONS = {
    "rapamycin_heavy": {
        "label": "Rapamycin-heavy",
        "params": {
            "rapamycin_dose": 0.75, "nad_supplement": 0.25,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.25,
        },
    },
    "nad_heavy": {
        "label": "NAD-heavy",
        "params": {
            "rapamycin_dose": 0.25, "nad_supplement": 0.75,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.25,
        },
    },
    "full_cocktail": {
        "label": "Full cocktail",
        "params": {
            "rapamycin_dose": 0.5, "nad_supplement": 0.5,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5,
        },
    },
    "transplant_focused": {
        "label": "Transplant-focused",
        "params": {
            "rapamycin_dose": 0.1, "nad_supplement": 0.25,
            "senolytic_dose": 0.1, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.75, "exercise_level": 0.1,
        },
    },
    "yamanaka_cautious": {
        "label": "Yamanaka (cautious)",
        "params": {
            "rapamycin_dose": 0.25, "nad_supplement": 0.5,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.25,
            "transplant_rate": 0.0, "exercise_level": 0.25,
        },
    },
}

# Test patient: moderate 60-year-old
TEST_PATIENT = {
    "baseline_age": 60.0, "baseline_heteroplasmy": 0.40,
    "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0, "inflammation_level": 0.3,
}


# ── Interpolation helpers ──────────────────────────────────────────────────

def interpolate_interventions(intv_a, intv_b, alpha):
    """Linearly interpolate between two intervention dicts.

    Args:
        intv_a: Dict of intervention params (alpha=0 endpoint).
        intv_b: Dict of intervention params (alpha=1 endpoint).
        alpha: Interpolation weight (0.0 to 1.0).

    Returns:
        Interpolated intervention dict.
    """
    result = {}
    for k in INTERVENTION_NAMES:
        va = intv_a.get(k, 0.0)
        vb = intv_b.get(k, 0.0)
        result[k] = va + alpha * (vb - va)
    return result


def evaluate_intervention(intervention, patient):
    """Run simulation and return key metrics."""
    result = simulate(intervention=intervention, patient=patient)
    return {
        "final_atp": float(result["states"][-1, 2]),
        "final_het": float(result["heteroplasmy"][-1]),
        "min_atp": float(np.min(result["states"][:, 2])),
        "mean_atp": float(np.mean(result["states"][:, 2])),
        "max_het": float(np.max(result["heteroplasmy"])),
        "final_copy_number": float(result["states"][-1, 0] + result["states"][-1, 1]),
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    out_path = PROJECT / "artifacts" / "protocol_interpolation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    alpha_steps = np.linspace(0.0, 1.0, 21)
    trial_num = 0

    # Estimate total sims
    n_pairs = len(list(combinations(CHAMPIONS.keys(), 2)))
    n_total = (n_pairs * len(alpha_steps)      # pairwise interpolation
               + len(CHAMPIONS) * len(alpha_steps)  # radial from center
               + 10 * 10 * 10)                       # 3D grid
    print(f"{'=' * 70}")
    print(f"PROTOCOL INTERPOLATION EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Champions: {list(CHAMPIONS.keys())}")
    print(f"Alpha steps: {len(alpha_steps)}")
    print(f"Estimated sims: ~{n_total}")
    print()

    # ── Baseline ────────────────────────────────────────────────────────────
    baseline_metrics = evaluate_intervention(DEFAULT_INTERVENTION, TEST_PATIENT)
    print(f"Baseline (no treatment): ATP={baseline_metrics['final_atp']:.3f} "
          f"het={baseline_metrics['final_het']:.3f}")

    # ── 1. Pairwise interpolation ──────────────────────────────────────────
    print(f"\n--- Pairwise Interpolation ({n_pairs} pairs × {len(alpha_steps)} steps) ---")
    pairwise_results = []

    for name_a, name_b in combinations(CHAMPIONS.keys(), 2):
        intv_a = CHAMPIONS[name_a]["params"]
        intv_b = CHAMPIONS[name_b]["params"]
        pair_label = f"{name_a} → {name_b}"

        print(f"  {pair_label}", end="", flush=True)
        pair_data = []

        for alpha in alpha_steps:
            trial_num += 1
            intv = interpolate_interventions(intv_a, intv_b, alpha)
            metrics = evaluate_intervention(intv, TEST_PATIENT)
            pair_data.append({
                "alpha": float(alpha),
                "intervention": intv,
                **metrics,
            })

        # Find peak along this interpolation path
        best = max(pair_data, key=lambda d: d["final_atp"])
        endpoint_max = max(pair_data[0]["final_atp"], pair_data[-1]["final_atp"])
        is_super = best["final_atp"] > endpoint_max + 0.005

        print(f" -> peak ATP={best['final_atp']:.3f} at α={best['alpha']:.2f}"
              f"{' (SUPER-PROTOCOL!)' if is_super else ''}")

        pairwise_results.append({
            "pair": [name_a, name_b],
            "pair_label": pair_label,
            "has_super_protocol": is_super,
            "peak_alpha": best["alpha"],
            "peak_atp": best["final_atp"],
            "endpoint_a_atp": pair_data[0]["final_atp"],
            "endpoint_b_atp": pair_data[-1]["final_atp"],
            "data": pair_data,
        })

    # ── 2. Radial sweep from no-treatment center ──────────────────────────
    print(f"\n--- Radial Sweep (center → {len(CHAMPIONS)} champions) ---")
    radial_results = []

    for name, champ in CHAMPIONS.items():
        print(f"  center → {name}", end="", flush=True)
        radial_data = []

        for alpha in alpha_steps:
            trial_num += 1
            intv = interpolate_interventions(DEFAULT_INTERVENTION, champ["params"], alpha)
            metrics = evaluate_intervention(intv, TEST_PATIENT)
            radial_data.append({"alpha": float(alpha), **metrics})

        # Marginal gain analysis
        gains = [radial_data[i+1]["final_atp"] - radial_data[i]["final_atp"]
                 for i in range(len(radial_data) - 1)]
        diminishing_idx = None
        for j in range(1, len(gains)):
            if gains[j] < gains[j - 1] * 0.3:  # sharp drop in marginal gain
                diminishing_idx = j
                break

        print(f" -> full ATP={radial_data[-1]['final_atp']:.3f}"
              f" diminishing returns at α≈{alpha_steps[diminishing_idx]:.2f}"
              if diminishing_idx else
              f" -> full ATP={radial_data[-1]['final_atp']:.3f} (no diminishing returns)")

        radial_results.append({
            "champion": name,
            "champion_label": champ["label"],
            "data": radial_data,
            "diminishing_returns_alpha": (float(alpha_steps[diminishing_idx])
                                          if diminishing_idx else None),
        })

    # ── 3. 3D grid: rapamycin × NAD × exercise ──────────────────────────
    print(f"\n--- 3D Grid: rapamycin × NAD × exercise (10×10×10) ---")
    grid_values = np.linspace(0.0, 1.0, 10)
    grid_results = []

    for i, rapa in enumerate(grid_values):
        for j, nad_val in enumerate(grid_values):
            for k, exercise in enumerate(grid_values):
                trial_num += 1
                intv = dict(DEFAULT_INTERVENTION)
                intv["rapamycin_dose"] = float(rapa)
                intv["nad_supplement"] = float(nad_val)
                intv["exercise_level"] = float(exercise)
                metrics = evaluate_intervention(intv, TEST_PATIENT)
                grid_results.append({
                    "rapamycin": float(rapa),
                    "nad": float(nad_val),
                    "exercise": float(exercise),
                    **metrics,
                })
        if (i + 1) % 3 == 0:
            print(f"  [{trial_num}] rapamycin sweep {i+1}/10...")

    # Find optimal point in 3D grid
    best_grid = max(grid_results, key=lambda d: d["final_atp"])
    print(f"  Best 3D point: rapa={best_grid['rapamycin']:.1f} "
          f"nad={best_grid['nad']:.1f} exercise={best_grid['exercise']:.1f} "
          f"→ ATP={best_grid['final_atp']:.3f}")

    elapsed = time.time() - start_time

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PROTOCOL INTERPOLATION COMPLETE — {elapsed:.1f}s ({trial_num} sims)")
    print(f"{'=' * 70}")

    # Super-protocol summary
    super_protocols = [p for p in pairwise_results if p["has_super_protocol"]]
    print(f"\nSuper-protocols found: {len(super_protocols)}/{len(pairwise_results)} pairs")
    for sp in super_protocols:
        print(f"  {sp['pair_label']}: peak ATP={sp['peak_atp']:.3f} at α={sp['peak_alpha']:.2f} "
              f"(endpoints: {sp['endpoint_a_atp']:.3f}, {sp['endpoint_b_atp']:.3f})")

    # Save
    output = {
        "experiment": "protocol_interpolation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "n_sims": trial_num,
        "test_patient": TEST_PATIENT,
        "baseline": baseline_metrics,
        "pairwise": pairwise_results,
        "radial": radial_results,
        "grid_3d": grid_results,
        "grid_3d_best": best_grid,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
