#!/usr/bin/env python3
"""
dark_matter.py

Random sampling of the 12D intervention space to classify "futile" interventions
— protocols that fail to improve patient outcomes despite nonzero effort.

Adapted from analyze_dark_matter.py in the parent Evolutionary-Robotics project,
which studied "dead gaits" (|DX|<1m) from random weight trials and classified
them as spinners, rockers, vibrators, circlers, or inert.

Here we classify intervention outcomes into:
  - thriving:    final ATP > 0.8 AND het < 0.5
  - stable:      final ATP > 0.5 AND het < 0.7
  - declining:   final ATP > 0.2, het > 0.5
  - collapsed:   final ATP < 0.2
  - paradoxical: WORSE than no-treatment baseline on both ATP and het

The "paradoxical" category is the most interesting — interventions that
actively harm the patient. We subcategorize by which parameter is the culprit.

Scale: 500 random vectors (moderate patient) + 200 (near-cliff patient) = 700 sims
Estimated time: ~2 minutes
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from constants import (
    INTERVENTION_PARAMS,
    INTERVENTION_NAMES, DEFAULT_INTERVENTION,
    HETEROPLASMY_CLIFF,
)
from simulator import simulate
from analytics import NumpyEncoder

PROJECT = Path(__file__).resolve().parent


# ── Random vector generation ────────────────────────────────────────────────

def random_intervention(rng, clinical_weights=True):
    """Generate a random intervention vector snapped to grid.

    Args:
        rng: numpy RandomState.
        clinical_weights: If True, weight Yamanaka sampling toward low
            doses (clinical feasibility). Grid [0, 0.1, 0.25, 0.5, 0.75, 1.0]
            gets weights [0.30, 0.25, 0.20, 0.15, 0.07, 0.03] — 75% of
            draws are at dose <= 0.25, reflecting that high-intensity
            reprogramming is experimental and extremely costly (3-5 MU ATP).
    """
    # Yamanaka clinical weights: heavily favor low doses
    YAMANAKA_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])

    intervention = {}
    for name, spec in INTERVENTION_PARAMS.items():
        grid = spec["grid"]
        if clinical_weights and name == "yamanaka_intensity":
            weights = YAMANAKA_WEIGHTS[:len(grid)]
            weights = weights / weights.sum()
            intervention[name] = float(rng.choice(grid, p=weights))
        else:
            intervention[name] = float(rng.choice(grid))
    return intervention


# ── Outcome classification ──────────────────────────────────────────────────

def classify_outcome(result, baseline_result):
    """Classify a simulation outcome into a category.

    Args:
        result: simulate() output with intervention.
        baseline_result: simulate() output without intervention.

    Returns:
        (category_string, detail_dict)
    """
    final_atp = float(result["states"][-1, 2])
    final_het = float(result["heteroplasmy"][-1])
    base_atp = float(baseline_result["states"][-1, 2])
    base_het = float(baseline_result["heteroplasmy"][-1])

    # Check for paradoxical: worse than baseline on BOTH metrics
    if final_atp < base_atp - 0.01 and final_het > base_het + 0.01:
        return "paradoxical", {
            "atp_loss": base_atp - final_atp,
            "het_increase": final_het - base_het,
        }

    if final_atp < 0.2:
        return "collapsed", {}
    if final_atp > 0.8 and final_het < 0.5:
        return "thriving", {}
    if final_atp > 0.5 and final_het < HETEROPLASMY_CLIFF:
        return "stable", {}
    return "declining", {}


def identify_culprit(intervention, patient, baseline_result):
    """Find which intervention parameter is most responsible for harm.

    Tests by removing each nonzero parameter one at a time and checking
    if the outcome improves.

    Returns:
        (culprit_param_name, improvement_when_removed)
    """
    best_culprit = None
    best_improvement = 0.0

    for param_name in INTERVENTION_NAMES:
        dose = intervention.get(param_name, 0.0)
        if dose <= 0.0:
            continue

        # Remove this one parameter
        modified = dict(intervention)
        modified[param_name] = 0.0
        result_without = simulate(intervention=modified, patient=patient)
        final_atp_without = float(result_without["states"][-1, 2])
        base_atp = float(baseline_result["states"][-1, 2])

        improvement = final_atp_without - base_atp
        if improvement > best_improvement:
            best_improvement = improvement
            best_culprit = param_name

    return best_culprit, best_improvement


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(n_moderate=500, n_cliff=200, seed=42):
    out_path = PROJECT / "artifacts" / "dark_matter.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    results = []
    start_time = time.time()

    # Patient profiles
    patients = {
        "moderate_60": {
            "baseline_age": 60.0, "baseline_heteroplasmy": 0.40,
            "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.3,
        },
        "near_cliff_75": {
            "baseline_age": 75.0, "baseline_heteroplasmy": 0.60,
            "baseline_nad_level": 0.4, "genetic_vulnerability": 1.25,
            "metabolic_demand": 1.0, "inflammation_level": 0.5,
        },
    }

    patient_trials = [
        ("moderate_60", n_moderate),
        ("near_cliff_75", n_cliff),
    ]

    n_total = n_moderate + n_cliff
    trial_num = 0

    print(f"{'=' * 70}")
    print(f"DARK MATTER EXPERIMENT — Futile Intervention Taxonomy")
    print(f"{'=' * 70}")
    print(f"Moderate patient: {n_moderate} random vectors")
    print(f"Near-cliff patient: {n_cliff} random vectors")
    print(f"Total: {n_total} sims")
    print()

    for pat_id, n_trials in patient_trials:
        patient = patients[pat_id]

        # Compute baseline for this patient
        baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
        base_atp = float(baseline["states"][-1, 2])
        base_het = float(baseline["heteroplasmy"][-1])

        print(f"\n--- {pat_id} (baseline: ATP={base_atp:.3f}, het={base_het:.3f}) ---")

        category_counts = Counter()

        for i in range(n_trials):
            trial_num += 1
            intervention = random_intervention(rng)

            result = simulate(intervention=intervention, patient=patient)
            final_atp = float(result["states"][-1, 2])
            final_het = float(result["heteroplasmy"][-1])

            category, detail = classify_outcome(result, baseline)
            category_counts[category] += 1

            # For paradoxical cases, identify the culprit parameter
            culprit = None
            if category == "paradoxical":
                culprit, _ = identify_culprit(intervention, patient, baseline)

            if trial_num % 100 == 0:
                print(f"  [{trial_num}/{n_total}] categories so far: {dict(category_counts)}")

            results.append({
                "trial": trial_num,
                "patient": pat_id,
                "intervention": intervention,
                "final_atp": final_atp,
                "final_het": final_het,
                "category": category,
                "culprit": culprit,
                "atp_vs_baseline": final_atp - base_atp,
                "het_vs_baseline": final_het - base_het,
            })

        print(f"  {pat_id} categories: {dict(category_counts)}")

    elapsed = time.time() - start_time

    # ── Analysis ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"DARK MATTER ANALYSIS COMPLETE — {elapsed:.1f}s ({len(results)} sims)")
    print(f"{'=' * 70}")

    # Overall taxonomy
    all_categories = Counter(r["category"] for r in results)
    print(f"\nOverall taxonomy:")
    for cat, count in all_categories.most_common():
        pct = 100.0 * count / len(results)
        print(f"  {cat:15s}: {count:4d} ({pct:.1f}%)")

    # Per-patient
    for pat_id, _ in patient_trials:
        pat_results = [r for r in results if r["patient"] == pat_id]
        cats = Counter(r["category"] for r in pat_results)
        print(f"\n{pat_id}:")
        for cat, count in cats.most_common():
            pct = 100.0 * count / len(pat_results)
            print(f"  {cat:15s}: {count:4d} ({pct:.1f}%)")

    # Paradoxical culprit analysis
    paradoxical = [r for r in results if r["category"] == "paradoxical"]
    if paradoxical:
        print(f"\nParadoxical interventions ({len(paradoxical)} total):")
        culprit_counts = Counter(r["culprit"] for r in paradoxical if r["culprit"])
        print(f"  Culprit parameters:")
        for param, count in culprit_counts.most_common():
            print(f"    {param:25s}: {count:3d} ({100*count/len(paradoxical):.0f}%)")

        # Mean intervention vector for paradoxical vs thriving
        thriving = [r for r in results if r["category"] == "thriving"]
        if thriving:
            print(f"\n  Mean intervention doses (paradoxical vs thriving):")
            for param in INTERVENTION_NAMES:
                para_mean = np.mean([r["intervention"][param] for r in paradoxical])
                thrv_mean = np.mean([r["intervention"][param] for r in thriving])
                print(f"    {param:25s}: paradoxical={para_mean:.2f}  thriving={thrv_mean:.2f}")

    # Best and worst interventions
    sorted_by_atp = sorted(results, key=lambda r: r["final_atp"], reverse=True)
    print(f"\nTop 5 best interventions:")
    for r in sorted_by_atp[:5]:
        print(f"  ATP={r['final_atp']:.3f} het={r['final_het']:.3f} [{r['patient']}] "
              f"rapa={r['intervention']['rapamycin_dose']:.2f} "
              f"nad={r['intervention']['nad_supplement']:.2f} "
              f"seno={r['intervention']['senolytic_dose']:.2f} "
              f"yama={r['intervention']['yamanaka_intensity']:.2f}")

    print(f"\nTop 5 worst interventions:")
    for r in sorted_by_atp[-5:]:
        print(f"  ATP={r['final_atp']:.3f} het={r['final_het']:.3f} [{r['patient']}] "
              f"rapa={r['intervention']['rapamycin_dose']:.2f} "
              f"nad={r['intervention']['nad_supplement']:.2f} "
              f"seno={r['intervention']['senolytic_dose']:.2f} "
              f"yama={r['intervention']['yamanaka_intensity']:.2f}")

    # Save
    output = {
        "experiment": "dark_matter",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "seed": seed,
        "n_results": len(results),
        "taxonomy": dict(all_categories),
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
