#!/usr/bin/env python3
"""
perturbation_probing.py

Measure intervention fragility by perturbing each of 12 parameters ±1 grid
step around LLM-generated or known-good intervention vectors.

Adapted from perturbation_probing.py in the parent Evolutionary-Robotics
project, which measured "cliffiness" at LLM-generated weight vectors using
a 6-direction perturbation protocol. Here we do the same in 12D space,
measuring how sensitive patient outcomes are to small parameter changes.

For each probe vector:
  - Perturb each of 12 parameters ±1 grid step (24 perturbations)
  - Run simulation at perturbed point
  - Compute sensitivity = |delta_outcome| / |delta_param|
  - Map the fragility landscape

Scale: 50 probe points × 25 sims each = 1250 sims
Estimated time: ~5 minutes (pure simulation, or +LLM if generating fresh vectors)

Can run standalone with built-in probe vectors, or load from oeis/character
experiment results.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    INTERVENTION_PARAMS, PATIENT_PARAMS,
    INTERVENTION_NAMES, ALL_PARAM_NAMES,
    DEFAULT_INTERVENTION,
)
from simulator import simulate
from analytics import NumpyEncoder

# ── Built-in probe vectors ──────────────────────────────────────────────────
# Interesting points in the parameter space to probe even without LLM data.

BUILTIN_PROBES = [
    {"label": "no_treatment_moderate", "intervention": dict(DEFAULT_INTERVENTION),
     "patient": {"baseline_age": 60.0, "baseline_heteroplasmy": 0.40,
                 "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
                 "metabolic_demand": 1.0, "inflammation_level": 0.3}},
    {"label": "full_cocktail_moderate", "intervention": {
        "rapamycin_dose": 0.5, "nad_supplement": 0.5, "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.5},
     "patient": {"baseline_age": 60.0, "baseline_heteroplasmy": 0.40,
                 "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
                 "metabolic_demand": 1.0, "inflammation_level": 0.3}},
    {"label": "near_cliff_no_treatment", "intervention": dict(DEFAULT_INTERVENTION),
     "patient": {"baseline_age": 75.0, "baseline_heteroplasmy": 0.65,
                 "baseline_nad_level": 0.4, "genetic_vulnerability": 1.25,
                 "metabolic_demand": 1.0, "inflammation_level": 0.5}},
    {"label": "near_cliff_with_treatment", "intervention": {
        "rapamycin_dose": 0.75, "nad_supplement": 0.75, "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.0, "transplant_rate": 0.5, "exercise_level": 0.25},
     "patient": {"baseline_age": 75.0, "baseline_heteroplasmy": 0.65,
                 "baseline_nad_level": 0.4, "genetic_vulnerability": 1.25,
                 "metabolic_demand": 1.0, "inflammation_level": 0.5}},
    {"label": "young_biohacker", "intervention": {
        "rapamycin_dose": 0.25, "nad_supplement": 0.5, "senolytic_dose": 0.1,
        "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.75},
     "patient": {"baseline_age": 30.0, "baseline_heteroplasmy": 0.10,
                 "baseline_nad_level": 0.9, "genetic_vulnerability": 0.75,
                 "metabolic_demand": 1.0, "inflammation_level": 0.1}},
    {"label": "yamanaka_aggressive", "intervention": {
        "rapamycin_dose": 0.25, "nad_supplement": 0.75, "senolytic_dose": 0.25,
        "yamanaka_intensity": 0.75, "transplant_rate": 0.0, "exercise_level": 0.1},
     "patient": {"baseline_age": 50.0, "baseline_heteroplasmy": 0.35,
                 "baseline_nad_level": 0.7, "genetic_vulnerability": 1.0,
                 "metabolic_demand": 1.0, "inflammation_level": 0.25}},
    {"label": "high_inflammation", "intervention": {
        "rapamycin_dose": 0.5, "nad_supplement": 0.5, "senolytic_dose": 0.75,
        "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.25},
     "patient": {"baseline_age": 65.0, "baseline_heteroplasmy": 0.45,
                 "baseline_nad_level": 0.5, "genetic_vulnerability": 1.5,
                 "metabolic_demand": 1.25, "inflammation_level": 0.75}},
    {"label": "transplant_heavy", "intervention": {
        "rapamycin_dose": 0.1, "nad_supplement": 0.25, "senolytic_dose": 0.1,
        "yamanaka_intensity": 0.0, "transplant_rate": 1.0, "exercise_level": 0.1},
     "patient": {"baseline_age": 70.0, "baseline_heteroplasmy": 0.55,
                 "baseline_nad_level": 0.5, "genetic_vulnerability": 1.0,
                 "metabolic_demand": 1.0, "inflammation_level": 0.4}},
]


# ── Grid step computation ──────────────────────────────────────────────────

def get_grid_neighbors(param_name, current_value):
    """Get the ±1 grid step neighbors for a parameter.

    Returns (lower_value, upper_value) or None for each if at boundary.
    """
    params = INTERVENTION_PARAMS if param_name in INTERVENTION_PARAMS else PATIENT_PARAMS
    grid = sorted(params[param_name]["grid"])
    lo, hi = params[param_name]["range"]

    # Find nearest grid point
    current_grid = min(grid, key=lambda g: abs(g - current_value))
    idx = grid.index(current_grid)

    lower = grid[idx - 1] if idx > 0 else None
    upper = grid[idx + 1] if idx < len(grid) - 1 else None
    return lower, upper


def evaluate_point(intervention, patient):
    """Simulate and return key outcome metrics."""
    result = simulate(intervention=intervention, patient=patient)
    return {
        "final_atp": float(result["states"][-1, 2]),
        "final_het": float(result["heteroplasmy"][-1]),
        "min_atp": float(np.min(result["states"][:, 2])),
        "mean_atp": float(np.mean(result["states"][:, 2])),
    }


# ── Perturbation probing ──────────────────────────────────────────────────

def probe_point(intervention, patient, label=""):
    """Probe a single point with ±1 grid step perturbations.

    Returns dict with center metrics, per-parameter sensitivities,
    and overall fragility score.
    """
    center = evaluate_point(intervention, patient)
    perturbations = []

    for param_name in ALL_PARAM_NAMES:
        is_intervention = param_name in INTERVENTION_NAMES
        current_val = (intervention.get(param_name, 0.0) if is_intervention
                       else patient.get(param_name, 0.0))
        lower, upper = get_grid_neighbors(param_name, current_val)

        for direction, new_val in [("lower", lower), ("upper", upper)]:
            if new_val is None:
                continue

            # Create perturbed version
            if is_intervention:
                perturbed_intv = dict(intervention)
                perturbed_intv[param_name] = new_val
                perturbed_pat = patient
            else:
                perturbed_intv = intervention
                perturbed_pat = dict(patient)
                perturbed_pat[param_name] = new_val

            metrics = evaluate_point(perturbed_intv, perturbed_pat)
            delta_param = abs(new_val - current_val)
            delta_atp = abs(metrics["final_atp"] - center["final_atp"])
            delta_het = abs(metrics["final_het"] - center["final_het"])
            sensitivity_atp = delta_atp / delta_param if delta_param > 1e-12 else 0.0
            sensitivity_het = delta_het / delta_param if delta_param > 1e-12 else 0.0

            perturbations.append({
                "param": param_name,
                "direction": direction,
                "original_value": current_val,
                "perturbed_value": new_val,
                "delta_param": delta_param,
                "delta_atp": delta_atp,
                "delta_het": delta_het,
                "sensitivity_atp": sensitivity_atp,
                "sensitivity_het": sensitivity_het,
                "perturbed_metrics": metrics,
            })

    # Aggregate sensitivities per parameter
    param_sensitivities = {}
    for param_name in ALL_PARAM_NAMES:
        param_perts = [p for p in perturbations if p["param"] == param_name]
        if param_perts:
            mean_sens_atp = np.mean([p["sensitivity_atp"] for p in param_perts])
            mean_sens_het = np.mean([p["sensitivity_het"] for p in param_perts])
            param_sensitivities[param_name] = {
                "sensitivity_atp": float(mean_sens_atp),
                "sensitivity_het": float(mean_sens_het),
                "combined": float(mean_sens_atp + mean_sens_het),
            }

    # Overall fragility score (sum of all sensitivities)
    fragility = sum(s["combined"] for s in param_sensitivities.values())

    # Most and least sensitive parameters
    sorted_sens = sorted(param_sensitivities.items(),
                         key=lambda x: x[1]["combined"], reverse=True)
    most_sensitive = sorted_sens[0][0] if sorted_sens else "none"
    least_sensitive = sorted_sens[-1][0] if sorted_sens else "none"

    return {
        "label": label,
        "center_metrics": center,
        "n_perturbations": len(perturbations),
        "fragility_score": fragility,
        "most_sensitive_param": most_sensitive,
        "least_sensitive_param": least_sensitive,
        "param_sensitivities": param_sensitivities,
        "perturbations": perturbations,
    }


# ── Load LLM-generated vectors ────────────────────────────────────────────

def load_llm_vectors(max_vectors=42):
    """Try to load vectors from OEIS or character seed experiment results."""
    vectors = []

    for filename in ["oeis_seed_experiment.json", "character_seed_experiment.json"]:
        path = PROJECT / "artifacts" / filename
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if r.get("success") and r.get("intervention") and r.get("patient"):
                label = r.get("seq_id") or r.get("character_story", "unknown")
                vectors.append({
                    "label": f"{filename.split('_')[0]}:{label}:{r.get('model', '?')}",
                    "intervention": r["intervention"],
                    "patient": r["patient"],
                })
                if len(vectors) >= max_vectors:
                    return vectors
    return vectors


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    out_path = PROJECT / "artifacts" / "perturbation_probing.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Collect probe vectors: builtin + LLM-generated
    probes = list(BUILTIN_PROBES)
    llm_probes = load_llm_vectors(max_vectors=42)
    if llm_probes:
        print(f"Loaded {len(llm_probes)} LLM-generated vectors")
        probes.extend(llm_probes)
    else:
        print("No LLM experiment results found — using only builtin probes")

    n_total = len(probes)
    print(f"{'=' * 70}")
    print(f"PERTURBATION PROBING — Intervention Fragility")
    print(f"{'=' * 70}")
    print(f"Probe points: {n_total} ({len(BUILTIN_PROBES)} builtin + "
          f"{len(llm_probes)} LLM-generated)")
    print(f"Perturbations per point: ~24")
    print(f"Estimated sims: ~{n_total * 25}")
    print()

    results = []
    for i, probe in enumerate(probes):
        label = probe.get("label", f"probe_{i}")
        print(f"[{i+1}/{n_total}] {label[:60]}", end=" ", flush=True)

        result = probe_point(probe["intervention"], probe["patient"], label)
        results.append(result)

        print(f"-> fragility={result['fragility_score']:.3f} "
              f"most_sensitive={result['most_sensitive_param']}")

    elapsed = time.time() - start_time

    # ── Analysis ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PERTURBATION PROBING COMPLETE — {elapsed:.1f}s ({len(results)} probes)")
    print(f"{'=' * 70}")

    # Fragility ranking
    sorted_by_fragility = sorted(results, key=lambda r: r["fragility_score"],
                                  reverse=True)
    print(f"\nTop 10 most fragile intervention points:")
    for r in sorted_by_fragility[:10]:
        print(f"  {r['label'][:50]:50s}: fragility={r['fragility_score']:.3f} "
              f"sensitive_to={r['most_sensitive_param']}")

    print(f"\nTop 5 most robust points:")
    for r in sorted_by_fragility[-5:]:
        print(f"  {r['label'][:50]:50s}: fragility={r['fragility_score']:.3f}")

    # Parameter sensitivity ranking (averaged across all probes)
    from collections import defaultdict
    param_total_sens = defaultdict(list)
    for r in results:
        for param, sens in r["param_sensitivities"].items():
            param_total_sens[param].append(sens["combined"])

    print(f"\nParameter sensitivity ranking (mean across all probes):")
    sorted_params = sorted(param_total_sens.items(),
                           key=lambda x: np.mean(x[1]), reverse=True)
    for param, sens_list in sorted_params:
        print(f"  {param:25s}: mean_sensitivity={np.mean(sens_list):.4f} "
              f"(max={max(sens_list):.4f})")

    # Save
    output = {
        "experiment": "perturbation_probing",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "n_probes": len(results),
        "n_builtin": len(BUILTIN_PROBES),
        "n_llm": len(llm_probes),
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
