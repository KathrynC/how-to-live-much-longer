"""Add-on robust optimizer across patient uncertainty."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics import NumpyEncoder
from search_addons import (
    PATIENT_PROFILES,
    gaussian_mutation,
    make_scalar_objective,
    random_protocol,
)


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"


def _sample_patient_ensemble(base: dict[str, float], rng: np.random.Generator, n: int, rel_sigma: float = 0.08) -> list[dict[str, float]]:
    """Sample perturbed patient profiles around a nominal profile."""
    out = []
    for _ in range(n):
        p = dict(base)
        # Relative perturbations with conservative clipping to declared ranges.
        p["baseline_age"] = float(np.clip(base["baseline_age"] + rng.normal(0.0, 4.0), 20.0, 90.0))
        p["baseline_heteroplasmy"] = float(np.clip(base["baseline_heteroplasmy"] + rng.normal(0.0, 0.05), 0.0, 0.95))
        p["baseline_nad_level"] = float(np.clip(base["baseline_nad_level"] + rng.normal(0.0, 0.06), 0.2, 1.0))
        p["genetic_vulnerability"] = float(np.clip(base["genetic_vulnerability"] * (1.0 + rng.normal(0.0, rel_sigma)), 0.5, 2.0))
        p["metabolic_demand"] = float(np.clip(base["metabolic_demand"] * (1.0 + rng.normal(0.0, rel_sigma)), 0.5, 2.0))
        p["inflammation_level"] = float(np.clip(base["inflammation_level"] + rng.normal(0.0, 0.08), 0.0, 1.0))
        out.append(p)
    return out


def run_robust(patient_name: str, metric: str, budget: int, ensemble_size: int, sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base_patient = PATIENT_PROFILES[patient_name]
    ensemble = _sample_patient_ensemble(base_patient, rng, n=ensemble_size)
    objectives = [make_scalar_objective(p, metric=metric) for p in ensemble]

    def robust_score(params: dict) -> dict:
        vals = []
        for obj in objectives:
            vals.append(float(obj(params)["fitness"]))
        arr = np.array(vals, dtype=float)
        # Robust target: mean - std (risk-adjusted expected fitness).
        score = float(np.mean(arr) - np.std(arr))
        return {
            "robust_fitness": score,
            "mean_fitness": float(np.mean(arr)),
            "std_fitness": float(np.std(arr)),
            "worst_fitness": float(np.min(arr)),
            "best_fitness": float(np.max(arr)),
        }

    current = random_protocol(rng)
    current_metrics = robust_score(current)
    best = {"params": current, **current_metrics}
    history = [{"step": 0, **current_metrics}]

    for step in range(1, budget + 1):
        cand = gaussian_mutation(current, rng, sigma=sigma)
        m = robust_score(cand)
        if m["robust_fitness"] > current_metrics["robust_fitness"]:
            current = cand
            current_metrics = m
        if current_metrics["robust_fitness"] > best["robust_fitness"]:
            best = {"params": current, **current_metrics}
        history.append({"step": step, **current_metrics})

    return {
        "experiment": "robust_optimizer",
        "patient": patient_name,
        "metric": metric,
        "seed": seed,
        "budget": budget,
        "ensemble_size": ensemble_size,
        "sigma": sigma,
        "best": best,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Robust optimizer under patient uncertainty (add-on).")
    parser.add_argument("--patient", default="default", type=str)
    parser.add_argument("--metric", default="combined", type=str)
    parser.add_argument("--budget", default=250, type=int)
    parser.add_argument("--ensemble-size", default=24, type=int)
    parser.add_argument("--sigma", default=0.08, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--out", default=None, type=str)
    args = parser.parse_args()
    if args.patient not in PATIENT_PROFILES:
        raise ValueError(f"Unknown patient '{args.patient}'")
    t0 = time.time()
    out = run_robust(args.patient, args.metric, args.budget, args.ensemble_size, args.sigma, args.seed)
    out["elapsed_seconds"] = round(time.time() - t0, 2)
    path = Path(args.out) if args.out else ARTIFACTS_DIR / f"robust_optimizer_{args.patient}_{time.strftime('%Y-%m-%d')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"Best robust fitness: {out['best']['robust_fitness']:.6f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
