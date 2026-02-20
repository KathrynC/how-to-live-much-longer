"""ML prefilter runner: additive surrogate ranking before true simulation.

This script does not alter existing analytics or simulator behavior.
It adds a machine-learning screening layer to prioritize candidates:
  1. Train surrogate on simulator-evaluated samples.
  2. Score a large candidate pool with surrogate.
  3. Re-evaluate top-K with the true simulator objective.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics import NumpyEncoder
from constants import INTERVENTION_NAMES, INTERVENTION_PARAMS
from gradient_refiner import load_seed_protocol
from surrogate_optimizer import (
    PATIENT_PROFILES,
    KNNRegressorSurrogate,
    build_training_data,
    clip_intervention,
    encode_features,
    evaluate_candidates,
    make_objective,
    random_intervention,
)


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"


def _perturb_seed(
    seed_iv: dict[str, float],
    rng: np.random.Generator,
    sigma: float,
    n: int,
) -> list[dict[str, float]]:
    """Generate bounded Gaussian perturbations around one seed protocol."""
    out = []
    for _ in range(n):
        row = {}
        for name in INTERVENTION_NAMES:
            lo, hi = INTERVENTION_PARAMS[name]["range"]
            val = float(seed_iv.get(name, 0.0) + rng.normal(0.0, sigma))
            row[name] = float(np.clip(val, lo, hi))
        out.append(row)
    return out


def build_candidate_pool(
    pool_size: int,
    rng: np.random.Generator,
    seed_protocols: list[dict[str, float]] | None = None,
    perturb_per_seed: int = 150,
    perturb_sigma: float = 0.08,
) -> list[dict[str, float]]:
    """Build candidate pool from random samples + local seed perturbations.

    Parameters
    ----------
    pool_size:
        Number of purely random interventions.
    rng:
        Numpy random generator.
    seed_protocols:
        Optional list of anchor protocols (e.g., EA champions).
    perturb_per_seed:
        Number of local neighbors per seed.
    perturb_sigma:
        Perturbation standard deviation in normalized parameter space.
    """
    pool = [random_intervention(rng) for _ in range(pool_size)]

    if seed_protocols:
        for iv in seed_protocols:
            pool.append(clip_intervention(iv))
            pool.extend(
                _perturb_seed(
                    clip_intervention(iv),
                    rng=rng,
                    sigma=perturb_sigma,
                    n=perturb_per_seed,
                )
            )
    return pool


def run_prefilter(
    patient_name: str,
    metric: str,
    train_samples: int,
    pool_size: int,
    top_k: int,
    seed: int,
    seed_protocols: list[dict[str, float]] | None = None,
    perturb_per_seed: int = 150,
    perturb_sigma: float = 0.08,
    knn_k: int = 11,
) -> dict:
    """Run full surrogate-prefilter experiment and return artifact dict.

    Workflow
    --------
    1. Train surrogate from true simulator labels.
    2. Score candidate pool with surrogate.
    3. Re-evaluate top-K with true objective.
    4. Compare top-K true fitness against random-control true fitness.
    """
    patient = PATIENT_PROFILES[patient_name]
    rng = np.random.default_rng(seed)

    objective = make_objective(patient=patient, metric=metric)

    # 1) Build supervised training set from true simulator calls.
    ds = build_training_data(
        patient=patient,
        n_samples=train_samples,
        seed=seed,
        objective_fn=objective,
    )
    x_train = ds["x"]
    y_train = ds["y"]
    model = KNNRegressorSurrogate(k=knn_k).fit(x_train, y_train)

    # 2) Build and score candidate pool.
    pool = build_candidate_pool(
        pool_size=pool_size,
        rng=rng,
        seed_protocols=seed_protocols,
        perturb_per_seed=perturb_per_seed,
        perturb_sigma=perturb_sigma,
    )
    x_pool = np.array([encode_features(iv, patient) for iv in pool], dtype=float)
    y_pred = model.predict(x_pool)

    k = min(top_k, len(pool))
    top_idx = np.argsort(y_pred)[::-1][:k]
    top_candidates = [pool[i] for i in top_idx]
    top_pred = [float(y_pred[i]) for i in top_idx]

    # Random-control baseline from same pool for efficacy sanity-check.
    rand_idx = rng.choice(len(pool), size=k, replace=False)
    random_candidates = [pool[int(i)] for i in rand_idx]

    # 3) True simulator re-evaluation.
    top_true = evaluate_candidates(top_candidates, objective_fn=objective)
    random_true = evaluate_candidates(random_candidates, objective_fn=objective)

    mean_top_true = float(np.mean([r["fitness"] for r in top_true])) if top_true else float("nan")
    mean_random_true = float(np.mean([r["fitness"] for r in random_true])) if random_true else float("nan")

    ranked = []
    for rank, (pred, row) in enumerate(zip(top_pred, top_true), start=1):
        ranked.append({
            "rank": rank,
            "predicted_fitness": pred,
            **row,
        })

    return {
        "experiment": "ml_prefilter_runner",
        "patient": patient_name,
        "metric": metric,
        "seed": seed,
        "train_samples": train_samples,
        "pool_size": len(pool),
        "top_k": k,
        "knn_k": knn_k,
        "seed_protocol_count": len(seed_protocols or []),
        "mean_top_true_fitness": mean_top_true,
        "mean_random_true_fitness": mean_random_true,
        "mean_true_lift_vs_random": mean_top_true - mean_random_true,
        "top_ranked_true": ranked,
        "random_control_true": random_true,
    }


def _default_out(patient_name: str) -> Path:
    """Default artifact path for one prefilter run."""
    stamp = time.strftime("%Y-%m-%d")
    return ARTIFACTS_DIR / f"ml_prefilter_{patient_name}_{stamp}.json"


def main():
    """CLI entry point for surrogate prefilter experiments."""
    parser = argparse.ArgumentParser(description="Run surrogate-based candidate prefiltering.")
    parser.add_argument("--patient", type=str, default="default", help=f"Patient profile ({', '.join(PATIENT_PROFILES)}).")
    parser.add_argument("--metric", type=str, default="combined", help="Fitness metric (combined|atp|het|crisis_delay).")
    parser.add_argument("--train-samples", type=int, default=160, help="True simulator samples for surrogate training.")
    parser.add_argument("--pool-size", type=int, default=1200, help="Random candidate pool size before surrogate ranking.")
    parser.add_argument("--top-k", type=int, default=24, help="Top-K surrogate-ranked candidates to re-evaluate.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--knn-k", type=int, default=11, help="K for KNN surrogate.")
    parser.add_argument("--seed-artifact", action="append", default=[], help="Artifact path(s) containing seed best_params.")
    parser.add_argument("--profile", type=str, default=None, help="Optional profile key for multi-profile artifacts.")
    parser.add_argument("--perturb-per-seed", type=int, default=150, help="Number of local perturbations per seed protocol.")
    parser.add_argument("--perturb-sigma", type=float, default=0.08, help="Gaussian sigma for seed perturbations.")
    parser.add_argument("--out", type=str, default=None, help="Output artifact path.")
    args = parser.parse_args()

    if args.patient not in PATIENT_PROFILES:
        raise ValueError(f"Unknown patient '{args.patient}'. Choose from {sorted(PATIENT_PROFILES)}.")

    t0 = time.time()

    seed_protocols = []
    for artifact in args.seed_artifact:
        seed_protocols.append(load_seed_protocol(artifact, profile=args.profile))

    result = run_prefilter(
        patient_name=args.patient,
        metric=args.metric,
        train_samples=args.train_samples,
        pool_size=args.pool_size,
        top_k=args.top_k,
        seed=args.seed,
        seed_protocols=seed_protocols,
        perturb_per_seed=args.perturb_per_seed,
        perturb_sigma=args.perturb_sigma,
        knn_k=args.knn_k,
    )
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    result["seed_artifacts"] = list(args.seed_artifact)
    result["profile"] = args.profile

    out_path = Path(args.out) if args.out else _default_out(args.patient)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))

    print("ML Prefilter Runner")
    print(f"Patient: {result['patient']}")
    print(f"Metric: {result['metric']}")
    print(f"Pool size: {result['pool_size']}, top_k: {result['top_k']}")
    print(f"Mean true top-k fitness: {result['mean_top_true_fitness']:.6f}")
    print(f"Mean random-control fitness: {result['mean_random_true_fitness']:.6f}")
    print(f"Lift vs random: {result['mean_true_lift_vs_random']:+.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
