"""Constrained Bayesian-style optimizer (surrogate + UCB acquisition).

This is an additive tool: it prioritizes candidates with a surrogate but uses
the true simulator objective for all accepted evaluations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics import NumpyEncoder
from search_addons import (
    PATIENT_PROFILES,
    clip_protocol,
)
from surrogate_optimizer import KNNRegressorSurrogate
from search_addons import make_scalar_objective, random_protocol


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"


def _feature_vec(protocol: dict[str, float]) -> np.ndarray:
    # Only intervention dimensions are used here (single fixed patient per run).
    from constants import INTERVENTION_NAMES
    return np.array([float(protocol[k]) for k in INTERVENTION_NAMES], dtype=float)


def _predict_with_uncertainty(model: KNNRegressorSurrogate, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate predictive uncertainty by local neighbor spread."""
    pred = model.predict(xq)
    x_train = model._x
    y_train = model._y
    k = max(1, min(model.k, len(x_train)))
    std = np.zeros(len(xq), dtype=float)
    for i, q in enumerate(xq):
        d = np.linalg.norm(x_train - q[None, :], axis=1)
        idx = np.argpartition(d, kth=k - 1)[:k]
        std[i] = float(np.std(y_train[idx]))
    return pred, std


def run_bo(
    patient_name: str,
    metric: str,
    n_init: int,
    iterations: int,
    candidate_pool: int,
    kappa: float,
    seed: int,
) -> dict:
    patient = PATIENT_PROFILES[patient_name]
    objective = make_scalar_objective(patient=patient, metric=metric)
    rng = np.random.default_rng(seed)

    # Initial design.
    samples = []
    for _ in range(n_init):
        p = random_protocol(rng)
        m = objective(p)
        samples.append({"params": clip_protocol(p), **m})

    for _ in range(iterations):
        x_train = np.array([_feature_vec(s["params"]) for s in samples], dtype=float)
        y_train = np.array([float(s["fitness"]) for s in samples], dtype=float)
        model = KNNRegressorSurrogate(k=min(11, len(samples))).fit(x_train, y_train)

        # Surrogate acquisition maximization over random proposals.
        pool = [random_protocol(rng) for _ in range(candidate_pool)]
        xq = np.array([_feature_vec(p) for p in pool], dtype=float)
        mu, sig = _predict_with_uncertainty(model, xq)
        acq = mu + kappa * sig  # UCB
        best_idx = int(np.argmax(acq))
        cand = pool[best_idx]

        m = objective(cand)
        samples.append({"params": clip_protocol(cand), **m, "acq": float(acq[best_idx])})

    best = max(samples, key=lambda r: float(r["fitness"]))
    return {
        "experiment": "bayesian_optimizer",
        "patient": patient_name,
        "metric": metric,
        "seed": seed,
        "n_init": n_init,
        "iterations": iterations,
        "candidate_pool": candidate_pool,
        "kappa": kappa,
        "n_evals": len(samples),
        "best": best,
        "fitness_trajectory": [float(r["fitness"]) for r in samples],
        "records": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Add-on Bayesian-style optimizer (UCB over KNN surrogate).")
    parser.add_argument("--patient", default="default", type=str)
    parser.add_argument("--metric", default="combined", type=str)
    parser.add_argument("--n-init", default=24, type=int)
    parser.add_argument("--iterations", default=120, type=int)
    parser.add_argument("--candidate-pool", default=400, type=int)
    parser.add_argument("--kappa", default=1.0, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--out", default=None, type=str)
    args = parser.parse_args()

    if args.patient not in PATIENT_PROFILES:
        raise ValueError(f"Unknown patient '{args.patient}'")

    t0 = time.time()
    out = run_bo(
        patient_name=args.patient,
        metric=args.metric,
        n_init=args.n_init,
        iterations=args.iterations,
        candidate_pool=args.candidate_pool,
        kappa=args.kappa,
        seed=args.seed,
    )
    out["elapsed_seconds"] = round(time.time() - t0, 2)
    path = Path(args.out) if args.out else ARTIFACTS_DIR / f"bayes_optimizer_{args.patient}_{time.strftime('%Y-%m-%d')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"Best fitness: {out['best']['fitness']:.6f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
