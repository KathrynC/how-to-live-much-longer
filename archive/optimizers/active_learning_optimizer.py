"""Add-on active-learning surrogate optimizer."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics import NumpyEncoder
from search_addons import PATIENT_PROFILES, make_scalar_objective, random_protocol
from surrogate_optimizer import KNNRegressorSurrogate


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"


def _vec(protocol: dict[str, float]) -> np.ndarray:
    from constants import INTERVENTION_NAMES
    return np.array([float(protocol[k]) for k in INTERVENTION_NAMES], dtype=float)


def _predict_stats(model: KNNRegressorSurrogate, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = model.predict(xq)
    x = model._x
    y = model._y
    k = max(1, min(model.k, len(x)))
    sd = np.zeros(len(xq), dtype=float)
    for i, q in enumerate(xq):
        d = np.linalg.norm(x - q[None, :], axis=1)
        idx = np.argpartition(d, kth=k - 1)[:k]
        sd[i] = float(np.std(y[idx]))
    return mu, sd


def run_active_learning(
    patient_name: str,
    metric: str,
    n_init: int,
    rounds: int,
    propose_per_round: int,
    seed: int,
) -> dict:
    patient = PATIENT_PROFILES[patient_name]
    objective = make_scalar_objective(patient=patient, metric=metric)
    rng = np.random.default_rng(seed)

    records = []
    for _ in range(n_init):
        p = random_protocol(rng)
        m = objective(p)
        records.append({"params": p, **m, "source": "init"})

    round_summaries = []
    for r in range(rounds):
        x = np.array([_vec(v["params"]) for v in records], dtype=float)
        y = np.array([float(v["fitness"]) for v in records], dtype=float)
        model = KNNRegressorSurrogate(k=min(11, len(records))).fit(x, y)

        pool = [random_protocol(rng) for _ in range(max(150, 20 * propose_per_round))]
        xq = np.array([_vec(p) for p in pool], dtype=float)
        mu, sd = _predict_stats(model, xq)
        # Active-learning utility: optimistic + uncertainty seeking.
        utility = mu + 1.1 * sd
        idx = np.argsort(utility)[::-1][:propose_per_round]

        accepted = []
        for i in idx:
            p = pool[int(i)]
            m = objective(p)
            row = {"params": p, **m, "source": f"round_{r}"}
            records.append(row)
            accepted.append(row)

        best = max(records, key=lambda z: float(z["fitness"]))
        round_summaries.append({
            "round": r,
            "accepted": len(accepted),
            "best_fitness_so_far": float(best["fitness"]),
            "mean_new_fitness": float(np.mean([a["fitness"] for a in accepted])) if accepted else float("nan"),
        })

    best = max(records, key=lambda z: float(z["fitness"]))
    return {
        "experiment": "active_learning_optimizer",
        "patient": patient_name,
        "metric": metric,
        "seed": seed,
        "n_init": n_init,
        "rounds": rounds,
        "propose_per_round": propose_per_round,
        "n_evals": len(records),
        "best": best,
        "round_summaries": round_summaries,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Active-learning surrogate optimizer (add-on).")
    parser.add_argument("--patient", default="default", type=str)
    parser.add_argument("--metric", default="combined", type=str)
    parser.add_argument("--n-init", default=24, type=int)
    parser.add_argument("--rounds", default=20, type=int)
    parser.add_argument("--propose-per-round", default=8, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--out", default=None, type=str)
    args = parser.parse_args()
    if args.patient not in PATIENT_PROFILES:
        raise ValueError(f"Unknown patient '{args.patient}'")
    t0 = time.time()
    out = run_active_learning(
        patient_name=args.patient,
        metric=args.metric,
        n_init=args.n_init,
        rounds=args.rounds,
        propose_per_round=args.propose_per_round,
        seed=args.seed,
    )
    out["elapsed_seconds"] = round(time.time() - t0, 2)
    path = Path(args.out) if args.out else ARTIFACTS_DIR / f"active_learning_{args.patient}_{time.strftime('%Y-%m-%d')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"Best fitness: {out['best']['fitness']:.6f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
