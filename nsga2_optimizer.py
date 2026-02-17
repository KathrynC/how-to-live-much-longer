"""Add-on NSGA-II search for multi-objective intervention discovery."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics import NumpyEncoder
from search_addons import (
    PATIENT_PROFILES,
    blend_crossover,
    crowding_distance,
    gaussian_mutation,
    make_scalar_objective,
    multi_objectives,
    non_dominated_sort,
    random_protocol,
)


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"
PARETO_KEYS = ["atp_benefit", "het_benefit", "crisis_delay_years", "neg_total_dose", "neg_final_het"]


def _tournament_select(pop: list[dict], ranks: dict[int, int], crowd: dict[int, float], rng: np.random.Generator) -> dict:
    i, j = rng.integers(0, len(pop), size=2)
    ri, rj = ranks[i], ranks[j]
    if ri < rj:
        return pop[i]
    if rj < ri:
        return pop[j]
    ci = crowd.get(i, 0.0)
    cj = crowd.get(j, 0.0)
    return pop[i] if ci >= cj else pop[j]


def run_nsga2(patient_name: str, metric: str, pop_size: int, generations: int, sigma: float, seed: int) -> dict:
    patient = PATIENT_PROFILES[patient_name]
    objective = make_scalar_objective(patient=patient, metric=metric)
    rng = np.random.default_rng(seed)

    def evaluate(params: dict) -> dict:
        mo = multi_objectives(params, objective)
        return {"params": params, **mo}

    pop = [evaluate(random_protocol(rng)) for _ in range(pop_size)]

    for _ in range(generations):
        fronts = non_dominated_sort(pop, keys=PARETO_KEYS)
        ranks = {}
        crowd = {}
        for r, front in enumerate(fronts):
            for i in front:
                ranks[i] = r
            crowd.update(crowding_distance(pop, front, PARETO_KEYS))

        children = []
        while len(children) < pop_size:
            p1 = _tournament_select(pop, ranks, crowd, rng)["params"]
            p2 = _tournament_select(pop, ranks, crowd, rng)["params"]
            c = blend_crossover(p1, p2, rng)
            c = gaussian_mutation(c, rng, sigma=sigma)
            children.append(evaluate(c))

        combined = pop + children
        fronts = non_dominated_sort(combined, keys=PARETO_KEYS)
        next_pop = []
        for front in fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend([combined[i] for i in front])
            else:
                cd = crowding_distance(combined, front, PARETO_KEYS)
                order = sorted(front, key=lambda i: cd.get(i, 0.0), reverse=True)
                need = pop_size - len(next_pop)
                next_pop.extend([combined[i] for i in order[:need]])
                break
        pop = next_pop

    final_fronts = non_dominated_sort(pop, keys=PARETO_KEYS)
    pareto = [pop[i] for i in final_fronts[0]] if final_fronts else []
    best_scalar = max(pop, key=lambda r: float(r["fitness"]))
    return {
        "experiment": "nsga2_optimizer",
        "patient": patient_name,
        "metric": metric,
        "seed": seed,
        "population_size": pop_size,
        "generations": generations,
        "sigma": sigma,
        "n_evals": pop_size * (generations + 1),
        "pareto_count": len(pareto),
        "pareto_front": pareto,
        "best_scalar": best_scalar,
    }


def main():
    parser = argparse.ArgumentParser(description="NSGA-II multi-objective optimizer (add-on).")
    parser.add_argument("--patient", default="default", type=str)
    parser.add_argument("--metric", default="combined", type=str)
    parser.add_argument("--pop-size", default=48, type=int)
    parser.add_argument("--generations", default=40, type=int)
    parser.add_argument("--sigma", default=0.08, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--out", default=None, type=str)
    args = parser.parse_args()
    if args.patient not in PATIENT_PROFILES:
        raise ValueError(f"Unknown patient '{args.patient}'")
    t0 = time.time()
    out = run_nsga2(args.patient, args.metric, args.pop_size, args.generations, args.sigma, args.seed)
    out["elapsed_seconds"] = round(time.time() - t0, 2)
    path = Path(args.out) if args.out else ARTIFACTS_DIR / f"nsga2_{args.patient}_{time.strftime('%Y-%m-%d')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"Pareto count: {out['pareto_count']}")
    print(f"Best scalar fitness: {out['best_scalar']['fitness']:.6f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
