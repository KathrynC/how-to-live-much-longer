"""Add-on MAP-Elites quality-diversity optimizer."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics import NumpyEncoder
from search_addons import PATIENT_PROFILES, gaussian_mutation, make_scalar_objective, random_protocol


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"


def _cell_index(value: float, lo: float, hi: float, bins: int) -> int:
    if value <= lo:
        return 0
    if value >= hi:
        return bins - 1
    t = (value - lo) / (hi - lo)
    return int(np.floor(t * bins))


def run_map_elites(patient_name: str, metric: str, budget: int, bins_atp: int, bins_het: int, sigma: float, seed: int) -> dict:
    patient = PATIENT_PROFILES[patient_name]
    objective = make_scalar_objective(patient=patient, metric=metric)
    rng = np.random.default_rng(seed)

    # Descriptor space: (final_atp, final_het). We seek diverse elites across cells.
    atp_lo, atp_hi = 0.0, 1.1
    het_lo, het_hi = 0.0, 1.0

    archive: dict[tuple[int, int], dict] = {}

    def try_insert(protocol: dict[str, float]):
        m = objective(protocol)
        i = _cell_index(float(m["final_atp"]), atp_lo, atp_hi, bins_atp)
        j = _cell_index(float(m["final_het"]), het_lo, het_hi, bins_het)
        key = (i, j)
        row = {"params": protocol, **m, "cell": [i, j]}
        if key not in archive or float(row["fitness"]) > float(archive[key]["fitness"]):
            archive[key] = row

    # Bootstrap with random solutions.
    init = max(40, budget // 6)
    for _ in range(init):
        try_insert(random_protocol(rng))

    # Emit from random elites and mutate.
    for _ in range(max(0, budget - init)):
        if archive:
            keys = list(archive.keys())
            key = keys[int(rng.integers(0, len(keys)))]
            elite = archive[key]["params"]
        else:
            elite = random_protocol(rng)
        child = gaussian_mutation(elite, rng, sigma=sigma)
        try_insert(child)

    elites = list(archive.values())
    best = max(elites, key=lambda e: float(e["fitness"])) if elites else None
    coverage = len(elites) / float(bins_atp * bins_het)
    return {
        "experiment": "map_elites_optimizer",
        "patient": patient_name,
        "metric": metric,
        "seed": seed,
        "budget": budget,
        "bins_atp": bins_atp,
        "bins_het": bins_het,
        "sigma": sigma,
        "coverage": coverage,
        "n_elites": len(elites),
        "best": best,
        "elites": elites,
    }


def main():
    parser = argparse.ArgumentParser(description="MAP-Elites optimizer (add-on).")
    parser.add_argument("--patient", default="default", type=str)
    parser.add_argument("--metric", default="combined", type=str)
    parser.add_argument("--budget", default=600, type=int)
    parser.add_argument("--bins-atp", default=14, type=int)
    parser.add_argument("--bins-het", default=14, type=int)
    parser.add_argument("--sigma", default=0.08, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--out", default=None, type=str)
    args = parser.parse_args()
    if args.patient not in PATIENT_PROFILES:
        raise ValueError(f"Unknown patient '{args.patient}'")
    t0 = time.time()
    out = run_map_elites(
        patient_name=args.patient,
        metric=args.metric,
        budget=args.budget,
        bins_atp=args.bins_atp,
        bins_het=args.bins_het,
        sigma=args.sigma,
        seed=args.seed,
    )
    out["elapsed_seconds"] = round(time.time() - t0, 2)
    path = Path(args.out) if args.out else ARTIFACTS_DIR / f"map_elites_{args.patient}_{time.strftime('%Y-%m-%d')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"Coverage: {out['coverage']:.3f} ({out['n_elites']} elites)")
    print(f"Best fitness: {out['best']['fitness']:.6f}" if out["best"] else "No elites.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
