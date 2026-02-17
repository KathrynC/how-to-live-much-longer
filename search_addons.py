"""Shared utilities for additive search methods.

These helpers centralize:
  - parameter bounds and clipping
  - mutation/crossover operators
  - objective wrappers around simulator + analytics
  - non-dominated sorting utilities

All methods here are additive and keep simulator/analytics as ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from analytics import compute_all
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT, INTERVENTION_NAMES, INTERVENTION_PARAMS
from simulator import simulate


PATIENT_PROFILES = {
    "default": DEFAULT_PATIENT,
    "near_cliff_80": {
        "baseline_age": 80.0,
        "baseline_heteroplasmy": 0.65,
        "baseline_nad_level": 0.4,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.5,
    },
    "young_prevention_25": {
        "baseline_age": 25.0,
        "baseline_heteroplasmy": 0.05,
        "baseline_nad_level": 0.95,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.0,
    },
    "melas_35": {
        "baseline_age": 35.0,
        "baseline_heteroplasmy": 0.50,
        "baseline_nad_level": 0.6,
        "genetic_vulnerability": 2.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.25,
    },
    "sarcopenia_75": {
        "baseline_age": 75.0,
        "baseline_heteroplasmy": 0.45,
        "baseline_nad_level": 0.4,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.5,
        "inflammation_level": 0.5,
    },
    "post_chemo_55": {
        "baseline_age": 55.0,
        "baseline_heteroplasmy": 0.40,
        "baseline_nad_level": 0.4,
        "genetic_vulnerability": 1.25,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.5,
    },
}


def bounds_dict() -> dict[str, tuple[float, float]]:
    """Return intervention bounds keyed by parameter name."""
    return {k: tuple(v["range"]) for k, v in INTERVENTION_PARAMS.items()}


def clip_protocol(protocol: dict[str, float]) -> dict[str, float]:
    """Clamp protocol into valid intervention bounds and canonical key set."""
    b = bounds_dict()
    return {k: float(np.clip(float(protocol.get(k, 0.0)), b[k][0], b[k][1])) for k in INTERVENTION_NAMES}


def vectorize(protocol: dict[str, float]) -> np.ndarray:
    """Convert intervention dict into ordered vector."""
    return np.array([float(protocol[k]) for k in INTERVENTION_NAMES], dtype=float)


def devectorize(x: np.ndarray) -> dict[str, float]:
    """Convert ordered vector into intervention dict."""
    return {k: float(v) for k, v in zip(INTERVENTION_NAMES, x)}


def random_protocol(rng: np.random.Generator) -> dict[str, float]:
    """Uniform random intervention in configured bounds."""
    b = bounds_dict()
    out = {}
    for k in INTERVENTION_NAMES:
        lo, hi = b[k]
        out[k] = float(rng.uniform(lo, hi))
    return out


def gaussian_mutation(protocol: dict[str, float], rng: np.random.Generator, sigma: float = 0.08) -> dict[str, float]:
    """Bounded Gaussian mutation around a protocol."""
    x = vectorize(clip_protocol(protocol))
    x = x + rng.normal(0.0, sigma, size=len(x))
    b = bounds_dict()
    for i, k in enumerate(INTERVENTION_NAMES):
        lo, hi = b[k]
        x[i] = float(np.clip(x[i], lo, hi))
    return devectorize(x)


def blend_crossover(a: dict[str, float], b: dict[str, float], rng: np.random.Generator, alpha: float = 0.5) -> dict[str, float]:
    """Simple convex blend crossover between two protocols."""
    xa = vectorize(clip_protocol(a))
    xb = vectorize(clip_protocol(b))
    lam = rng.uniform(0.0, 1.0) if alpha == 0.5 else alpha
    x = lam * xa + (1.0 - lam) * xb
    return clip_protocol(devectorize(x))


def make_scalar_objective(patient: dict[str, float], metric: str = "combined") -> Callable[[dict[str, float]], dict[str, float]]:
    """Build scalar objective from simulator + analytics.

    Metrics:
      - combined: atp_benefit + 0.5 * het_benefit
      - atp
      - het
      - crisis_delay
    """
    baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
    base_atp = float(baseline["states"][-1, 2])
    base_het = float(baseline["heteroplasmy"][-1])

    def objective(protocol: dict[str, float]) -> dict[str, float]:
        p = clip_protocol(protocol)
        res = simulate(intervention=p, patient=patient)
        analytics = compute_all(res, baseline)
        final_atp = float(res["states"][-1, 2])
        final_het = float(res["heteroplasmy"][-1])
        atp_benefit = final_atp - base_atp
        het_benefit = base_het - final_het
        if metric == "atp":
            fitness = atp_benefit
        elif metric == "het":
            fitness = het_benefit
        elif metric == "crisis_delay":
            fitness = float(analytics.get("intervention", {}).get("crisis_delay_years", 0.0))
        else:
            fitness = atp_benefit + 0.5 * het_benefit
        return {
            "fitness": float(fitness),
            "final_atp": final_atp,
            "final_het": final_het,
            "atp_benefit": atp_benefit,
            "het_benefit": het_benefit,
            "crisis_delay_years": float(analytics.get("intervention", {}).get("crisis_delay_years", 0.0)),
            "total_dose": float(sum(p.values())),
        }

    return objective


def multi_objectives(protocol: dict[str, float], objective_fn: Callable[[dict[str, float]], dict[str, float]]) -> dict[str, float]:
    """Convert scalar-objective metrics to a standard multi-objective record.

    We maximize:
      - ATP benefit
      - heteroplasmy benefit
      - crisis delay
    and minimize:
      - total dose
      - final heteroplasmy
    """
    m = objective_fn(protocol)
    return {
        "atp_benefit": float(m["atp_benefit"]),
        "het_benefit": float(m["het_benefit"]),
        "crisis_delay_years": float(m["crisis_delay_years"]),
        "neg_total_dose": float(-m["total_dose"]),
        "neg_final_het": float(-m["final_het"]),
        "fitness": float(m["fitness"]),
        "final_atp": float(m["final_atp"]),
        "final_het": float(m["final_het"]),
        "total_dose": float(m["total_dose"]),
    }


def dominates(a: dict[str, float], b: dict[str, float], keys: list[str]) -> bool:
    """Return True iff a Pareto-dominates b on maximize-keys."""
    ge = all(float(a[k]) >= float(b[k]) for k in keys)
    gt = any(float(a[k]) > float(b[k]) for k in keys)
    return ge and gt


def non_dominated_sort(rows: list[dict], keys: list[str]) -> list[list[int]]:
    """Fast-enough non-dominated sorting for moderate population sizes."""
    n = len(rows)
    dominates_set = [set() for _ in range(n)]
    dominated_count = [0] * n
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(rows[i], rows[j], keys):
                dominates_set[i].add(j)
                dominated_count[j] += 1
            elif dominates(rows[j], rows[i], keys):
                dominates_set[j].add(i)
                dominated_count[i] += 1

    for i in range(n):
        if dominated_count[i] == 0:
            fronts[0].append(i)

    f = 0
    while f < len(fronts) and fronts[f]:
        next_front = []
        for i in fronts[f]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        f += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def crowding_distance(rows: list[dict], idxs: list[int], keys: list[str]) -> dict[int, float]:
    """Compute NSGA-II crowding distance for one front."""
    if not idxs:
        return {}
    dist = {i: 0.0 for i in idxs}
    if len(idxs) <= 2:
        for i in idxs:
            dist[i] = float("inf")
        return dist
    for k in keys:
        sorted_idxs = sorted(idxs, key=lambda i: float(rows[i][k]))
        dist[sorted_idxs[0]] = float("inf")
        dist[sorted_idxs[-1]] = float("inf")
        lo = float(rows[sorted_idxs[0]][k])
        hi = float(rows[sorted_idxs[-1]][k])
        span = max(hi - lo, 1e-12)
        for p in range(1, len(sorted_idxs) - 1):
            i = sorted_idxs[p]
            if np.isinf(dist[i]):
                continue
            prev_v = float(rows[sorted_idxs[p - 1]][k])
            next_v = float(rows[sorted_idxs[p + 1]][k])
            dist[i] += (next_v - prev_v) / span
    return dist

