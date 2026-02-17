"""Surrogate-model utilities for sample-efficient protocol search.

This module is strictly additive. It does not replace simulator or analytics
logic; it learns an approximate mapping from parameters to fitness, then uses
that approximation to rank candidate protocols before expensive simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from analytics import compute_all
from constants import (
    DEFAULT_INTERVENTION,
    DEFAULT_PATIENT,
    INTERVENTION_NAMES,
    INTERVENTION_PARAMS,
    PATIENT_NAMES,
)
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


def clip_intervention(intervention: dict[str, float]) -> dict[str, float]:
    """Project intervention values into configured parameter bounds.

    Parameters
    ----------
    intervention:
        Partial or full intervention dictionary.

    Returns
    -------
    dict
        Full intervention dictionary over `INTERVENTION_NAMES`, clamped to
        each parameter's configured range.
    """
    out = {}
    for name in INTERVENTION_NAMES:
        lo, hi = INTERVENTION_PARAMS[name]["range"]
        out[name] = float(np.clip(float(intervention.get(name, 0.0)), lo, hi))
    return out


def random_intervention(rng: np.random.Generator) -> dict[str, float]:
    """Sample one intervention uniformly over each parameter range.

    Parameters
    ----------
    rng:
        Numpy random generator.

    Returns
    -------
    dict
        Intervention dictionary with one sampled value per intervention key.
    """
    sample = {}
    for name in INTERVENTION_NAMES:
        lo, hi = INTERVENTION_PARAMS[name]["range"]
        sample[name] = float(rng.uniform(lo, hi))
    return sample


def encode_features(intervention: dict[str, float], patient: dict[str, float]) -> np.ndarray:
    """Encode intervention+patient into one numeric feature vector.

    The ordering is:
      [INTERVENTION_NAMES..., PATIENT_NAMES...]

    This deterministic ordering is used consistently across fit/predict steps.
    """
    ints = [float(intervention[k]) for k in INTERVENTION_NAMES]
    pats = [float(patient[k]) for k in PATIENT_NAMES]
    return np.array(ints + pats, dtype=float)


def make_objective(patient: dict[str, float], metric: str = "combined") -> Callable[[dict[str, float]], dict[str, float]]:
    """Create objective function consistent with existing EA/analytics semantics.

    Parameters
    ----------
    patient:
        Patient profile dictionary.
    metric:
        One of {"combined", "atp", "het", "crisis_delay"}.

    Returns
    -------
    callable
        Function `objective(intervention) -> metrics_dict` where
        `metrics_dict["fitness"]` is the scalar objective.
    """
    baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
    base_atp = float(baseline["states"][-1, 2])
    base_het = float(baseline["heteroplasmy"][-1])

    def objective(intervention: dict[str, float]) -> dict[str, float]:
        iv = clip_intervention(intervention)
        result = simulate(intervention=iv, patient=patient)
        analytics = compute_all(result, baseline)

        final_atp = float(result["states"][-1, 2])
        final_het = float(result["heteroplasmy"][-1])
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
        }

    return objective


@dataclass
class KNNRegressorSurrogate:
    """Distance-weighted KNN regressor using only numpy.

    This implementation is intentionally lightweight (no sklearn dependency)
    and deterministic given fixed training data.
    """

    k: int = 11
    distance_eps: float = 1e-9

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNRegressorSurrogate":
        """Fit surrogate on tabular features and scalar targets."""
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if len(x) != len(y):
            raise ValueError("x and y length mismatch")
        if len(x) == 0:
            raise ValueError("empty training set")
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict scalar targets for one or many query vectors."""
        if not hasattr(self, "_x"):
            raise RuntimeError("model is not fitted")
        xq = np.asarray(x, dtype=float)
        if xq.ndim == 1:
            xq = xq[None, :]

        pred = np.zeros(len(xq), dtype=float)
        n_train = len(self._x)
        k = max(1, min(self.k, n_train))

        for i, q in enumerate(xq):
            d = np.linalg.norm(self._x - q[None, :], axis=1)
            idx = np.argpartition(d, kth=k - 1)[:k]
            dk = d[idx]
            w = 1.0 / (dk + self.distance_eps)
            pred[i] = float(np.sum(w * self._y[idx]) / np.sum(w))
        return pred


def build_training_data(
    patient: dict[str, float],
    n_samples: int,
    seed: int,
    objective_fn: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> dict:
    """Generate a supervised dataset by simulator evaluations.

    Returns a dictionary with:
      - `x`: feature matrix
      - `y`: scalar objective labels
      - `interventions`: raw intervention dicts used for each row
    """
    rng = np.random.default_rng(seed)
    objective = objective_fn or make_objective(patient=patient, metric="combined")

    x_rows = []
    y_rows = []
    interventions = []

    for _ in range(n_samples):
        iv = random_intervention(rng)
        metrics = objective(iv)
        x_rows.append(encode_features(iv, patient))
        y_rows.append(float(metrics["fitness"]))
        interventions.append(clip_intervention(iv))

    return {
        "x": np.array(x_rows, dtype=float),
        "y": np.array(y_rows, dtype=float),
        "interventions": interventions,
    }


def evaluate_candidates(
    candidates: list[dict[str, float]],
    objective_fn: Callable[[dict[str, float]], dict[str, float]],
) -> list[dict]:
    """Evaluate candidate interventions with the true simulator objective.

    This is the critical "ground-truth recheck" step that prevents surrogate
    prediction from replacing mechanistic simulation outputs.
    """
    rows = []
    for iv in candidates:
        metrics = objective_fn(iv)
        rows.append({
            "params": clip_intervention(iv),
            "fitness": float(metrics["fitness"]),
            "final_atp": float(metrics.get("final_atp", np.nan)),
            "final_het": float(metrics.get("final_het", np.nan)),
            "atp_benefit": float(metrics.get("atp_benefit", np.nan)),
            "het_benefit": float(metrics.get("het_benefit", np.nan)),
        })
    return rows
