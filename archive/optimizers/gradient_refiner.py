"""Gradient-based local refinement for mitochondrial intervention protocols.

This module is designed as a post-EA local optimizer:
  1. Start from a seed protocol (e.g., EA best_params from an artifact).
  2. Estimate gradients by central finite differences.
  3. Run bounded Adam ascent on the selected fitness metric.

The default objective is intentionally aligned with `ea_optimizer.MitoFitness`:
combined fitness = ATP benefit + 0.5 * heteroplasmy benefit.

Usage examples:
    python gradient_refiner.py
    python gradient_refiner.py --seed-artifact artifacts/ea_cma_es.json
    python gradient_refiner.py --seed-artifact artifacts/ea_protocol_transfer_2026-02-17.json --profile near_cliff_80 --patient near_cliff_80
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics import NumpyEncoder, compute_all
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT, INTERVENTION_NAMES, INTERVENTION_PARAMS
from simulator import simulate


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"


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

METRICS = {"combined", "atp", "het", "crisis_delay"}


def _bounds_dict() -> dict[str, tuple[float, float]]:
    """Return intervention bounds keyed by parameter name."""
    return {name: tuple(spec["range"]) for name, spec in INTERVENTION_PARAMS.items()}


def _clip_protocol(protocol: dict[str, float]) -> dict[str, float]:
    """Clamp a protocol into valid intervention bounds."""
    clipped = {}
    bounds = _bounds_dict()
    for name in INTERVENTION_NAMES:
        lo, hi = bounds[name]
        clipped[name] = float(np.clip(float(protocol.get(name, 0.0)), lo, hi))
    return clipped


def _vector_from_protocol(protocol: dict[str, float]) -> np.ndarray:
    """Convert parameter dict to dense vector with canonical ordering."""
    return np.array([float(protocol[k]) for k in INTERVENTION_NAMES], dtype=float)


def _protocol_from_vector(vec: np.ndarray) -> dict[str, float]:
    """Convert dense vector back to parameter dict."""
    return {k: float(v) for k, v in zip(INTERVENTION_NAMES, vec)}


def _extract_seed_protocol(data: dict, profile: str | None = None) -> dict[str, float]:
    """Extract a seed intervention dict from known EA artifact schemas.

    Supported layouts:
      - `{best_params: {...}}`
      - `{best_runs: {profile: {best_params: {...}}}}`
      - `{results: {... best_params ...}}`
    """
    if "best_params" in data and isinstance(data["best_params"], dict):
        return _clip_protocol(data["best_params"])

    if "best_runs" in data and isinstance(data["best_runs"], dict):
        key = profile or next(iter(data["best_runs"]), None)
        if key and "best_params" in data["best_runs"].get(key, {}):
            return _clip_protocol(data["best_runs"][key]["best_params"])

    if "results" in data and isinstance(data["results"], dict):
        # ea_comparison-style: choose highest-fitness algorithm unless profile specified.
        has_algo_rows = all(isinstance(v, dict) and "best_params" in v for v in data["results"].values())
        if has_algo_rows:
            best_key = max(
                data["results"].keys(),
                key=lambda k: float(data["results"][k].get("best_fitness", -np.inf)),
            )
            return _clip_protocol(data["results"][best_key]["best_params"])

        # constrained profile rows: select named profile or first row.
        key = profile or next(iter(data["results"]), None)
        if key and "best_params" in data["results"].get(key, {}):
            return _clip_protocol(data["results"][key]["best_params"])

    raise ValueError(
        "Could not extract seed protocol from artifact. "
        "Expected one of: best_params, best_runs[profile].best_params, "
        "or results[...].best_params."
    )


def load_seed_protocol(path: str, profile: str | None = None) -> dict[str, float]:
    """Load and normalize a seed protocol from JSON artifact.

    Parameters
    ----------
    path:
        Path to JSON artifact.
    profile:
        Optional profile key for multi-profile artifacts.
    """
    data = json.loads(Path(path).read_text())
    return _extract_seed_protocol(data, profile=profile)


def make_objective(patient: dict[str, float], metric: str = "combined"):
    """Create objective(protocol) -> metrics dict, with cached no-treatment baseline.

    This preserves metric semantics used elsewhere in the project so gradient
    refinement is comparable with EA runs.
    """
    baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
    baseline_analytics = compute_all(baseline)
    base_atp = float(baseline["states"][-1, 2])
    base_het = float(baseline["heteroplasmy"][-1])

    def objective(protocol: dict[str, float]) -> dict[str, float]:
        intervention = _clip_protocol(protocol)
        result = simulate(intervention=intervention, patient=patient)
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
            "baseline_atp": base_atp,
            "baseline_het": base_het,
            "baseline_atp_terminal_analytics": float(
                baseline_analytics.get("energy", {}).get("atp_terminal", base_atp)
            ),
        }

    return objective


def finite_difference_gradient(
    x: np.ndarray,
    objective_fn,
    rel_step: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Central finite-difference gradient for scalar fitness.

    Returns
    -------
    (grad, grad_norm)
        Gradient vector and its L2 norm.
    """
    bounds = _bounds_dict()
    grad = np.zeros_like(x)

    for i, name in enumerate(INTERVENTION_NAMES):
        lo, hi = bounds[name]
        span = hi - lo
        h = max(rel_step * span, 1e-6)

        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] = float(np.clip(x_plus[i] + h, lo, hi))
        x_minus[i] = float(np.clip(x_minus[i] - h, lo, hi))

        f_plus = float(objective_fn(_protocol_from_vector(x_plus))["fitness"])
        f_minus = float(objective_fn(_protocol_from_vector(x_minus))["fitness"])
        denom = max(x_plus[i] - x_minus[i], 1e-12)
        grad[i] = (f_plus - f_minus) / denom

    return grad, float(np.linalg.norm(grad))


def refine_protocol(
    seed_protocol: dict[str, float],
    objective_fn,
    steps: int = 40,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
    fd_rel_step: float = 1e-3,
    patience: int = 12,
) -> dict:
    """Run bounded Adam ascent from a seed protocol.

    Returns a dict containing seed/best protocol summaries plus optimization
    history for downstream audit and visualization.
    """
    x = _vector_from_protocol(_clip_protocol(seed_protocol))
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    bounds = _bounds_dict()

    current = objective_fn(_protocol_from_vector(x))
    best = {"params": _protocol_from_vector(x), **current, "step": 0}
    history = [{
        "step": 0,
        "fitness": current["fitness"],
        "grad_norm": 0.0,
        "params": _protocol_from_vector(x),
    }]

    no_improve = 0

    for t in range(1, steps + 1):
        grad, grad_norm = finite_difference_gradient(x, objective_fn, rel_step=fd_rel_step)

        # Adam ascent update for maximization.
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        x = x + lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        # Projection onto biologically allowed parameter bounds.
        for i, name in enumerate(INTERVENTION_NAMES):
            lo, hi = bounds[name]
            x[i] = float(np.clip(x[i], lo, hi))

        metrics = objective_fn(_protocol_from_vector(x))
        history.append({
            "step": t,
            "fitness": metrics["fitness"],
            "grad_norm": grad_norm,
            "params": _protocol_from_vector(x),
        })

        if metrics["fitness"] > best["fitness"] + 1e-12:
            best = {"params": _protocol_from_vector(x), **metrics, "step": t}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return {
        "seed_params": _clip_protocol(seed_protocol),
        "seed_fitness": history[0]["fitness"],
        "best_params": best["params"],
        "best_fitness": best["fitness"],
        "improvement": best["fitness"] - history[0]["fitness"],
        "best_step": best["step"],
        "history": history,
        "best_metrics": {
            "final_atp": best.get("final_atp"),
            "final_het": best.get("final_het"),
            "atp_benefit": best.get("atp_benefit"),
            "het_benefit": best.get("het_benefit"),
        },
    }


def _default_output_path(patient: str) -> Path:
    """Default artifact destination for one refine run."""
    stamp = time.strftime("%Y-%m-%d")
    return ARTIFACTS_DIR / f"gd_refiner_{patient}_{stamp}.json"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gradient-based local refiner for intervention protocols.",
    )
    parser.add_argument("--seed-artifact", type=str, default=None, help="JSON artifact path with best_params.")
    parser.add_argument("--profile", type=str, default=None, help="Profile key when artifact contains multi-profile results.")
    parser.add_argument("--patient", type=str, default="default", help=f"Patient profile name ({', '.join(PATIENT_PROFILES.keys())}).")
    parser.add_argument("--metric", type=str, default="combined", help=f"Fitness metric ({', '.join(sorted(METRICS))}).")
    parser.add_argument("--steps", type=int, default=40, help="Max optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--fd-rel-step", type=float, default=1e-3, help="Relative finite-difference step as fraction of parameter range.")
    parser.add_argument("--patience", type=int, default=12, help="Early-stop patience in no-improvement steps.")
    parser.add_argument("--seed", type=int, default=2026, help="Reserved for reproducible future stochastic extensions.")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: artifacts/gd_refiner_<patient>_<date>.json).")
    args = parser.parse_args()

    if args.metric not in METRICS:
        raise ValueError(f"Unknown metric '{args.metric}'. Choose from {sorted(METRICS)}.")
    if args.patient not in PATIENT_PROFILES:
        raise ValueError(f"Unknown patient '{args.patient}'. Choose from {sorted(PATIENT_PROFILES)}.")

    patient = PATIENT_PROFILES[args.patient]
    seed_protocol = (
        load_seed_protocol(args.seed_artifact, profile=args.profile)
        if args.seed_artifact
        else _clip_protocol(DEFAULT_INTERVENTION)
    )

    t0 = time.time()
    objective = make_objective(patient=patient, metric=args.metric)
    result = refine_protocol(
        seed_protocol=seed_protocol,
        objective_fn=objective,
        steps=args.steps,
        lr=args.lr,
        fd_rel_step=args.fd_rel_step,
        patience=args.patience,
    )
    elapsed = time.time() - t0

    output = {
        "experiment": "gradient_refiner",
        "patient": args.patient,
        "metric": args.metric,
        "seed_artifact": args.seed_artifact,
        "profile": args.profile,
        "steps": args.steps,
        "lr": args.lr,
        "fd_rel_step": args.fd_rel_step,
        "patience": args.patience,
        "elapsed_seconds": round(elapsed, 2),
        "seed": args.seed,
        **result,
    }

    out_path = Path(args.out) if args.out else _default_output_path(args.patient)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))

    print("Gradient Refiner")
    print(f"Patient: {args.patient}")
    print(f"Metric: {args.metric}")
    print(f"Seed fitness: {output['seed_fitness']:.6f}")
    print(f"Best fitness: {output['best_fitness']:.6f}")
    print(f"Improvement: {output['improvement']:+.6f}")
    print(f"Best step: {output['best_step']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
