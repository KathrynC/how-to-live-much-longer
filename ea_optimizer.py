"""EA-powered optimizer for mitochondrial intervention protocols.

Wraps the mitochondrial aging simulator as an ea-toolkit FitnessFunction and
provides access to 8 evolutionary algorithms for intervention optimization,
landscape analysis, and head-to-head algorithm comparison.

Algorithms available:
    hill_climber    — Greedy local search with restarts
    es              — (1+lambda) Evolution Strategy with adaptive sigma
    de              — Differential Evolution (population-based)
    cma_es          — CMA-ES (covariance-adaptive, state of the art)
    ridge_walker    — Multi-objective Pareto exploration
    cliff_mapper    — Seeks high-sensitivity cliff regions
    novelty_seeker  — Behavioral diversity without fitness pressure
    ensemble        — Parallel hill climbers with teleportation

Usage:
    python ea_optimizer.py                             # CMA-ES, budget 500
    python ea_optimizer.py --algo de --budget 1000     # Differential Evolution
    python ea_optimizer.py --compare --budget 300      # Head-to-head comparison
    python ea_optimizer.py --landscape --budget 200    # Landscape analysis
    python ea_optimizer.py --patient near_cliff_80     # Different patient profile

Requires:
    ea-toolkit (at ~/ea-toolkit)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
EA_TOOLKIT_PATH = PROJECT.parent / "ea-toolkit"
if str(EA_TOOLKIT_PATH) not in sys.path:
    sys.path.insert(0, str(EA_TOOLKIT_PATH))

# Project imports
from constants import (
    INTERVENTION_PARAMS, INTERVENTION_NAMES,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    CLINICAL_SEEDS,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder

# EA toolkit imports
from ea_toolkit import (
    HillClimber, OnePlusLambdaES, DifferentialEvolution, CMAES,
    RidgeWalker, CliffMapper, NoveltySeeker, EnsembleExplorer,
    GaussianMutation, AdaptiveMutation, CauchyMutation,
    ProgressPrinter, ConvergenceChecker,
)
from ea_toolkit.base import FitnessFunction
from ea_toolkit.landscape import LandscapeAnalyzer, probe_cliffiness

# ── Configuration ────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path("artifacts")

# Patient profiles for optimization
PATIENT_PROFILES = {
    "default": DEFAULT_PATIENT,
    "near_cliff_80": {
        "baseline_age": 80.0, "baseline_heteroplasmy": 0.65,
        "baseline_nad_level": 0.4, "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0, "inflammation_level": 0.5,
    },
    "young_prevention_25": {
        "baseline_age": 25.0, "baseline_heteroplasmy": 0.05,
        "baseline_nad_level": 0.95, "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0, "inflammation_level": 0.0,
    },
    "melas_35": {
        "baseline_age": 35.0, "baseline_heteroplasmy": 0.50,
        "baseline_nad_level": 0.6, "genetic_vulnerability": 2.0,
        "metabolic_demand": 1.0, "inflammation_level": 0.25,
    },
    "sarcopenia_75": {
        "baseline_age": 75.0, "baseline_heteroplasmy": 0.45,
        "baseline_nad_level": 0.4, "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.5, "inflammation_level": 0.5,
    },
    "post_chemo_55": {
        "baseline_age": 55.0, "baseline_heteroplasmy": 0.40,
        "baseline_nad_level": 0.4, "genetic_vulnerability": 1.25,
        "metabolic_demand": 1.0, "inflammation_level": 0.5,
    },
}

# Available algorithms with constructors
ALGO_REGISTRY = {
    "hill_climber": "HillClimber",
    "es": "OnePlusLambdaES",
    "de": "DifferentialEvolution",
    "cma_es": "CMAES",
    "ridge_walker": "RidgeWalker",
    "cliff_mapper": "CliffMapper",
    "novelty_seeker": "NoveltySeeker",
    "ensemble": "EnsembleExplorer",
}

FITNESS_METRICS = {
    "combined": "ATP benefit + 0.5 * heteroplasmy benefit (default)",
    "atp": "Terminal ATP improvement over baseline",
    "het": "Heteroplasmy reduction from baseline",
    "crisis_delay": "Years of ATP crisis delay",
}


# ── Fitness Function Adapter ─────────────────────────────────────────────────


class MitoFitness(FitnessFunction):
    """Wraps the mitochondrial simulator as an ea-toolkit FitnessFunction.

    Evaluates intervention protocols by simulating 30-year trajectories and
    computing fitness as ATP benefit + weighted heteroplasmy benefit over a
    no-treatment baseline.

    Args:
        patient: Patient parameter dict (defaults to DEFAULT_PATIENT).
        metric: Fitness metric name ("combined", "atp", "het", "crisis_delay").
    """

    def __init__(self, patient: dict | None = None,
                 metric: str = "combined"):
        self.patient = patient or dict(DEFAULT_PATIENT)
        self.metric = metric
        # Cache baseline simulation (no treatment)
        self._baseline = simulate(
            intervention=DEFAULT_INTERVENTION, patient=self.patient)
        self._baseline_analytics = compute_all(self._baseline)
        self._eval_count = 0

    def evaluate(self, params: dict) -> dict:
        """Evaluate an intervention protocol.

        Args:
            params: Dict mapping intervention parameter names to float values.

        Returns:
            Dict with 'fitness' key plus analytics summary.
        """
        intervention = {k: float(params.get(k, 0.0)) for k in INTERVENTION_NAMES}
        result = simulate(intervention=intervention, patient=self.patient)
        analytics = compute_all(result, self._baseline)
        self._eval_count += 1

        # Extract key metrics
        final_atp = float(result["states"][-1, 2])
        base_atp = float(self._baseline["states"][-1, 2])
        final_het = float(result["heteroplasmy"][-1])
        base_het = float(self._baseline["heteroplasmy"][-1])
        atp_benefit = final_atp - base_atp
        het_benefit = base_het - final_het

        # Compute selected fitness metric
        if self.metric == "atp":
            fitness = atp_benefit
        elif self.metric == "het":
            fitness = het_benefit
        elif self.metric == "crisis_delay":
            fitness = float(analytics.get("intervention", {}).get(
                "crisis_delay_years", 0.0))
        else:  # combined
            fitness = atp_benefit + 0.5 * het_benefit

        return {
            "fitness": fitness,
            "atp_benefit": atp_benefit,
            "het_benefit": het_benefit,
            "final_atp": final_atp,
            "final_het": final_het,
            "base_atp": base_atp,
            "base_het": base_het,
            # Behavioral descriptors (for novelty seeker)
            "behavior": [final_atp, final_het],
        }

    def param_spec(self) -> dict:
        """Return intervention parameter bounds."""
        return {name: spec["range"]
                for name, spec in INTERVENTION_PARAMS.items()}


# ── Algorithm Factory ────────────────────────────────────────────────────────


def make_algorithm(name: str, fitness_fn: FitnessFunction,
                   seed: int = 42, verbose: bool = True):
    """Create an algorithm instance by name.

    Args:
        name: Algorithm name (key in ALGO_REGISTRY).
        fitness_fn: FitnessFunction to optimize.
        seed: Random seed.
        verbose: Whether to attach a progress printer.

    Returns:
        Configured Algorithm instance.
    """
    mutation = AdaptiveMutation(sigma_init=0.15, sigma_min=0.02, sigma_max=0.5)
    callbacks = []
    if verbose:
        callbacks.append(ProgressPrinter())
        callbacks.append(ConvergenceChecker(patience=50, min_delta=1e-5))

    if name == "hill_climber":
        algo = HillClimber(fitness_fn, mutation=mutation, seed=seed)
    elif name == "es":
        algo = OnePlusLambdaES(fitness_fn, mutation=mutation, seed=seed,
                               lam=7)
    elif name == "de":
        algo = DifferentialEvolution(fitness_fn, seed=seed,
                                     pop_size=20, F=0.7, CR=0.9)
    elif name == "cma_es":
        algo = CMAES(fitness_fn, seed=seed, pop_size=14, sigma0=0.3)
    elif name == "ridge_walker":
        algo = RidgeWalker(fitness_fn, mutation=mutation, seed=seed)
    elif name == "cliff_mapper":
        algo = CliffMapper(fitness_fn, mutation=mutation, seed=seed)
    elif name == "novelty_seeker":
        algo = NoveltySeeker(fitness_fn, mutation=mutation, seed=seed, k=5)
    elif name == "ensemble":
        algo = EnsembleExplorer(fitness_fn, mutation=mutation, seed=seed,
                                n_walkers=5)
    else:
        raise ValueError(f"Unknown algorithm: {name}. "
                         f"Choose from: {list(ALGO_REGISTRY.keys())}")

    algo.callbacks.extend(callbacks)
    return algo


# ── Run Modes ────────────────────────────────────────────────────────────────


def run_single(algo_name: str, budget: int, patient_name: str,
               metric: str, seed: int) -> dict:
    """Run a single algorithm and report results."""
    patient = PATIENT_PROFILES.get(patient_name, DEFAULT_PATIENT)
    fitness_fn = MitoFitness(patient=patient, metric=metric)
    print(f"Patient: {patient_name}")
    print(f"Metric: {metric} ({FITNESS_METRICS.get(metric, '')})")
    print(f"Algorithm: {algo_name} ({ALGO_REGISTRY[algo_name]})")
    print(f"Budget: {budget} evaluations")
    base_atp = fitness_fn._baseline_analytics.get("energy", {}).get("atp_terminal")
    base_het = float(fitness_fn._baseline["heteroplasmy"][-1])
    print(f"Baseline ATP: {base_atp:.4f}" if base_atp else "Baseline ATP: N/A")
    print(f"Baseline het: {base_het:.4f}")
    print()

    t0 = time.time()
    algo = make_algorithm(algo_name, fitness_fn, seed=seed)
    history = algo.run(budget=budget)
    elapsed = time.time() - t0

    best = algo.best()
    if best is None:
        print("No evaluations completed.")
        return {}

    print(f"\n{'='*60}")
    print(f"Best fitness: {best['fitness']:.6f}")
    print(f"Best protocol:")
    for k in INTERVENTION_NAMES:
        print(f"  {k}: {best['params'][k]:.3f}")
    print(f"  → ATP: {best.get('final_atp', '?'):.4f} "
          f"(+{best.get('atp_benefit', '?'):.4f})")
    print(f"  → Het: {best.get('final_het', '?'):.4f} "
          f"({best.get('het_benefit', '?'):+.4f})")
    print(f"Evaluations: {len(history)}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/len(history)*1000:.0f}ms/eval)")

    result = {
        "algorithm": algo_name,
        "budget": budget,
        "patient": patient_name,
        "metric": metric,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "n_evaluations": len(history),
        "best_fitness": best["fitness"],
        "best_params": best["params"],
        "best_atp": best.get("final_atp"),
        "best_het": best.get("final_het"),
        "atp_benefit": best.get("atp_benefit"),
        "het_benefit": best.get("het_benefit"),
        "fitness_trajectory": [h["fitness"] for h in history],
    }
    return result


def run_comparison(budget: int, patient_name: str, metric: str,
                   seed: int) -> dict:
    """Head-to-head comparison of all algorithms."""
    patient = PATIENT_PROFILES.get(patient_name, DEFAULT_PATIENT)
    algos_to_compare = ["hill_climber", "es", "de", "cma_es", "ensemble"]

    print(f"Algorithm Comparison — {len(algos_to_compare)} algorithms")
    print(f"Patient: {patient_name}")
    print(f"Metric: {metric}")
    print(f"Budget: {budget} per algorithm")
    print(f"Total: ~{budget * len(algos_to_compare)} evaluations")
    print()

    results = {}
    for algo_name in algos_to_compare:
        print(f"\n{'─'*60}")
        print(f"Running {algo_name} ({ALGO_REGISTRY[algo_name]})")
        print(f"{'─'*60}")

        fitness_fn = MitoFitness(patient=patient, metric=metric)
        t0 = time.time()
        algo = make_algorithm(algo_name, fitness_fn, seed=seed, verbose=False)
        history = algo.run(budget=budget)
        elapsed = time.time() - t0

        best = algo.best()
        if best:
            results[algo_name] = {
                "best_fitness": best["fitness"],
                "best_params": best["params"],
                "best_atp": best.get("final_atp"),
                "best_het": best.get("final_het"),
                "n_evaluations": len(history),
                "elapsed_seconds": round(elapsed, 1),
                "fitness_trajectory": [h["fitness"] for h in history],
            }
            print(f"  Best fitness: {best['fitness']:.6f} "
                  f"(ATP: {best.get('final_atp', 0):.4f}, "
                  f"Het: {best.get('final_het', 0):.4f}) "
                  f"[{elapsed:.1f}s]")

    # Rankings
    print(f"\n{'='*60}")
    print("RANKINGS")
    print(f"{'='*60}")
    ranked = sorted(results.items(), key=lambda x: x[1]["best_fitness"],
                    reverse=True)
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:20s} fitness={r['best_fitness']:.6f} "
              f"ATP={r['best_atp']:.4f} het={r['best_het']:.4f}")

    return {
        "experiment": "ea_comparison",
        "patient": patient_name,
        "metric": metric,
        "budget_per_algo": budget,
        "seed": seed,
        "results": results,
        "ranking": [name for name, _ in ranked],
    }


def run_landscape(budget: int, patient_name: str, metric: str,
                  seed: int) -> dict:
    """Landscape analysis of the intervention space."""
    patient = PATIENT_PROFILES.get(patient_name, DEFAULT_PATIENT)
    fitness_fn = MitoFitness(patient=patient, metric=metric)

    print(f"Landscape Analysis — budget {budget}")
    print(f"Patient: {patient_name}")
    print(f"Metric: {metric}")
    print()

    t0 = time.time()
    analyzer = LandscapeAnalyzer(fitness_fn, seed=seed)
    report = analyzer.run_analysis(budget=budget)
    elapsed = time.time() - t0

    print(f"\nLandscape Report ({elapsed:.1f}s):")
    print(f"  Samples: {report.get('n_samples', '?')}")
    print(f"  Fitness range: [{report.get('fitness_min', '?'):.4f}, "
          f"{report.get('fitness_max', '?'):.4f}]")
    print(f"  Mean fitness: {report.get('fitness_mean', '?'):.4f}")
    print(f"  Fitness std: {report.get('fitness_std', '?'):.4f}")
    print(f"  Mean cliffiness: {report.get('mean_cliffiness', '?'):.4f}")
    print(f"  Mean gradient magnitude: "
          f"{report.get('mean_gradient_magnitude', '?'):.4f}")

    roughness = report.get("roughness")
    if roughness is not None:
        print(f"  Roughness: {roughness:.4f}")
    sign_flip = report.get("gradient_sign_flip_rate")
    if sign_flip is not None:
        print(f"  Gradient sign-flip rate: {sign_flip:.4f}")

    result = {
        "experiment": "ea_landscape",
        "patient": patient_name,
        "metric": metric,
        "budget": budget,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        **report,
    }
    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EA-powered mitochondrial intervention optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available algorithms: {', '.join(ALGO_REGISTRY.keys())}
Available patients: {', '.join(PATIENT_PROFILES.keys())}
Available metrics: {', '.join(FITNESS_METRICS.keys())}
""",
    )
    parser.add_argument("--algo", type=str, default="cma_es",
                        help="Algorithm name (default: cma_es)")
    parser.add_argument("--budget", type=int, default=500,
                        help="Evaluation budget (default: 500)")
    parser.add_argument("--patient", type=str, default="default",
                        help="Patient profile name (default: default)")
    parser.add_argument("--metric", type=str, default="combined",
                        help="Fitness metric (default: combined)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--compare", action="store_true",
                        help="Run head-to-head algorithm comparison")
    parser.add_argument("--landscape", action="store_true",
                        help="Run landscape analysis")
    args = parser.parse_args()

    print(f"EA Optimizer for Mitochondrial Aging Simulator")
    print(f"ea-toolkit: {EA_TOOLKIT_PATH}")
    print()

    if args.compare:
        result = run_comparison(args.budget, args.patient, args.metric,
                                args.seed)
        out_file = "ea_comparison"
    elif args.landscape:
        result = run_landscape(args.budget, args.patient, args.metric,
                               args.seed)
        out_file = "ea_landscape"
    else:
        result = run_single(args.algo, args.budget, args.patient,
                            args.metric, args.seed)
        out_file = f"ea_{args.algo}"

    # Save artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{out_file}.json"
    out_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
