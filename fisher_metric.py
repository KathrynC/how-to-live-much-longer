#!/usr/bin/env python3
"""
fisher_metric.py

Measure LLM output variance to quantify clinical certainty.

Adapted from fisher_metric.py in the parent Evolutionary-Robotics project,
which queried Ollama 10× per seed to measure weight output variance and
build a statistical manifold. Here we measure how consistently each LLM
generates the same 12D intervention+patient vector for the same clinical
scenario.

High variance = genuine clinical ambiguity (the LLM is uncertain).
Low variance = strong opinion (the LLM has a clear protocol recommendation).

Experiments:
  - 10 clinical scenarios × 10 repeats × 4 models = 400 LLM queries
  - Compute 12×12 covariance matrix per (scenario, model)
  - Fisher information = precision of the LLM's "clinical intuition"
  - Identify high/low certainty scenarios

Scale: 400 Ollama queries
Estimated time: ~30-60 minutes (depends on model speed)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    CLINICAL_SEEDS, ALL_PARAM_NAMES,
    DEFAULT_INTERVENTION,
)
from analytics import NumpyEncoder
from llm_common import MODELS, query_ollama


# ── Configuration ───────────────────────────────────────────────────────────

N_REPEATS = 10  # queries per (scenario, model) pair

PROMPT_TEMPLATE = (
    "You are a mitochondrial medicine specialist designing a personalized "
    "intervention protocol.\n\n"
    "CLINICAL SCENARIO:\n{scenario}\n\n"
    "PARAMETERS (output a JSON object with ALL 12 keys):\n"
    "  Intervention (0.0-1.0 each):\n"
    "    rapamycin_dose, nad_supplement, senolytic_dose,\n"
    "    yamanaka_intensity, transplant_rate, exercise_level\n"
    "  Patient:\n"
    "    baseline_age (20-90), baseline_heteroplasmy (0.0-0.95),\n"
    "    baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0),\n"
    "    metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)\n\n"
    "Choose values thoughtfully based on the clinical scenario.\n"
    "Output ONLY the JSON object. Keep reasoning SHORT (1-2 sentences max)."
)


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(seeds=None, n_repeats=N_REPEATS):
    if seeds is None:
        seeds = CLINICAL_SEEDS

    out_path = PROJECT / "artifacts" / "fisher_metric.json"
    checkpoint_path = PROJECT / "artifacts" / "fisher_metric_checkpoint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    completed = {}
    raw_results = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        raw_results = ckpt.get("raw_results", [])
        for r in raw_results:
            key = f"{r['seed_id']}|{r['model']}|{r['repeat']}"
            completed[key] = True
        print(f"Resumed from checkpoint: {len(raw_results)} queries done")

    n_total = len(seeds) * len(MODELS) * n_repeats
    n_remaining = n_total - len(completed)
    print(f"{'=' * 70}")
    print(f"FISHER METRIC EXPERIMENT — Clinical Certainty")
    print(f"{'=' * 70}")
    print(f"Scenarios: {len(seeds)}, Models: {len(MODELS)}, Repeats: {n_repeats}")
    print(f"Total queries: {n_total}, Remaining: {n_remaining}")
    print()

    start_time = time.time()
    query_num = len(completed)
    failures = 0

    for seed in seeds:
        seed_id = seed["id"]
        scenario = seed["description"]
        prompt = PROMPT_TEMPLATE.format(scenario=scenario)

        for model_info in MODELS:
            model_name = model_info["name"]

            for rep in range(n_repeats):
                key = f"{seed_id}|{model_name}|{rep}"
                if key in completed:
                    continue

                query_num += 1
                print(f"[{query_num}/{n_total}] {seed_id} | {model_name} | rep {rep+1}/{n_repeats}",
                      end=" ", flush=True)

                vector, raw_resp = query_ollama(model_name, prompt,
                                                temperature=0.8)

                if vector is None:
                    failures += 1
                    print("-> FAIL")
                    raw_results.append({
                        "seed_id": seed_id, "model": model_name,
                        "repeat": rep, "success": False, "vector": None,
                    })
                else:
                    # Convert to flat 12D array
                    vec_flat = [vector.get(k, 0.0) for k in ALL_PARAM_NAMES]
                    print(f"-> [{', '.join(f'{v:.2f}' for v in vec_flat[:6])}...]")
                    raw_results.append({
                        "seed_id": seed_id, "model": model_name,
                        "repeat": rep, "success": True,
                        "vector": vector, "vector_flat": vec_flat,
                    })

                completed[key] = True

                # Checkpoint every 40 queries
                if len(completed) % 40 == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump({"raw_results": raw_results},
                                  f, indent=2, cls=NumpyEncoder)

    total_elapsed = time.time() - start_time

    # ── Analysis ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FISHER METRIC ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Total time: {total_elapsed:.1f}s, Failures: {failures}")

    analyses = []

    for seed in seeds:
        seed_id = seed["id"]
        for model_info in MODELS:
            model_name = model_info["name"]

            # Collect successful vectors for this (scenario, model)
            vectors = []
            for r in raw_results:
                if (r["seed_id"] == seed_id and r["model"] == model_name
                        and r["success"] and r.get("vector_flat")):
                    vectors.append(r["vector_flat"])

            if len(vectors) < 3:
                continue

            vecs = np.array(vectors)  # shape: (n_repeats, 12)
            mean_vec = np.mean(vecs, axis=0)
            cov_matrix = np.cov(vecs, rowvar=False)  # 12×12

            # Total variance (trace of covariance)
            total_variance = float(np.trace(cov_matrix))

            # Per-parameter variance
            param_variances = {ALL_PARAM_NAMES[i]: float(cov_matrix[i, i])
                               for i in range(len(ALL_PARAM_NAMES))
                               if i < cov_matrix.shape[0]}

            # Most and least certain parameters
            sorted_vars = sorted(param_variances.items(), key=lambda x: x[1])
            most_certain = sorted_vars[0][0]
            least_certain = sorted_vars[-1][0]

            # Fisher information (inverse covariance, regularized)
            try:
                reg_cov = cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0])
                fisher = np.linalg.inv(reg_cov)
                fisher_trace = float(np.trace(fisher))
            except np.linalg.LinAlgError:
                fisher_trace = 0.0

            # Effective dimensionality (how many params actually vary)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                probs = eigenvalues / eigenvalues.sum()
                eff_dim = float(np.exp(-np.sum(probs * np.log(probs + 1e-15))))
            else:
                eff_dim = 0.0

            analyses.append({
                "seed_id": seed_id,
                "model": model_name,
                "n_samples": len(vectors),
                "mean_vector": {ALL_PARAM_NAMES[i]: float(mean_vec[i])
                                for i in range(len(ALL_PARAM_NAMES))
                                if i < len(mean_vec)},
                "total_variance": total_variance,
                "param_variances": param_variances,
                "most_certain_param": most_certain,
                "least_certain_param": least_certain,
                "fisher_trace": fisher_trace,
                "effective_dimensionality": eff_dim,
                "covariance_matrix": cov_matrix.tolist(),
            })

    # Summary
    if analyses:
        print(f"\n{'Scenario':30s} {'Model':20s} {'TotalVar':>10s} {'EffDim':>8s} {'Most certain':>20s}")
        print("-" * 92)
        for a in sorted(analyses, key=lambda x: x["total_variance"]):
            print(f"{a['seed_id']:30s} {a['model']:20s} "
                  f"{a['total_variance']:10.4f} {a['effective_dimensionality']:8.2f} "
                  f"{a['most_certain_param']:>20s}")

        # Most/least certain scenarios (averaged across models)
        from collections import defaultdict
        scenario_vars = defaultdict(list)
        for a in analyses:
            scenario_vars[a["seed_id"]].append(a["total_variance"])

        print(f"\nScenarios ranked by mean variance (lower = more clinical certainty):")
        sorted_scenarios = sorted(scenario_vars.items(),
                                  key=lambda x: np.mean(x[1]))
        for sid, vars_list in sorted_scenarios:
            print(f"  {sid:30s}: mean_var={np.mean(vars_list):.4f} "
                  f"(across {len(vars_list)} models)")

        # Model certainty comparison
        model_vars = defaultdict(list)
        for a in analyses:
            model_vars[a["model"]].append(a["total_variance"])

        print(f"\nModels ranked by mean variance:")
        sorted_models = sorted(model_vars.items(),
                                key=lambda x: np.mean(x[1]))
        for mname, vars_list in sorted_models:
            print(f"  {mname:20s}: mean_var={np.mean(vars_list):.4f}")

    # Save
    output = {
        "experiment": "fisher_metric",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": total_elapsed,
        "n_queries": len(raw_results),
        "n_failures": failures,
        "n_repeats": n_repeats,
        "analyses": analyses,
        "raw_results": raw_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    run_experiment()
