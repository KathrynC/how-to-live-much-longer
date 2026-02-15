#!/usr/bin/env python3
"""
clinical_consensus.py

Multi-model agreement analysis for mitochondrial intervention design.

Queries all available local LLMs with the same clinical scenario and measures
how much they agree on the intervention protocol. Consensus indicates robust
clinical reasoning; disagreement flags genuinely ambiguous decisions.

Adapted from the multi-model comparison aspects of the parent project's
structured_random_compare.py.

Experiments:
  - 10 clinical scenarios × 4 models = 40 queries
  - Pairwise cosine similarity between model outputs
  - Per-parameter agreement (standard deviation across models)
  - Identify controversial vs consensus scenarios
  - Simulate all proposals, compare outcomes

Scale: 40 Ollama queries + 40 simulations
Estimated time: ~15-20 minutes
"""

import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    CLINICAL_SEEDS, ALL_PARAM_NAMES,
    INTERVENTION_NAMES, PATIENT_NAMES,
    DEFAULT_INTERVENTION,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder
from llm_common import MODELS, query_ollama, split_vector


# ── Prompt ──────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are a mitochondrial medicine specialist. Design an optimal "
    "intervention protocol for this patient.\n\n"
    "CLINICAL SCENARIO:\n{scenario}\n\n"
    "Output a JSON object with ALL 12 keys:\n"
    "  rapamycin_dose (0-1), nad_supplement (0-1), senolytic_dose (0-1),\n"
    "  yamanaka_intensity (0-1), transplant_rate (0-1), exercise_level (0-1),\n"
    "  baseline_age (20-90), baseline_heteroplasmy (0-0.95),\n"
    "  baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0),\n"
    "  metabolic_demand (0.5-2.0), inflammation_level (0-1.0)\n\n"
    "Think step by step about the patient's condition, then output the JSON. "
    "Keep reasoning to 2-3 sentences."
)


# ── Similarity metrics ─────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def param_agreement(vectors_dict, param_names):
    """Compute per-parameter standard deviation across models.

    Lower std = more agreement.
    """
    if len(vectors_dict) < 2:
        return {}
    arrays = []
    for v in vectors_dict.values():
        arrays.append([v.get(k, 0.0) for k in param_names])
    mat = np.array(arrays)
    stds = np.std(mat, axis=0)
    return {param_names[i]: float(stds[i]) for i in range(len(param_names))}


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(seeds=None):
    if seeds is None:
        seeds = CLINICAL_SEEDS

    out_path = PROJECT / "artifacts" / "clinical_consensus.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    query_num = 0
    n_total = len(seeds) * len(MODELS)

    print(f"{'=' * 70}")
    print(f"CLINICAL CONSENSUS EXPERIMENT — Multi-Model Agreement")
    print(f"{'=' * 70}")
    print(f"Scenarios: {len(seeds)}, Models: {len(MODELS)}")
    print(f"Total queries: {n_total}")
    print()

    scenario_results = []

    for seed in seeds:
        seed_id = seed["id"]
        scenario = seed["description"]
        prompt = PROMPT_TEMPLATE.format(scenario=scenario)

        print(f"\n--- {seed_id} ---")

        model_vectors = {}  # model_name -> snapped vector dict
        model_outcomes = {}  # model_name -> simulation analytics

        for model_info in MODELS:
            model_name = model_info["name"]
            query_num += 1
            print(f"  [{query_num}/{n_total}] {model_name}", end=" ", flush=True)

            vector, raw_resp = query_ollama(model_name, prompt, temperature=0.7)

            if vector is None:
                print("-> FAIL")
                continue

            model_vectors[model_name] = vector
            intervention, patient = split_vector(vector)

            # Simulate
            result = simulate(intervention=intervention, patient=patient)
            baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
            analytics = compute_all(result, baseline)
            model_outcomes[model_name] = analytics

            atp = analytics["energy"]["atp_final"]
            het = analytics["damage"]["het_final"]
            benefit = analytics["intervention"]["atp_benefit_terminal"]
            print(f"-> ATP={atp:.3f} het={het:.3f} benefit={benefit:+.3f}")

        if len(model_vectors) < 2:
            print(f"  Too few successful models for consensus analysis")
            continue

        # ── Consensus metrics ──────────────────────────────────────────────
        model_names = list(model_vectors.keys())

        # Pairwise cosine similarity
        flat_vectors = {}
        for mname, vec in model_vectors.items():
            flat_vectors[mname] = np.array([vec.get(k, 0.0) for k in ALL_PARAM_NAMES])

        pair_sims = []
        for m1, m2 in combinations(model_names, 2):
            sim = cosine_similarity(flat_vectors[m1], flat_vectors[m2])
            pair_sims.append({"models": [m1, m2], "cosine_similarity": sim})

        mean_cosine = float(np.mean([p["cosine_similarity"] for p in pair_sims]))

        # Per-parameter agreement
        intv_agreement = param_agreement(
            {m: model_vectors[m] for m in model_names}, INTERVENTION_NAMES)
        patient_agreement = param_agreement(
            {m: model_vectors[m] for m in model_names}, PATIENT_NAMES)

        # Most/least agreed-upon parameters
        all_agreement = {**intv_agreement, **patient_agreement}
        if all_agreement:
            sorted_agree = sorted(all_agreement.items(), key=lambda x: x[1])
            most_agreed = sorted_agree[0]
            least_agreed = sorted_agree[-1]
        else:
            most_agreed = ("none", 0)
            least_agreed = ("none", 0)

        # Outcome agreement (do models produce similar patient outcomes?)
        outcome_atps = []
        outcome_hets = []
        outcome_benefits = []
        for mname in model_names:
            if mname in model_outcomes:
                outcome_atps.append(model_outcomes[mname]["energy"]["atp_final"])
                outcome_hets.append(model_outcomes[mname]["damage"]["het_final"])
                outcome_benefits.append(
                    model_outcomes[mname]["intervention"]["atp_benefit_terminal"])

        outcome_atp_std = float(np.std(outcome_atps)) if outcome_atps else 0.0
        outcome_het_std = float(np.std(outcome_hets)) if outcome_hets else 0.0

        # Majority vote on intervention strategy
        intervention_sums = defaultdict(float)
        for mname in model_names:
            for param in INTERVENTION_NAMES:
                intervention_sums[param] += model_vectors[mname].get(param, 0.0)
        consensus_intervention = {
            k: round(v / len(model_names), 2) for k, v in intervention_sums.items()
        }

        print(f"  Consensus: cosine={mean_cosine:.3f}, "
              f"most agreed={most_agreed[0]} (std={most_agreed[1]:.3f}), "
              f"least agreed={least_agreed[0]} (std={least_agreed[1]:.3f})")

        scenario_results.append({
            "seed_id": seed_id,
            "scenario": scenario,
            "n_models": len(model_vectors),
            "model_vectors": model_vectors,
            "consensus_intervention": consensus_intervention,
            "mean_cosine_similarity": mean_cosine,
            "pairwise_similarities": pair_sims,
            "intervention_agreement": intv_agreement,
            "patient_agreement": patient_agreement,
            "most_agreed_param": most_agreed[0],
            "least_agreed_param": least_agreed[0],
            "outcome_atp_std": outcome_atp_std,
            "outcome_het_std": outcome_het_std,
            "outcome_atps": {m: float(model_outcomes[m]["energy"]["atp_final"])
                            for m in model_names if m in model_outcomes},
            "outcome_benefits": {m: float(model_outcomes[m]["intervention"]["atp_benefit_terminal"])
                                for m in model_names if m in model_outcomes},
        })

    total_elapsed = time.time() - start_time

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"CLINICAL CONSENSUS COMPLETE — {total_elapsed:.1f}s")
    print(f"{'=' * 70}")

    if scenario_results:
        # Rank scenarios by consensus (high cosine = agreement)
        sorted_scenarios = sorted(scenario_results,
                                  key=lambda x: x["mean_cosine_similarity"],
                                  reverse=True)

        print(f"\nScenarios ranked by model agreement (cosine similarity):")
        for s in sorted_scenarios:
            print(f"  {s['seed_id']:30s}: cosine={s['mean_cosine_similarity']:.3f} "
                  f"ATP_std={s['outcome_atp_std']:.3f} "
                  f"({s['n_models']} models)")

        # Most controversial parameters overall
        all_param_stds = defaultdict(list)
        for s in scenario_results:
            for param, std in {**s["intervention_agreement"],
                               **s["patient_agreement"]}.items():
                all_param_stds[param].append(std)

        print(f"\nParameters ranked by mean disagreement:")
        sorted_params = sorted(all_param_stds.items(),
                               key=lambda x: np.mean(x[1]), reverse=True)
        for param, stds in sorted_params:
            print(f"  {param:25s}: mean_std={np.mean(stds):.3f}")

    # Save
    output = {
        "experiment": "clinical_consensus",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": total_elapsed,
        "n_scenarios": len(scenario_results),
        "n_models": len(MODELS),
        "results": scenario_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
