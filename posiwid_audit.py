#!/usr/bin/env python3
"""
posiwid_audit.py

POSIWID alignment audit: "The Purpose Of a System Is What It Does."

Measures the gap between LLM-intended outcomes and actual simulation
outcomes. Zimmerman (2025) §3.5.2 discusses how the tokenizer's objective
(compress orthographically) competes with the model's objective (learn
semantics), and how system purposes are revealed by what systems actually
do, not what they claim. This script quantifies the gap between what
an LLM says it intends and what the simulation actually produces.

Pipeline per scenario:
  1. Ask LLM: "What outcome do you INTEND for this patient?"
     (expected heteroplasmy change, expected ATP trajectory)
  2. Ask LLM: "Generate a 12D intervention+patient vector."
  3. Simulate with the generated vector
  4. Compare intended vs actual outcomes
  5. Score alignment

This reveals systematic biases: does the LLM understand the model's
dynamics, or is it producing "plausible-sounding" parameters that
don't achieve the intended effect?

Scale: 10 scenarios x 4 models x 2 queries = 80 LLM calls + 40 sims
Estimated time: ~15-20 min (requires Ollama)

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning
    Construction in Language, as Implemented in Humans and Large
    Language Models (LLMs)." PhD dissertation, University of Vermont.
    (POSIWID from Stafford Beer, 1974)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    CLINICAL_SEEDS, DEFAULT_INTERVENTION,
    INTERVENTION_NAMES, PATIENT_NAMES,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder
from llm_common import MODELS, query_ollama_raw, parse_json_response, split_vector


# ── Prompts ────────────────────────────────────────────────────────────────

INTENTION_PROMPT = """\
You are a mitochondrial medicine specialist. Read this clinical scenario:

{scenario}

Before designing a treatment, state your INTENDED OUTCOMES:

1. What do you expect to happen to the patient's heteroplasmy (fraction
   of damaged mtDNA) over 30 years with your treatment? Will it increase,
   decrease, or stabilize? By roughly how much?

2. What do you expect to happen to the patient's ATP production (cellular
   energy, baseline = 1.0 MU/day) over 30 years? Will it improve,
   decline, or stabilize?

3. What is the biggest risk of your planned intervention?

Output a JSON object:
{{"expected_het_change": <float, negative=improvement>,
  "expected_final_het": <float, 0.0-1.0>,
  "expected_atp_change": <float, positive=improvement>,
  "expected_final_atp": <float, 0.0-1.5>,
  "biggest_risk": "<string>",
  "confidence": <float, 0.0-1.0>}}"""


PROTOCOL_PROMPT = """\
You are a mitochondrial medicine specialist. Design a treatment for:

{scenario}

Output a JSON object with ALL 12 keys:
  Intervention (0.0-1.0): rapamycin_dose, nad_supplement, senolytic_dose,
    yamanaka_intensity (WARNING: costs 3-5 MU ATP!), transplant_rate,
    exercise_level
  Patient: baseline_age (20-90), baseline_heteroplasmy (0.0-0.95),
    baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0),
    metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)

Choose intervention values from: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
Brief reasoning (1-2 sentences), then ONLY the JSON object."""


# ── Alignment scoring ─────────────────────────────────────────────────────

def score_alignment(intention, actual_het_change, actual_atp_change,
                    actual_final_het, actual_final_atp):
    """Score how well actual simulation outcomes match LLM's stated intentions.

    POSIWID principle (Stafford Beer, 1974): "The Purpose Of a System Is What
    It Does" — not what it claims to intend. This function quantifies the gap
    between what the LLM SAYS it expects and what the simulation ACTUALLY
    produces. Low alignment reveals that the LLM generates "plausible-sounding"
    parameters without understanding the ODE dynamics (Zimmerman 2025 §3.5.2:
    system purpose = what it does, not what it claims).

    Scoring components:

    Direction alignment (binary 0/1):
      Did the metric move in the expected direction? This is the most basic
      test — an LLM that says "heteroplasmy will decrease" but produces
      parameters that increase it has fundamentally misunderstood the model.

    Magnitude alignment (continuous 0-1):
      How close was the predicted final value? The × 2 scaling factor means
      an error of 0.5 in final heteroplasmy (e.g., predicted 0.3, actual 0.8)
      scores zero. This is calibrated to the biologically meaningful range:
      the difference between "healthy" (het=0.2) and "cliff" (het=0.7) is 0.5,
      so errors of that magnitude deserve a zero score.

    The × 5 scaling for "expected no change" means a change of 0.2 in
    heteroplasmy scores zero — reflecting that stability prediction should be
    fairly precise (0.2 change over 30yr is clinically significant).

    Args:
        intention: Dict from LLM intention query with expected_het_change,
            expected_final_het, expected_atp_change, expected_final_atp.
        actual_het_change: Float, actual heteroplasmy change over 30yr.
        actual_atp_change: Float, actual ATP change over 30yr.
        actual_final_het: Float, final heteroplasmy.
        actual_final_atp: Float, final ATP.

    Returns:
        Dict with alignment scores (0-1, higher = better alignment):
        het_direction, het_magnitude, atp_direction, atp_magnitude, overall.
    """
    scores = {}

    # Direction alignment: did het move in the expected direction?
    exp_het = intention.get("expected_het_change", 0)
    if exp_het != 0:
        # Same sign = correct direction (binary: got it right or wrong)
        direction_match = 1.0 if (exp_het * actual_het_change > 0) else 0.0
    else:
        # Expected no change — score by magnitude of unexpected change.
        # ×5: an unexpected change of 0.2 over 30yr scores zero (clinically significant)
        direction_match = max(0.0, 1.0 - abs(actual_het_change) * 5)
    scores["het_direction"] = direction_match

    # Magnitude alignment: how close was the predicted final value?
    # ×2: error of 0.5 (= full healthy-to-cliff range) scores zero
    exp_final_het = intention.get("expected_final_het", 0.3)
    het_mag_error = abs(actual_final_het - exp_final_het)
    scores["het_magnitude"] = max(0.0, 1.0 - het_mag_error * 2)

    # ATP direction (same logic as het)
    exp_atp = intention.get("expected_atp_change", 0)
    if exp_atp != 0:
        direction_match_atp = 1.0 if (exp_atp * actual_atp_change > 0) else 0.0
    else:
        direction_match_atp = max(0.0, 1.0 - abs(actual_atp_change) * 5)
    scores["atp_direction"] = direction_match_atp

    # ATP magnitude: ×2 scaling same rationale (ATP range ~0-1.5 MU/day;
    # Cramer Ch. VIII.A Table 3 p.100: baseline = 1.0 MU/day)
    exp_final_atp = intention.get("expected_final_atp", 0.8)
    atp_mag_error = abs(actual_final_atp - exp_final_atp)
    scores["atp_magnitude"] = max(0.0, 1.0 - atp_mag_error * 2)

    # Overall: unweighted mean of all 4 components
    scores["overall"] = np.mean([
        scores["het_direction"],
        scores["het_magnitude"],
        scores["atp_direction"],
        scores["atp_magnitude"],
    ])

    return scores


# ── Main ──────────────────────────────────────────────────────────────────

def run_audit(models=None, seeds=None):
    """Run the POSIWID alignment audit.

    Args:
        models: List of model dicts (default: MODELS).
        seeds: List of scenario seed dicts (default: CLINICAL_SEEDS).

    Returns:
        Dict with audit results.
    """
    if models is None:
        models = MODELS[:2]  # Use 2 models by default to save time
    if seeds is None:
        seeds = CLINICAL_SEEDS

    print("=" * 70)
    print("POSIWID Alignment Audit")
    print(f"Models: {[m['name'] for m in models]}")
    print(f"Scenarios: {len(seeds)}")
    print("=" * 70)

    results = []
    t0 = time.time()

    for seed in seeds:
        scenario = seed["description"]
        seed_id = seed["id"]

        for model_info in models:
            model = model_info["name"]
            print(f"\n  [{seed_id}] [{model}]")

            # Step 1: Get intention
            print(f"    Querying intention...")
            intention_prompt = INTENTION_PROMPT.format(scenario=scenario)
            intention_raw = query_ollama_raw(model, intention_prompt,
                                             temperature=0.3, max_tokens=600)
            intention = parse_json_response(intention_raw)

            if intention is None:
                print(f"    WARNING: Could not parse intention response")
                intention = {
                    "expected_het_change": 0.0,
                    "expected_final_het": 0.3,
                    "expected_atp_change": 0.0,
                    "expected_final_atp": 0.8,
                    "confidence": 0.0,
                }

            print(f"    Intention: het_change={intention.get('expected_het_change', '?')}  "
                  f"atp_change={intention.get('expected_atp_change', '?')}  "
                  f"confidence={intention.get('confidence', '?')}")

            # Step 2: Get protocol
            print(f"    Querying protocol...")
            protocol_prompt = PROTOCOL_PROMPT.format(scenario=scenario)
            protocol_raw = query_ollama_raw(model, protocol_prompt,
                                             temperature=0.7, max_tokens=800)
            protocol = parse_json_response(protocol_raw)

            if protocol is None:
                print(f"    WARNING: Could not parse protocol response")
                continue

            # Split into intervention and patient
            from llm_common import parse_intervention_vector
            snapped = parse_intervention_vector(protocol_raw)
            if snapped is None:
                print(f"    WARNING: Could not snap protocol to grid")
                continue

            intervention, patient = split_vector(snapped)

            # Step 3: Simulate
            result = simulate(intervention=intervention, patient=patient)
            baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)

            # Extract outcomes
            het_initial = result["heteroplasmy"][0]
            het_final = result["heteroplasmy"][-1]
            actual_het_change = het_final - het_initial
            atp_initial = result["states"][0, 2]
            atp_final = result["states"][-1, 2]
            actual_atp_change = atp_final - atp_initial

            print(f"    Actual: het {het_initial:.3f}→{het_final:.3f} "
                  f"(Δ={actual_het_change:+.3f})  "
                  f"ATP {atp_initial:.3f}→{atp_final:.3f} "
                  f"(Δ={actual_atp_change:+.3f})")

            # Step 4: Score alignment
            alignment = score_alignment(
                intention, actual_het_change, actual_atp_change,
                het_final, atp_final)

            print(f"    Alignment: overall={alignment['overall']:.3f}  "
                  f"het_dir={alignment['het_direction']:.1f}  "
                  f"het_mag={alignment['het_magnitude']:.3f}  "
                  f"atp_dir={alignment['atp_direction']:.1f}  "
                  f"atp_mag={alignment['atp_magnitude']:.3f}")

            results.append({
                "seed_id": seed_id,
                "model": model,
                "intention": intention,
                "intervention": intervention,
                "patient": patient,
                "actual": {
                    "het_initial": float(het_initial),
                    "het_final": float(het_final),
                    "het_change": float(actual_het_change),
                    "atp_initial": float(atp_initial),
                    "atp_final": float(atp_final),
                    "atp_change": float(actual_atp_change),
                },
                "alignment": alignment,
            })

    elapsed = time.time() - t0

    # Summary statistics
    if results:
        overall_scores = [r["alignment"]["overall"] for r in results]
        het_dir_scores = [r["alignment"]["het_direction"] for r in results]
        atp_dir_scores = [r["alignment"]["atp_direction"] for r in results]

        print("\n" + "=" * 70)
        print("POSIWID AUDIT SUMMARY")
        print("=" * 70)
        print(f"  Trials: {len(results)}")
        print(f"  Overall alignment: {np.mean(overall_scores):.3f} "
              f"(std={np.std(overall_scores):.3f})")
        print(f"  Het direction accuracy: {np.mean(het_dir_scores):.1%}")
        print(f"  ATP direction accuracy: {np.mean(atp_dir_scores):.1%}")
        print(f"  Time: {elapsed:.0f}s")

        # Per-model breakdown
        for model_info in models:
            model = model_info["name"]
            model_results = [r for r in results if r["model"] == model]
            if model_results:
                model_overall = [r["alignment"]["overall"] for r in model_results]
                print(f"\n  {model}: alignment={np.mean(model_overall):.3f} "
                      f"(n={len(model_results)})")

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": len(results),
        "elapsed_sec": elapsed,
        "models": [m["name"] for m in models],
        "results": results,
        "summary": {
            "mean_overall": float(np.mean(overall_scores)) if results else 0,
            "het_direction_accuracy": float(np.mean(het_dir_scores)) if results else 0,
            "atp_direction_accuracy": float(np.mean(atp_dir_scores)) if results else 0,
        } if results else {},
    }

    return output


if __name__ == "__main__":
    result = run_audit()

    output_path = PROJECT / "artifacts" / "posiwid_audit.json"
    with open(output_path, "w") as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)
    print(f"\nResults saved to {output_path}")
