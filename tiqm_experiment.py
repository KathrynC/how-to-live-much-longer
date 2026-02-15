"""TIQM experiment: LLM-driven mitochondrial intervention design.

Implements the Transactional Interpretation of Quantum Mechanics (TIQM)
pipeline adapted from the parent Evolutionary-Robotics project:

    1. Offer wave: LLM generates a 12D intervention + patient vector
       from a clinical scenario seed
    2. Simulation: Run the mitochondrial ODE for 30 years, compute
       4-pillar analytics
    3. Confirmation wave: A DIFFERENT VLM/LLM evaluates the trajectory
       and rates resonance

Uses different models for offer vs confirmation to prevent
self-confirmation bias (Cramer's TIQM principle: the offer and
confirmation waves must originate from different sources).

Usage:
    python tiqm_experiment.py
"""

import json
import os
import time as time_module
from datetime import datetime

import numpy as np

from constants import (
    OFFER_MODEL, CONFIRMATION_MODEL,
    INTERVENTION_NAMES, PATIENT_NAMES,
    CLINICAL_SEEDS, DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    SIM_YEARS, HETEROPLASMY_CLIFF,
    snap_all,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder
from llm_common import query_ollama_raw, parse_json_response


# ── Prompt templates ─────────────────────────────────────────────────────────
# Import from prompt_templates.py for both numeric (original) and diegetic
# (Zimmerman-informed) prompt styles. The style is selected at runtime
# via the --style flag.

from prompt_templates import PROMPT_STYLES

# Default offer prompt (numeric style, for backwards compatibility)
OFFER_PROMPT = PROMPT_STYLES["numeric"]["offer"]


# ── Confirmation wave prompt ─────────────────────────────────────────────────

CONFIRMATION_PROMPT = """\
A mitochondrial aging simulation was run with these results:

  Simulation: {sim_years} years, starting age {age}
  Initial heteroplasmy: {het_initial:.2f} (cliff at {cliff})
  Final heteroplasmy: {het_final:.2f}
  Time to cliff: {time_to_cliff}
  Initial ATP: {atp_initial:.3f} MU/day
  Final ATP: {atp_final:.3f} MU/day
  ATP slope: {atp_slope:+.4f} MU/day/year
  Time to energy crisis: {time_to_crisis}
  ROS-heteroplasmy correlation: {ros_het_corr:.3f}
  Final senescent fraction: {sen_final:.3f}
  Membrane potential CV: {psi_cv:.3f}
  Intervention benefit (ATP): {atp_benefit:+.3f} vs no treatment
  Intervention benefit (heteroplasmy): {het_benefit:+.3f} vs no treatment

  Intervention used:
    Rapamycin: {rapamycin_dose}
    NAD+ supplement: {nad_supplement}
    Senolytics: {senolytic_dose}
    Yamanaka: {yamanaka_intensity}
    Transplant: {transplant_rate}
    Exercise: {exercise_level}

The clinical scenario was: "{scenario}"

Questions:
1. Describe the trajectory in 1-2 sentences (as if watching a patient's \
cellular health over decades).
2. Does this intervention protocol MATCH the clinical scenario? Rate the \
clinical resonance from 0.0 (no connection) to 1.0 (perfect match).
3. Does the simulation trajectory look physiologically plausible? Rate \
trajectory resonance from 0.0 (unrealistic) to 1.0 (highly plausible).
4. What would you change to better serve this patient?

Output a JSON object:
{{"trajectory_description": "...", "resonance_behavior": 0.X, \
"resonance_trajectory": 0.X, "suggestion": "..."}}"""


# ── Run single experiment ────────────────────────────────────────────────────

def run_experiment(seed, offer_model=None, confirm_model=None, verbose=True):
    """Run a single TIQM experiment for a clinical scenario seed.

    Args:
        seed: Dict with "id" and "description" from CLINICAL_SEEDS.
        offer_model: Model for offer wave (default: OFFER_MODEL).
        confirm_model: Model for confirmation wave (default: CONFIRMATION_MODEL).
        verbose: Print progress.

    Returns:
        Dict with full experiment results, or None on failure.
    """
    if offer_model is None:
        offer_model = OFFER_MODEL
    if confirm_model is None:
        confirm_model = CONFIRMATION_MODEL

    seed_id = seed["id"]
    scenario = seed["description"]

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Experiment: {seed_id}")
        print(f"  Offer model: {offer_model}")
        print(f"  Confirmation model: {confirm_model}")
        print(f"{'─' * 60}")

    # ── Offer wave ────────────────────────────────────────────────────────
    if verbose:
        print("  [1/3] Offer wave: generating intervention protocol...")

    offer_prompt = OFFER_PROMPT.format(scenario=scenario)
    offer_response = query_ollama_raw(offer_model, offer_prompt,
                                      temperature=0.7, max_tokens=800)

    if offer_response is None:
        if verbose:
            print("  ERROR: Offer wave failed (Ollama unreachable?)")
        return None

    raw_params = parse_json_response(offer_response)
    if raw_params is None:
        if verbose:
            print(f"  ERROR: Could not parse offer response")
            print(f"  Raw response: {offer_response[:200]}...")
        return None

    # Snap to grid
    snapped = snap_all(raw_params)

    # Split into intervention and patient
    intervention = {k: snapped.get(k, DEFAULT_INTERVENTION[k])
                    for k in INTERVENTION_NAMES}
    patient = {k: snapped.get(k, DEFAULT_PATIENT[k])
               for k in PATIENT_NAMES}

    if verbose:
        print(f"  Intervention: {intervention}")
        print(f"  Patient: {patient}")

    # ── Simulation ────────────────────────────────────────────────────────
    if verbose:
        print("  [2/3] Simulation: running 30-year ODE...")

    t0 = time_module.time()
    result = simulate(intervention=intervention, patient=patient)
    sim_time = time_module.time() - t0

    # Compute analytics
    baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
    analytics = compute_all(result, baseline)

    if verbose:
        print(f"  Simulation completed in {sim_time:.2f}s")
        print(f"  Final ATP: {result['states'][-1, 2]:.4f}")
        print(f"  Final heteroplasmy: {result['heteroplasmy'][-1]:.4f}")

    # ── Confirmation wave ─────────────────────────────────────────────────
    if verbose:
        print("  [3/3] Confirmation wave: evaluating trajectory...")

    energy = analytics["energy"]
    damage = analytics["damage"]
    dynamics = analytics["dynamics"]
    interv = analytics["intervention"]

    confirm_prompt = CONFIRMATION_PROMPT.format(
        sim_years=SIM_YEARS,
        age=patient["baseline_age"],
        het_initial=damage["het_initial"],
        cliff=HETEROPLASMY_CLIFF,
        het_final=damage["het_final"],
        time_to_cliff=(f"{damage['time_to_cliff_years']:.1f} years"
                       if damage["time_to_cliff_years"] < 999
                       else "never (within simulation)"),
        atp_initial=energy["atp_initial"],
        atp_final=energy["atp_final"],
        atp_slope=energy["atp_slope"],
        time_to_crisis=(f"{energy['time_to_crisis_years']:.1f} years"
                        if energy["time_to_crisis_years"] < 999
                        else "never (within simulation)"),
        ros_het_corr=dynamics["ros_het_correlation"],
        sen_final=dynamics["senescent_final"],
        psi_cv=dynamics["membrane_potential_cv"],
        atp_benefit=interv["atp_benefit_terminal"],
        het_benefit=interv["het_benefit_terminal"],
        rapamycin_dose=intervention["rapamycin_dose"],
        nad_supplement=intervention["nad_supplement"],
        senolytic_dose=intervention["senolytic_dose"],
        yamanaka_intensity=intervention["yamanaka_intensity"],
        transplant_rate=intervention["transplant_rate"],
        exercise_level=intervention["exercise_level"],
        scenario=scenario,
    )

    confirm_response = query_ollama_raw(confirm_model, confirm_prompt,
                                        temperature=0.3, max_tokens=600)

    confirmation = None
    if confirm_response:
        confirmation = parse_json_response(confirm_response)

    if verbose:
        if confirmation:
            print(f"  Resonance (behavior): {confirmation.get('resonance_behavior', '?')}")
            print(f"  Resonance (trajectory): {confirmation.get('resonance_trajectory', '?')}")
            if "trajectory_description" in confirmation:
                desc = confirmation["trajectory_description"]
                print(f"  Description: {desc[:100]}...")
        else:
            print("  WARNING: Confirmation wave failed or unparseable")

    # ── Assemble artifact ─────────────────────────────────────────────────
    artifact = {
        "seed_id": seed_id,
        "scenario": scenario,
        "offer_model": offer_model,
        "confirmation_model": confirm_model,
        "timestamp": datetime.now().isoformat(),
        "intervention": intervention,
        "patient": patient,
        "raw_params": raw_params,
        "analytics": analytics,
        "confirmation": confirmation,
        "resonance_behavior": (confirmation.get("resonance_behavior", 0.0)
                               if confirmation else 0.0),
        "resonance_trajectory": (confirmation.get("resonance_trajectory", 0.0)
                                  if confirmation else 0.0),
        "simulation_time_sec": sim_time,
    }

    return artifact


# ── Run all experiments ──────────────────────────────────────────────────────

def run_all_experiments(seeds=None, output_dir="output"):
    """Run TIQM experiments for all clinical scenario seeds.

    Args:
        seeds: List of seed dicts (default: CLINICAL_SEEDS).
        output_dir: Directory for JSON artifacts.

    Returns:
        List of experiment artifact dicts.
    """
    if seeds is None:
        seeds = CLINICAL_SEEDS

    os.makedirs(output_dir, exist_ok=True)
    artifacts = []
    start_time = time_module.time()

    print("=" * 60)
    print("  TIQM Mitochondrial Intervention Experiment")
    print(f"  {len(seeds)} clinical scenarios")
    print(f"  Offer model: {OFFER_MODEL}")
    print(f"  Confirmation model: {CONFIRMATION_MODEL}")
    print("=" * 60)

    for i, seed in enumerate(seeds):
        print(f"\n  [{i+1}/{len(seeds)}] {seed['id']}")
        artifact = run_experiment(seed)
        if artifact:
            artifacts.append(artifact)
            # Save individual artifact
            path = os.path.join(output_dir, f"tiqm_{seed['id']}.json")
            with open(path, "w") as f:
                json.dump(artifact, f, cls=NumpyEncoder, indent=2)
            print(f"  Saved: {path}")
        else:
            print(f"  SKIPPED (experiment failed)")

    # Save combined results
    total_time = time_module.time() - start_time
    summary = {
        "experiment": "tiqm_mitochondrial_intervention",
        "timestamp": datetime.now().isoformat(),
        "n_seeds": len(seeds),
        "n_completed": len(artifacts),
        "total_time_sec": total_time,
        "offer_model": OFFER_MODEL,
        "confirmation_model": CONFIRMATION_MODEL,
        "artifacts": artifacts,
    }
    summary_path = os.path.join(output_dir, "tiqm_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, cls=NumpyEncoder, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    # Print summary statistics
    if artifacts:
        res_b = [a["resonance_behavior"] for a in artifacts]
        res_t = [a["resonance_trajectory"] for a in artifacts]
        print(f"\n  Resonance (behavior):  mean={np.mean(res_b):.3f}, "
              f"min={np.min(res_b):.3f}, max={np.max(res_b):.3f}")
        print(f"  Resonance (trajectory): mean={np.mean(res_t):.3f}, "
              f"min={np.min(res_t):.3f}, max={np.max(res_t):.3f}")

    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 60)

    return artifacts


# ── Standalone execution ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    # Parse --style flag
    style = "numeric"
    if "--style" in args:
        idx = args.index("--style")
        if idx + 1 < len(args):
            style = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        if style not in PROMPT_STYLES:
            print(f"Unknown style '{style}'. Available: {list(PROMPT_STYLES.keys())}")
            sys.exit(1)
        OFFER_PROMPT = PROMPT_STYLES[style]["offer"]
        print(f"Using prompt style: {style}")

    if "--single" in args:
        # Run just the first seed for quick testing
        seed = CLINICAL_SEEDS[0]
        artifact = run_experiment(seed)
        if artifact:
            print(json.dumps(artifact, cls=NumpyEncoder, indent=2))
    elif "--contrastive" in args:
        # Run contrastive mode: generate cautious vs bold protocols
        contrastive_prompt = PROMPT_STYLES["contrastive"]["offer"]
        seeds = CLINICAL_SEEDS[:3] if "--single" not in args else CLINICAL_SEEDS[:1]
        print("=" * 60)
        print("  CONTRASTIVE MODE: Dr. Cautious vs Dr. Bold")
        print("=" * 60)
        for seed in seeds:
            scenario = seed["description"]
            prompt = contrastive_prompt.format(scenario=scenario)
            response = query_ollama_raw(OFFER_MODEL, prompt,
                                         temperature=0.7, max_tokens=1200)
            parsed = parse_json_response(response)
            if parsed and "cautious" in parsed and "bold" in parsed:
                print(f"\n  {seed['id']}:")
                for approach in ["cautious", "bold"]:
                    params = parsed[approach]
                    snapped = snap_all(params)
                    intervention = {k: snapped.get(k, DEFAULT_INTERVENTION[k])
                                    for k in INTERVENTION_NAMES}
                    patient = {k: snapped.get(k, DEFAULT_PATIENT[k])
                               for k in PATIENT_NAMES}
                    result = simulate(intervention=intervention, patient=patient)
                    print(f"    {approach:10s}: het {result['heteroplasmy'][0]:.3f}"
                          f"→{result['heteroplasmy'][-1]:.3f}  "
                          f"ATP {result['states'][0,2]:.3f}"
                          f"→{result['states'][-1,2]:.3f}")
            else:
                print(f"\n  {seed['id']}: could not parse contrastive response")
    else:
        # Run all 10 clinical scenarios
        run_all_experiments()
