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
import subprocess
import time as time_module
from datetime import datetime

import numpy as np

from constants import (
    OLLAMA_URL, OFFER_MODEL, CONFIRMATION_MODEL, REASONING_MODELS,
    INTERVENTION_PARAMS, PATIENT_PARAMS,
    INTERVENTION_NAMES, PATIENT_NAMES, ALL_PARAM_NAMES,
    CLINICAL_SEEDS, DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    SIM_YEARS, HETEROPLASMY_CLIFF,
    snap_all,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder


# ── Ollama interface ─────────────────────────────────────────────────────────

def query_ollama(model, prompt, temperature=0.8, max_tokens=800, timeout=120):
    """Send a prompt to local Ollama and return the response text.

    Args:
        model: Ollama model name.
        prompt: The prompt string.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        Response string, or None on failure.
    """
    effective_max = 2000 if model in REASONING_MODELS else max_tokens
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": effective_max},
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        if "error" in data:
            return None
        return data["response"]
    except Exception:
        return None


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_response(response):
    """Parse a JSON object from an LLM response, handling common artifacts.

    Strips markdown code fences, <think>...</think> tags, and finds the
    outermost { ... } pair.

    Args:
        response: Raw string from LLM.

    Returns:
        Parsed dict, or None on failure.
    """
    if not response:
        return None

    text = response.strip()

    # Strip think tags (reasoning models)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ── Offer wave prompt ────────────────────────────────────────────────────────

OFFER_PROMPT = """\
You are a mitochondrial medicine specialist designing a personalized \
intervention protocol. You must choose BOTH intervention parameters AND \
characterize the patient based on the clinical scenario.

INTERVENTION PARAMETERS (6 params, each 0.0 to 1.0):
  rapamycin_dose: mTOR inhibition → enhanced mitophagy (0=none, 1=maximum)
  nad_supplement: NAD+ precursor (NMN/NR) dose (0=none, 1=maximum)
  senolytic_dose: Senolytic drug dose (dasatinib+quercetin) (0=none, 1=maximum)
  yamanaka_intensity: Partial reprogramming (OSKM) intensity (0=none, 1=max) \
WARNING: costs 3-5 MU of ATP — only use if patient can afford the energy
  transplant_rate: Mitochondrial transplant rate via mitlets (0=none, 1=maximum)
  exercise_level: Exercise intensity for hormetic adaptation (0=sedentary, 1=intense)

PATIENT PARAMETERS (6 params):
  baseline_age: Starting age in years (20-90)
  baseline_heteroplasmy: Fraction of damaged mtDNA (0.0-0.95). \
CRITICAL: the heteroplasmy cliff is at ~0.7 — above this, ATP collapses
  baseline_nad_level: NAD+ level (0.2-1.0, declines with age)
  genetic_vulnerability: Susceptibility to mtDNA damage (0.5-2.0, 1.0=normal)
  metabolic_demand: Tissue energy need (0.5=skin, 1.0=normal, 2.0=brain)
  inflammation_level: Chronic inflammation (0.0-1.0)

CLINICAL SCENARIO:
{scenario}

Think carefully about this patient:
- How close are they to the heteroplasmy cliff?
- What is the most urgent intervention?
- Can they afford Yamanaka's energy cost?
- Would transplant help (adding healthy copies)?
- Is exercise safe given their current energy reserves?

Choose intervention values from: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
Choose patient values within the ranges described above.

Output a single JSON object with ALL 12 keys:
{{"rapamycin_dose":_, "nad_supplement":_, "senolytic_dose":_, \
"yamanaka_intensity":_, "transplant_rate":_, "exercise_level":_, \
"baseline_age":_, "baseline_heteroplasmy":_, "baseline_nad_level":_, \
"genetic_vulnerability":_, "metabolic_demand":_, "inflammation_level":_}}"""


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
    offer_response = query_ollama(offer_model, offer_prompt,
                                   temperature=0.7, max_tokens=800)

    if offer_response is None:
        if verbose:
            print("  ERROR: Offer wave failed (Ollama unreachable?)")
        return None

    raw_params = parse_response(offer_response)
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

    confirm_response = query_ollama(confirm_model, confirm_prompt,
                                     temperature=0.3, max_tokens=600)

    confirmation = None
    if confirm_response:
        confirmation = parse_response(confirm_response)

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

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # Run just the first seed for quick testing
        seed = CLINICAL_SEEDS[0]
        artifact = run_experiment(seed)
        if artifact:
            print(json.dumps(artifact, cls=NumpyEncoder, indent=2))
    else:
        # Run all 10 clinical scenarios
        run_all_experiments()
