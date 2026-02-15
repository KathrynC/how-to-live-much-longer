#!/usr/bin/env python3
"""
archetype_matchmaker.py

Which character archetypes produce the best intervention protocols
for which patient types?

Combines PDS mapping (Zimmerman 2025 Ch. 4) with character seed
experiment data to identify archetype→outcome patterns.

Pipeline:
  1. Load character_seed_experiment results
  2. Compute PDS scores for each character (from archetypometrics TSV)
  3. Classify simulation outcomes into tiers (thriving/stable/declining/collapsed)
  4. Identify which PDS regions produce best outcomes per patient profile
  5. Recommend: for a given patient type, which character archetype
     should seed the LLM prompt?

This is a Tier 4 script: requires prior data from character_seed_experiment.py
(no Ollama needed at runtime).

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning
    Construction in Language, as Implemented in Humans and Large
    Language Models (LLMs)." PhD dissertation, University of Vermont.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    INTERVENTION_NAMES, PATIENT_NAMES,
    HETEROPLASMY_CLIFF,
)
from analytics import NumpyEncoder
from pds_mapping import load_characters, compute_pds, pds_to_patient


# ── Outcome classification ─────────────────────────────────────────────────

def classify_outcome(analytics):
    """Classify a simulation outcome into clinically meaningful tiers.

    Tier definitions grounded in Cramer's biological model:

      "collapsed": het >= 0.70 (the heteroplasmy cliff; Cramer Ch. V.K p.66).
        Past this threshold, ATP production drops catastrophically via the
        sigmoid cliff factor. Essentially irreversible due to bistability (fix C4).

      "declining": atp_final < 0.5 OR het_final > 0.6.
        ATP below 0.5 MU/day (half baseline; Cramer Ch. VIII.A Table 3 p.100)
        means severe energy deficit — symptomatic and progressive. het > 0.6
        means dangerously close to the cliff (within 14%).

      "thriving": atp_benefit > 0.05 AND het_final < 0.4.
        Active improvement: the intervention is working (positive ATP benefit
        beyond noise threshold of 0.05) and damage is well-controlled (het < 0.4
        = comfortable 43% margin below cliff). The 0.05 threshold filters out
        trivially small improvements that might be simulation noise.

      "stable": everything else — not collapsed or declining, but not
        demonstrably improving either. Maintenance-level outcome.

    Args:
        analytics: Dict from compute_all() with energy/damage/intervention keys.

    Returns:
        String: "thriving", "stable", "declining", or "collapsed".
    """
    atp_final = analytics["energy"]["atp_final"]
    het_final = analytics["damage"]["het_final"]
    atp_benefit = analytics["intervention"]["atp_benefit_terminal"]

    if het_final >= HETEROPLASMY_CLIFF:
        return "collapsed"
    if atp_final < 0.5 or het_final > 0.6:
        return "declining"
    if atp_benefit > 0.05 and het_final < 0.4:
        return "thriving"
    return "stable"


# ── Data loading ────────────────────────────────────────────────────────────

def load_experiment_results():
    """Load character seed experiment results."""
    candidates = [
        PROJECT / "artifacts" / "character_seed_experiment.json",
        PROJECT / "output" / "character_seed_experiment.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results = data.get("results", data.get("artifacts", []))
            print(f"Loaded {len(results)} trials from {path}")
            return results
    return None


def build_character_pds_lookup(characters):
    """Build a lookup from character name → PDS scores.

    Args:
        characters: List of character dicts from load_characters().

    Returns:
        Dict mapping lowercase character_name → PDS dict.
    """
    if not characters:
        return {}

    available_cols = set(characters[0].keys())
    lookup = {}
    for char in characters:
        name = char.get("character_name", char.get("name", "")).lower().strip()
        if name:
            lookup[name] = compute_pds(char, available_cols)
    return lookup


# ── PDS binning ─────────────────────────────────────────────────────────────

def pds_bin(pds, n_bins=3):
    """Bin PDS scores into discrete categories for aggregation.

    Boundaries at ±0.33 divide the [-1, +1] PDS range into equal thirds:
      low  = [-1.0, -0.33): strongly negative on this dimension
      mid  = [-0.33, +0.33]: neutral / ambiguous
      high = (+0.33, +1.0]: strongly positive on this dimension

    Equal-width bins are chosen over quantile bins because PDS dimensions
    have interpretable zero points (e.g., Power=0 means neither powerful
    nor weak; Zimmerman 2025 Ch. 4). Tercile boundaries at ±1/3 of the
    range are the simplest symmetric partition.

    Args:
        pds: Dict with power, danger, structure (each -1 to +1).
        n_bins: 3 = low/mid/high (only 3 supported).

    Returns:
        Tuple of (power_bin, danger_bin, structure_bin) as strings.
    """
    def bin_val(v):
        if v < -0.33:
            return "low"
        elif v > 0.33:
            return "high"
        return "mid"

    return (bin_val(pds["power"]), bin_val(pds["danger"]),
            bin_val(pds["structure"]))


# ── Patient profile clustering ──────────────────────────────────────────────

def patient_profile_label(patient):
    """Create a human-readable label for a patient profile.

    Args:
        patient: Dict with patient parameter values.

    Returns:
        String label like "young_healthy" or "elderly_inflamed".
    """
    age = patient.get("baseline_age", 50)
    het = patient.get("baseline_heteroplasmy", 0.3)
    inflam = patient.get("inflammation_level", 0.25)

    age_label = "young" if age < 40 else ("middle" if age < 60 else "elderly")
    health_label = ("healthy" if het < 0.3
                    else ("moderate" if het < 0.6 else "damaged"))
    inflam_label = "_inflamed" if inflam >= 0.5 else ""

    return f"{age_label}_{health_label}{inflam_label}"


# ── Main analysis ───────────────────────────────────────────────────────────

def run_matchmaker():
    """Run the archetype-to-protocol matchmaker analysis.

    Returns:
        Dict with analysis results.
    """
    # Load experiment data
    results = load_experiment_results()
    if results is None:
        print("No character experiment data found.")
        print("Run character_seed_experiment.py first.")
        return None

    # Load character PDS data
    characters = load_characters()
    pds_lookup = build_character_pds_lookup(characters)

    successes = [r for r in results if r.get("success") and r.get("analytics")]
    print(f"Successful trials: {len(successes)}/{len(results)}")

    if not successes:
        print("No successful trials to analyze.")
        return None

    # ── Classify outcomes ────────────────────────────────────────────────
    for r in successes:
        r["outcome_tier"] = classify_outcome(r["analytics"])

    tiers = {}
    for r in successes:
        tier = r["outcome_tier"]
        tiers.setdefault(tier, []).append(r)

    print(f"\nOutcome distribution:")
    for tier in ["thriving", "stable", "declining", "collapsed"]:
        n = len(tiers.get(tier, []))
        pct = 100 * n / len(successes) if successes else 0
        print(f"  {tier:12s}: {n:5d} ({pct:.1f}%)")

    # ── Match PDS scores to outcomes ─────────────────────────────────────
    pds_matched = 0
    pds_by_tier = {t: [] for t in ["thriving", "stable", "declining", "collapsed"]}
    pds_by_bin = {}  # (p_bin, d_bin, s_bin) → list of outcome tiers

    for r in successes:
        char_name = r.get("character", "").lower().strip()
        pds = pds_lookup.get(char_name)
        if pds is None:
            continue

        pds_matched += 1
        tier = r["outcome_tier"]
        pds_by_tier[tier].append(pds)

        pds_bin_key = pds_bin(pds)
        pds_by_bin.setdefault(pds_bin_key, []).append(tier)

    print(f"\nPDS-matched trials: {pds_matched}/{len(successes)}")

    # ── PDS statistics per tier ──────────────────────────────────────────
    tier_pds_stats = {}
    for tier in ["thriving", "stable", "declining", "collapsed"]:
        scores = pds_by_tier[tier]
        if len(scores) >= 3:
            powers = [s["power"] for s in scores]
            dangers = [s["danger"] for s in scores]
            structures = [s["structure"] for s in scores]
            tier_pds_stats[tier] = {
                "n": len(scores),
                "power": {"mean": float(np.mean(powers)),
                          "std": float(np.std(powers))},
                "danger": {"mean": float(np.mean(dangers)),
                           "std": float(np.std(dangers))},
                "structure": {"mean": float(np.mean(structures)),
                              "std": float(np.std(structures))},
            }
            print(f"\n  {tier} (n={len(scores)}):")
            print(f"    Power:     {np.mean(powers):+.3f} ± {np.std(powers):.3f}")
            print(f"    Danger:    {np.mean(dangers):+.3f} ± {np.std(dangers):.3f}")
            print(f"    Structure: {np.mean(structures):+.3f} ± {np.std(structures):.3f}")

    # ── PDS bin → success rate ───────────────────────────────────────────
    bin_success_rates = {}
    print(f"\nPDS bin → outcome rates:")
    for bin_key in sorted(pds_by_bin.keys()):
        tier_list = pds_by_bin[bin_key]
        n = len(tier_list)
        if n < 5:
            continue
        thriving_rate = sum(1 for t in tier_list if t == "thriving") / n
        collapsed_rate = sum(1 for t in tier_list if t == "collapsed") / n
        label = f"P={bin_key[0]:4s} D={bin_key[1]:4s} S={bin_key[2]:4s}"
        bin_success_rates[str(bin_key)] = {
            "n": n,
            "thriving_rate": float(thriving_rate),
            "collapsed_rate": float(collapsed_rate),
        }
        print(f"  {label}  n={n:4d}  "
              f"thriving={thriving_rate:.1%}  collapsed={collapsed_rate:.1%}")

    # ── Per-patient-profile analysis ─────────────────────────────────────
    profile_archetypes = {}
    for r in successes:
        patient = r.get("patient")
        if patient is None:
            continue
        label = patient_profile_label(patient)
        profile_archetypes.setdefault(label, []).append(r)

    print(f"\nPatient profile → best archetype bins:")
    profile_recommendations = {}
    for label in sorted(profile_archetypes.keys()):
        trials = profile_archetypes[label]
        if len(trials) < 10:
            continue

        # Find PDS bin with best outcomes for this patient profile
        profile_pds_bins = {}
        for r in trials:
            char_name = r.get("character", "").lower().strip()
            pds = pds_lookup.get(char_name)
            if pds is None:
                continue
            bin_key = pds_bin(pds)
            profile_pds_bins.setdefault(bin_key, []).append(r)

        best_bin = None
        best_score = -999
        for bin_key, bin_trials in profile_pds_bins.items():
            if len(bin_trials) < 3:
                continue
            benefits = [t["analytics"]["intervention"]["atp_benefit_terminal"]
                        for t in bin_trials]
            score = float(np.median(benefits))
            if score > best_score:
                best_score = score
                best_bin = bin_key

        if best_bin is not None:
            n_trials = len(profile_pds_bins[best_bin])
            profile_recommendations[label] = {
                "best_pds_bin": best_bin,
                "median_benefit": best_score,
                "n_trials": n_trials,
                "total_trials": len(trials),
            }
            print(f"  {label:30s}: "
                  f"P={best_bin[0]:4s} D={best_bin[1]:4s} S={best_bin[2]:4s}  "
                  f"benefit={best_score:+.3f} (n={n_trials})")

    # ── Top character-protocol pairs ─────────────────────────────────────
    print(f"\nTop 15 character-protocol pairs (by ATP benefit):")
    sorted_by_benefit = sorted(
        successes,
        key=lambda r: r["analytics"]["intervention"]["atp_benefit_terminal"],
        reverse=True)

    top_pairs = []
    for r in sorted_by_benefit[:15]:
        char_name = r.get("character", "")
        story = r.get("story", "")
        benefit = r["analytics"]["intervention"]["atp_benefit_terminal"]
        het_final = r["analytics"]["damage"]["het_final"]
        model = r.get("model", "")
        pds = pds_lookup.get(char_name.lower().strip())
        pds_str = (f"P={pds['power']:+.2f} D={pds['danger']:+.2f} "
                   f"S={pds['structure']:+.2f}" if pds else "no PDS")
        print(f"  {char_name:25s} ({story:20s}) "
              f"benefit={benefit:+.3f} het={het_final:.3f} "
              f"[{model}] {pds_str}")
        top_pairs.append({
            "character": char_name,
            "story": story,
            "model": model,
            "atp_benefit": float(benefit),
            "het_final": float(het_final),
            "pds": pds,
        })

    # ── Intervention patterns by PDS ─────────────────────────────────────
    print(f"\nIntervention patterns by PDS archetype:")
    archetype_interventions = {}
    for r in successes:
        char_name = r.get("character", "").lower().strip()
        pds = pds_lookup.get(char_name)
        intervention = r.get("intervention")
        if pds is None or intervention is None:
            continue
        bin_key = pds_bin(pds)
        archetype_interventions.setdefault(bin_key, []).append(intervention)

    intervention_by_archetype = {}
    for bin_key in sorted(archetype_interventions.keys()):
        interventions = archetype_interventions[bin_key]
        if len(interventions) < 10:
            continue
        means = {}
        label = f"P={bin_key[0]:4s} D={bin_key[1]:4s} S={bin_key[2]:4s}"
        parts = []
        for param in INTERVENTION_NAMES:
            vals = [iv[param] for iv in interventions if param in iv]
            if vals:
                mean_val = float(np.mean(vals))
                means[param] = mean_val
                parts.append(f"{param[:4]}={mean_val:.2f}")
        intervention_by_archetype[str(bin_key)] = means
        print(f"  {label}  n={len(interventions):4d}  {', '.join(parts)}")

    # ── Assemble output ──────────────────────────────────────────────────
    output = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": len(results),
        "n_successes": len(successes),
        "n_pds_matched": pds_matched,
        "outcome_distribution": {
            tier: len(tiers.get(tier, []))
            for tier in ["thriving", "stable", "declining", "collapsed"]
        },
        "tier_pds_stats": tier_pds_stats,
        "bin_success_rates": bin_success_rates,
        "profile_recommendations": profile_recommendations,
        "top_pairs": top_pairs,
        "intervention_by_archetype": intervention_by_archetype,
    }

    return output


# ── Standalone ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Archetype-to-Protocol Matchmaker (Zimmerman 2025 + Cramer 2025)")
    print("=" * 70)

    output = run_matchmaker()

    if output:
        out_path = PROJECT / "artifacts" / "archetype_matchmaker.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, cls=NumpyEncoder, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        print("\nNo results to save. Run character_seed_experiment.py first.")
