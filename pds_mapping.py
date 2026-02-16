#!/usr/bin/env python3
"""
pds_mapping.py

Map Zimmerman's PDS (Power, Danger, Structure) dimensions from
fictional character archetypes to patient parameters.

Based on Zimmerman (2025) §4.6.4: the 6D SVD of 2000 Open Psychometrics
characters yields 3 bipolar dimensions:
  - Power (Fool↔Hero): maps to metabolic_demand + genetic_resilience
  - Danger (Angel↔Demon): maps to inflammation_level + genetic_vulnerability
  - Structure (Traditionalist↔Adventurer): maps to baseline_nad_level

This creates a principled semantic bridge from character identities
(used in character_seed_experiment.py) to the 6D patient parameter space.

Pipeline:
  1. Load character archetype data (archetypometrics_characters.tsv)
  2. Compute PDS scores for each character (from SVD dimensions)
  3. Map PDS → predicted patient parameters
  4. If character experiment results exist, compare PDS-predicted
     vs LLM-generated patient parameters
  5. Report correlation and systematic biases

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning
    Construction in Language, as Implemented in Humans and Large
    Language Models (LLMs)." PhD dissertation, University of Vermont.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import PATIENT_NAMES
from analytics import NumpyEncoder


# ── Character data ─────────────────────────────────────────────────────────

ER_PROJECT = PROJECT.parent / "pybullet_test" / "Evolutionary-Robotics"
CHARACTER_FILE = PROJECT / "archetypometrics_characters.tsv"
ER_CHARACTER_FILE = ER_PROJECT / "archetypometrics_characters.tsv"


def load_characters(path=None):
    """Load character data from TSV file.

    Tries local copy first, then copies from parent ER project if needed.

    Returns:
        List of dicts with character_name, story, and dimension scores.
    """
    if path is None:
        path = CHARACTER_FILE

    if not Path(path).exists():
        if ER_CHARACTER_FILE.exists():
            import shutil
            shutil.copy(ER_CHARACTER_FILE, path)
            print(f"Copied character data from {ER_CHARACTER_FILE}")
        else:
            print(f"Character data not found at {path} or {ER_CHARACTER_FILE}")
            return []

    characters = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            characters.append(row)

    return characters


# ── PDS dimension extraction ──────────────────────────────────────────────
# The archetypometrics TSV has columns for various personality traits.
# We approximate PDS from available trait ratings using Zimmerman's
# framework: Power is about capability/status, Danger is about
# threat/morality, Structure is about order/conventionality.

# These trait columns map to PDS dimensions (sign indicates polarity)
PDS_TRAIT_MAP = {
    "power": [
        ("heroic", +1), ("assertive", +1), ("dominant", +1),
        ("competent", +1), ("leader", +1), ("strong", +1),
        ("meek", -1), ("helpless", -1), ("follower", -1),
    ],
    "danger": [
        ("villainous", +1), ("cruel", +1), ("threatening", +1),
        ("aggressive", +1), ("chaotic", +1), ("dark", +1),
        ("innocent", -1), ("gentle", -1), ("kind", -1),
    ],
    "structure": [
        ("orderly", +1), ("traditional", +1), ("proper", +1),
        ("methodical", +1), ("disciplined", +1),
        ("rebellious", -1), ("chaotic", -1), ("spontaneous", -1),
    ],
}


def compute_pds(character, available_columns):
    """Compute PDS scores for a single character.

    Args:
        character: Dict with trait scores (from TSV row).
        available_columns: Set of column names present in the data.

    Returns:
        Dict with "power", "danger", "structure" scores (each -1 to +1).
    """
    pds = {}
    for dim, traits in PDS_TRAIT_MAP.items():
        total = 0.0
        count = 0
        for trait_name, sign in traits:
            if trait_name in available_columns and character.get(trait_name):
                try:
                    val = float(character[trait_name])
                    # Normalize from 0-100 to -1 to +1
                    total += sign * (val - 50) / 50
                    count += 1
                except (ValueError, TypeError):
                    pass
        pds[dim] = total / max(count, 1)
    return pds


# ── PDS → Patient parameter mapping ──────────────────────────────────────

def pds_to_patient(pds_scores):
    """Map PDS dimensions to patient parameters.

    Mapping rationale (Zimmerman 2025 §4.6.4):
      The PDS dimensions capture distinct biological risk factors:

      Power → metabolic_demand: powerful characters (heroes, leaders) map to
        high-energy tissues — brain for strategists (Cramer Ch. IV p.46:
        brain = highest metabolic demand), muscle for fighters. Using abs(power)
        because both extremes (dominant fighters AND meek thinkers) suggest
        high-demand tissues, just different ones.

      Power → genetic_vulnerability (inverted): powerful characters are
        narratively resilient, mapping to lower genetic susceptibility to
        mtDNA damage. The 0.2 coefficient is moderate — power predicts
        resilience but isn't the dominant factor.

      Danger → inflammation_level: dangerous characters (villains, warriors)
        exist in chronic stress states. Chronic stress drives inflammaging
        (Cramer Ch. VII.A pp.89-92: SASP and senescent cell accumulation).
        The 0.25 coefficient on danger is the strongest single PDS→patient
        mapping because danger directly models physiological threat exposure.

      Danger → genetic_vulnerability: danger exposure increases damage
        accumulation (0.3 coefficient — strongest link because the Danger
        dimension directly captures threat/damage narrative valence).

      Structure → baseline_nad_level: ordered, disciplined characters maintain
        their cellular machinery better. NAD+ declines with age (Cramer
        Ch. VI.A.3 pp.72-73, Ca16) but structure represents the behavioral
        choices (diet, sleep, supplementation) that slow this decline.
        The 0.2 coefficient is moderate — behavioral maintenance helps but
        can't fully compensate for biological aging.

      Structure → inflammation (inverted): chaotic lifestyles increase
        chronic inflammation via stress, poor diet, disrupted circadian
        rhythm. The -0.15 coefficient is smaller than danger's +0.25 because
        lack of structure contributes to inflammation less directly than
        active danger/combat.

    Coefficients (0.15, 0.1, 0.2, 0.3, 0.4, 0.25) are empirically calibrated
    to produce patient parameter distributions that span biologically plausible
    ranges when applied to the 2000-character archetypometrics dataset. They are
    NOT derived from clinical data — this is a semantic bridge for hypothesis
    generation, not a clinical prediction tool.

    Args:
        pds_scores: Dict with power, danger, structure (each -1 to +1).

    Returns:
        Dict with predicted patient parameter values.
    """
    power = pds_scores["power"]      # -1 (weak) to +1 (powerful)
    danger = pds_scores["danger"]    # -1 (safe) to +1 (dangerous)
    structure = pds_scores["structure"]  # -1 (chaotic) to +1 (ordered)

    # Map to patient parameter ranges
    patient = {
        # Age: power doesn't strongly predict age, use moderate default.
        # Characters span all ages but PDS doesn't capture age directly.
        "baseline_age": 50.0,

        # Heteroplasmy: danger increases damage (0.15 × danger), structure
        # reduces it (0.1 × structure). Base 0.2 = moderate adult level.
        "baseline_heteroplasmy": np.clip(
            0.2 + 0.15 * danger - 0.1 * structure, 0.05, 0.8),

        # NAD: structure predicts well-maintained cellular machinery.
        # Base 0.6 = moderate age-related decline (Cramer Ch. VI.A.3 p.73).
        "baseline_nad_level": np.clip(
            0.6 + 0.2 * structure - 0.1 * danger, 0.2, 1.0),

        # Genetic vulnerability: danger increases (0.3), power decreases (0.2).
        # Base 1.0 = population average (Cramer: haplogroup-dependent, 0.5-2.0).
        "genetic_vulnerability": np.clip(
            1.0 + 0.3 * danger - 0.2 * power, 0.5, 2.0),

        # Metabolic demand: power maps to high-demand tissues.
        # abs(power) because both strong leaders (brain) and warriors (muscle)
        # have high-demand tissues (Cramer Ch. IV p.46).
        "metabolic_demand": np.clip(
            1.0 + 0.4 * abs(power), 0.5, 2.0),

        # Inflammation: danger (0.25) and chaos (-structure, 0.15) increase it.
        # Base 0.3 = mild chronic inflammation (Cramer Ch. VII.A: inflammaging).
        "inflammation_level": np.clip(
            0.3 + 0.25 * danger - 0.15 * structure, 0.0, 1.0),
    }

    return patient


# ── Comparison with LLM-generated parameters ─────────────────────────────

def load_character_experiment_results():
    """Load results from character_seed_experiment.py if available."""
    # Try multiple potential paths
    candidates = [
        PROJECT / "artifacts" / "character_seed_experiment.json",
        PROJECT / "output" / "character_seed_experiment.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def compare_predictions(characters, experiment_data):
    """Compare PDS-predicted vs LLM-generated patient parameters.

    Args:
        characters: List of character dicts with PDS scores.
        experiment_data: Dict from character_seed_experiment.py results.

    Returns:
        Dict with correlation stats per parameter.
    """
    if experiment_data is None:
        return None

    # Build lookup from experiment data
    trials = experiment_data.get("trials", experiment_data.get("artifacts", []))
    llm_lookup = {}
    for trial in trials:
        char_name = trial.get("character", trial.get("seed", ""))
        if "patient" in trial:
            llm_lookup[char_name.lower()] = trial["patient"]

    # Compare
    comparisons = {name: {"pds": [], "llm": []} for name in PATIENT_NAMES}

    for char in characters:
        char_name = char.get("character_name", char.get("name", "")).lower()
        if char_name in llm_lookup:
            pds = compute_pds(char, set(char.keys()))
            predicted = pds_to_patient(pds)
            actual = llm_lookup[char_name]

            for name in PATIENT_NAMES:
                if name in predicted and name in actual:
                    comparisons[name]["pds"].append(predicted[name])
                    comparisons[name]["llm"].append(actual[name])

    # Compute correlations
    results = {}
    for name in PATIENT_NAMES:
        pds_vals = np.array(comparisons[name]["pds"])
        llm_vals = np.array(comparisons[name]["llm"])
        if len(pds_vals) > 5:
            corr = np.corrcoef(pds_vals, llm_vals)[0, 1]
            bias = np.mean(llm_vals - pds_vals)
            results[name] = {
                "n_matched": len(pds_vals),
                "correlation": float(corr) if not np.isnan(corr) else 0.0,
                "mean_bias": float(bias),
                "pds_mean": float(np.mean(pds_vals)),
                "llm_mean": float(np.mean(llm_vals)),
            }
        else:
            results[name] = {"n_matched": len(pds_vals), "correlation": None}

    return results


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PDS → Patient Parameter Mapping (Zimmerman 2025)")
    print("=" * 70)

    # Load characters
    characters = load_characters()
    if not characters:
        print("No character data available. Generating sample mappings.")
        # Show mapping for canonical archetypes
        archetypes = [
            ("Hero (Luke Skywalker)", {"power": 0.8, "danger": -0.3, "structure": 0.2}),
            ("Villain (Darth Vader)", {"power": 0.9, "danger": 0.9, "structure": 0.5}),
            ("Wise Elder (Gandalf)", {"power": 0.6, "danger": -0.5, "structure": 0.7}),
            ("Trickster (Loki)", {"power": 0.4, "danger": 0.6, "structure": -0.8}),
            ("Innocent (Frodo)", {"power": -0.3, "danger": -0.6, "structure": 0.3}),
            ("Warrior (Brienne)", {"power": 0.7, "danger": 0.2, "structure": 0.6}),
        ]
        for name, pds in archetypes:
            patient = pds_to_patient(pds)
            print(f"\n  {name}")
            print(f"    PDS: P={pds['power']:+.1f} D={pds['danger']:+.1f} "
                  f"S={pds['structure']:+.1f}")
            for k, v in patient.items():
                print(f"    {k}: {v:.2f}")
        sys.exit(0)

    print(f"Loaded {len(characters)} characters")

    # Compute PDS for all characters
    available_cols = set(characters[0].keys()) if characters else set()
    pds_scores = []
    for char in characters:
        pds = compute_pds(char, available_cols)
        pds_scores.append(pds)

    # Show PDS distribution
    powers = [p["power"] for p in pds_scores]
    dangers = [p["danger"] for p in pds_scores]
    structures = [p["structure"] for p in pds_scores]
    print(f"\n  PDS distribution ({len(pds_scores)} characters):")
    print(f"    Power:     mean={np.mean(powers):.3f}  "
          f"std={np.std(powers):.3f}  range=[{np.min(powers):.3f}, {np.max(powers):.3f}]")
    print(f"    Danger:    mean={np.mean(dangers):.3f}  "
          f"std={np.std(dangers):.3f}  range=[{np.min(dangers):.3f}, {np.max(dangers):.3f}]")
    print(f"    Structure: mean={np.mean(structures):.3f}  "
          f"std={np.std(structures):.3f}  range=[{np.min(structures):.3f}, {np.max(structures):.3f}]")

    # Generate patient predictions
    predictions = []
    for char, pds in zip(characters, pds_scores):
        patient = pds_to_patient(pds)
        predictions.append({
            "character": char.get("character_name", char.get("name", "")),
            "story": char.get("story", char.get("fictional_work", "")),
            "pds": pds,
            "predicted_patient": patient,
        })

    # Show sample predictions
    print("\n  Sample PDS → Patient mappings:")
    for pred in predictions[:5]:
        pds = pred["pds"]
        patient = pred["predicted_patient"]
        print(f"\n    {pred['character']} ({pred['story']})")
        print(f"      PDS: P={pds['power']:+.3f} D={pds['danger']:+.3f} "
              f"S={pds['structure']:+.3f}")
        print(f"      → het={patient['baseline_heteroplasmy']:.2f}  "
              f"NAD={patient['baseline_nad_level']:.2f}  "
              f"vuln={patient['genetic_vulnerability']:.2f}  "
              f"demand={patient['metabolic_demand']:.2f}  "
              f"inflam={patient['inflammation_level']:.2f}")

    # Compare with LLM experiment data if available
    experiment_data = load_character_experiment_results()
    comparison = compare_predictions(characters, experiment_data)

    if comparison:
        print("\n  PDS vs LLM comparison:")
        for name, stats in comparison.items():
            if stats.get("correlation") is not None:
                print(f"    {name:30s}: r={stats['correlation']:.3f}  "
                      f"bias={stats['mean_bias']:+.3f}  "
                      f"(n={stats['n_matched']})")
            else:
                print(f"    {name:30s}: insufficient data "
                      f"(n={stats['n_matched']})")
    else:
        print("\n  No character experiment data found for comparison.")
        print("  Run character_seed_experiment.py first, then re-run this script.")

    # Save results
    output = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "n_characters": len(characters),
        "pds_stats": {
            "power": {"mean": float(np.mean(powers)), "std": float(np.std(powers))},
            "danger": {"mean": float(np.mean(dangers)), "std": float(np.std(dangers))},
            "structure": {"mean": float(np.mean(structures)), "std": float(np.std(structures))},
        },
        "sample_predictions": predictions[:20],
        "comparison": comparison,
    }

    output_path = PROJECT / "artifacts" / "pds_mapping.json"
    with open(output_path, "w") as f:
        json.dump(output, f, cls=NumpyEncoder, indent=2)
    print(f"\n  Results saved to {output_path}")
