#!/usr/bin/env python3
"""
character_seed_experiment.py

Translate fictional character identities into mitochondrial intervention
protocols via LLMs, then simulate and analyze.

Adapted from character_seed_experiment.py in the parent Evolutionary-Robotics
project. The LLM reasons about each character's personality, lifestyle, age,
and health trajectory to generate a 12D intervention + patient vector.

Pipeline per trial:
  1. Prompt LLM: "Design a mitochondrial protocol for [Character/Story]"
  2. Parse 12D intervention + patient vector
  3. Run ODE simulation (30 years, 3000 steps)
  4. Compute 4-pillar analytics
  5. Record everything

Scale: 2000 characters × 4 local Ollama models = 8000 trials
Estimated time: ~4-5 hours (overnight run)

Character data: uses archetypometrics_characters.tsv from the parent project.
If not found locally, copies from parent project.
"""

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    DEFAULT_INTERVENTION,
    HETEROPLASMY_CLIFF,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder
from llm_common import MODELS, query_ollama, split_vector


# ── Prompt template ─────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are a mitochondrial medicine specialist who sees patients through "
    "the lens of fictional archetypes.\n\n"
    "A patient presents who reminds you strongly of '{character}' from "
    "'{story}'. Think about this character's:\n"
    "  - Age and life stage\n"
    "  - Energy level and physical activity\n"
    "  - Stress level and inflammation (villains, warriors → high; "
    "peaceful characters → low)\n"
    "  - Resilience and genetic robustness\n"
    "  - Tissue vulnerability (intellectual characters → brain/high demand; "
    "physical characters → muscle; sedentary → low demand)\n\n"
    "Design a personalized mitochondrial intervention protocol AND "
    "characterize the patient based on this archetype.\n\n"
    "PARAMETERS (output a JSON object with ALL 12 keys):\n"
    "  Intervention (0.0-1.0 each):\n"
    "    rapamycin_dose: mTOR inhibition → enhanced mitophagy\n"
    "    nad_supplement: NAD+ precursor (NMN/NR)\n"
    "    senolytic_dose: Senescent cell clearance\n"
    "    yamanaka_intensity: Partial reprogramming (HIGH energy cost!)\n"
    "    transplant_rate: Mitochondrial transplant via mitlets\n"
    "    exercise_level: Hormetic exercise\n"
    "  Patient:\n"
    "    baseline_age (20-90), baseline_heteroplasmy (0.0-0.95),\n"
    "    baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0),\n"
    "    metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)\n\n"
    "In 1-2 sentences, describe why this character-archetype maps to "
    "this protocol. Then output ONLY the JSON object. Keep reasoning SHORT."
)


# ── Load characters ─────────────────────────────────────────────────────────

def load_characters():
    """Load fictional characters from archetypometrics TSV.

    Looks in local artifacts/ first, then tries parent project.
    """
    local_path = PROJECT / "artifacts" / "archetypometrics_characters.tsv"
    parent_path = (PROJECT.parent / "pybullet_test" / "Evolutionary-Robotics"
                   / "artifacts" / "archetypometrics_characters.tsv")

    if not local_path.exists() and parent_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_path, local_path)
        print(f"Copied character data from parent project")

    if not local_path.exists():
        print(f"ERROR: Character data not found at {local_path}")
        print(f"  Expected: archetypometrics_characters.tsv in artifacts/")
        sys.exit(1)

    characters = []
    with open(local_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            char_story = row.get("character/story", "")
            parts = char_story.split("/", 1)
            character = parts[0].strip() if parts else row.get("character", "")
            story = parts[1].strip() if len(parts) > 1 else "Unknown"
            characters.append({
                "index": int(row.get("index", 0)),
                "character": character,
                "story": story,
                "character_story": char_story,
                "card_url": row.get("card url", ""),
            })
    return characters


# ── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(path, results, metadata):
    completed_keys = {f"{r['character_story']}|{r['model']}" for r in results}
    with open(path, "w") as f:
        json.dump({
            "metadata": metadata,
            "completed_keys": list(completed_keys),
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    characters = load_characters()
    print(f"Loaded {len(characters)} characters from archetypometrics TSV")

    out_path = PROJECT / "artifacts" / "character_seed_experiment.json"
    checkpoint_path = PROJECT / "artifacts" / "character_seed_experiment_checkpoint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    results = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        completed_keys = set(ckpt.get("completed_keys", []))
        print(f"Resumed from checkpoint: {len(results)} trials already done")

    n_total = len(characters) * len(MODELS)
    n_remaining = n_total - len(completed_keys)
    print(f"Character Seed Experiment (mitochondrial): "
          f"{len(characters)} characters × {len(MODELS)} models = {n_total} trials")
    print(f"Remaining: {n_remaining} trials")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print()

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "character_seed_mitochondrial",
        "n_characters": len(characters),
        "n_models": len(MODELS),
        "models": [m["name"] for m in MODELS],
        "source": "archetypometrics_characters.tsv",
    }

    start_time = time.time()
    trial_num = len(completed_keys)
    failures = 0
    checkpoint_interval = 50

    for char_info in characters:
        character = char_info["character"]
        story = char_info["story"]
        char_story = char_info["character_story"]

        prompt = PROMPT_TEMPLATE.format(character=character, story=story)

        for model_info in MODELS:
            model_name = model_info["name"]
            key = f"{char_story}|{model_name}"

            if key in completed_keys:
                continue

            trial_num += 1
            print(f"[{trial_num}/{n_total}] {model_name} | {character} ({story})",
                  end=" ", flush=True)

            vector, raw_resp = query_ollama(model_name, prompt)

            if vector is None:
                failures += 1
                print("-> PARSE FAIL")
                results.append({
                    "character": character, "story": story,
                    "character_story": char_story,
                    "character_index": char_info["index"],
                    "model": model_name, "success": False,
                    "raw_response": raw_resp[:500] if raw_resp else "",
                    "intervention": None, "patient": None, "analytics": None,
                })
                completed_keys.add(key)
                continue

            intervention, patient = split_vector(vector)

            try:
                result = simulate(intervention=intervention, patient=patient)
                baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
                analytics = compute_all(result, baseline)
            except Exception as e:
                failures += 1
                print(f"-> SIM ERROR: {e}")
                results.append({
                    "character": character, "story": story,
                    "character_story": char_story,
                    "character_index": char_info["index"],
                    "model": model_name, "success": False,
                    "raw_response": raw_resp[:500] if raw_resp else "",
                    "intervention": intervention, "patient": patient,
                    "analytics": None,
                })
                completed_keys.add(key)
                continue

            final_atp = analytics["energy"]["atp_final"]
            final_het = analytics["damage"]["het_final"]
            atp_benefit = analytics["intervention"]["atp_benefit_terminal"]
            print(f"-> ATP={final_atp:.3f} het={final_het:.3f} benefit={atp_benefit:+.3f}")

            results.append({
                "character": character, "story": story,
                "character_story": char_story,
                "character_index": char_info["index"],
                "model": model_name, "success": True,
                "intervention": intervention, "patient": patient,
                "analytics": analytics,
            })
            completed_keys.add(key)

            if len(completed_keys) % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, results, metadata)
                elapsed = time.time() - start_time
                rate = (trial_num - (n_total - n_remaining)) / max(elapsed, 1)
                remaining_time = (n_total - len(completed_keys)) / max(rate, 0.01)
                print(f"  [checkpoint] {len(completed_keys)}/{n_total} done, "
                      f"{elapsed:.0f}s elapsed, ~{remaining_time/60:.0f}min remaining")

    total_elapsed = time.time() - start_time

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"CHARACTER SEED EXPERIMENT (MITOCHONDRIAL) COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f} hours)")
    print(f"Trials: {len(results)} ({failures} failures)")

    successes = [r for r in results if r["success"]]
    if successes:
        atps = [r["analytics"]["energy"]["atp_final"] for r in successes]
        hets = [r["analytics"]["damage"]["het_final"] for r in successes]
        benefits = [r["analytics"]["intervention"]["atp_benefit_terminal"] for r in successes]

        print(f"\nOverall:")
        print(f"  Median final ATP: {np.median(atps):.3f}")
        print(f"  Median final het: {np.median(hets):.3f}")
        print(f"  Median ATP benefit: {np.median(benefits):+.3f}")
        cliff_crossed = sum(1 for h in hets if h >= HETEROPLASMY_CLIFF)
        print(f"  Cliff crossed: {cliff_crossed}/{len(successes)} "
              f"({100*cliff_crossed/len(successes):.1f}%)")

        # Per-model
        print(f"\nPer-model:")
        for model_info in MODELS:
            mname = model_info["name"]
            m_results = [r for r in successes if r["model"] == mname]
            if m_results:
                m_atps = [r["analytics"]["energy"]["atp_final"] for r in m_results]
                m_benefits = [r["analytics"]["intervention"]["atp_benefit_terminal"]
                             for r in m_results]
                print(f"  {mname:20s}: {len(m_results):4d} trials, "
                      f"median ATP={np.median(m_atps):.3f}, "
                      f"median benefit={np.median(m_benefits):+.3f}")

        # Per-story summary (top 10 by median benefit)
        from collections import defaultdict
        story_benefits = defaultdict(list)
        for r in successes:
            story_benefits[r["story"]].append(
                r["analytics"]["intervention"]["atp_benefit_terminal"])

        print(f"\nTop 10 stories by median ATP benefit:")
        sorted_stories = sorted(story_benefits.items(),
                                key=lambda x: np.median(x[1]), reverse=True)
        for story, ben_list in sorted_stories[:10]:
            print(f"  {story:40s}: median benefit={np.median(ben_list):+.3f} "
                  f"({len(ben_list)} characters)")

        # Most effective characters
        from collections import defaultdict as dd
        char_best = dd(lambda: -999)
        char_info_map = {}
        for r in successes:
            benefit = r["analytics"]["intervention"]["atp_benefit_terminal"]
            if benefit > char_best[r["character_story"]]:
                char_best[r["character_story"]] = benefit
                char_info_map[r["character_story"]] = r

        sorted_chars = sorted(char_best.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 most effective character-protocols:")
        for cs, best_benefit in sorted_chars[:10]:
            r = char_info_map[cs]
            print(f"  {cs:50s}: benefit={best_benefit:+.3f} ({r['model']})")

        print(f"\nTop 10 most harmful character-protocols:")
        for cs, best_benefit in sorted_chars[-10:]:
            r = char_info_map[cs]
            print(f"  {cs:50s}: benefit={best_benefit:+.3f} ({r['model']})")

    # Save
    metadata["elapsed_seconds"] = total_elapsed
    metadata["n_results"] = len(results)
    metadata["n_failures"] = failures

    with open(out_path, "w") as f:
        json.dump({"metadata": metadata, "results": results},
                  f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint removed")


if __name__ == "__main__":
    run_experiment()
