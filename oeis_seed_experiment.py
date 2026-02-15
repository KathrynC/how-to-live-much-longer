#!/usr/bin/env python3
"""
oeis_seed_experiment.py

Use notable integer sequences from OEIS as semantic seeds for the
LLM → intervention vector → ODE simulation pipeline.

Adapted from oeis_seed_experiment.py in the parent Evolutionary-Robotics
project. Each sequence is presented with its ID, name, and first 16 terms.
The LLM translates the mathematical "energy" of the sequence into a 12D
intervention + patient vector for the mitochondrial aging model.

Pipeline per trial:
  1. Fetch sequence from OEIS (cached locally)
  2. Prompt LLM with sequence ID, name, description, first 16 terms
  3. Parse 12D intervention + patient vector from LLM response
  4. Run mitochondrial ODE simulation (30 years, 3000 steps)
  5. Compute 4-pillar analytics
  6. Record everything

Scale: ~99 sequences × 4 local Ollama models = ~396 trials
"""

import json
import subprocess
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
from llm_common import (
    MODELS, query_ollama, split_vector,
)


# ── Curated OEIS sequences ─────────────────────────────────────────────────
# Same library as parent project, organized by mathematical flavor.

SEQUENCES = {
    # Growth & Accumulation
    "A000027": "The positive integers",
    "A000079": "Powers of 2",
    "A000142": "Factorial numbers",
    "A000290": "The squares",
    "A000578": "The cubes",
    "A000217": "Triangular numbers",
    "A000292": "Tetrahedral numbers",
    "A000326": "Pentagonal numbers",
    # Fibonacci & Golden Ratio
    "A000045": "Fibonacci numbers",
    "A000032": "Lucas numbers",
    "A001622": "Decimal expansion of golden ratio",
    "A000931": "Padovan sequence",
    "A001608": "Perrin sequence",
    "A000073": "Tribonacci numbers",
    # Primes & Divisibility
    "A000040": "The prime numbers",
    "A001358": "Semiprimes",
    "A000961": "Prime powers",
    "A002808": "Composite numbers",
    "A000005": "d(n) — number of divisors",
    "A000010": "Euler totient function",
    "A000203": "sigma(n) — sum of divisors",
    "A008683": "Moebius function",
    "A000720": "pi(n) — prime counting function",
    "A001223": "Prime gaps",
    "A000668": "Mersenne primes",
    # Combinatorial
    "A000108": "Catalan numbers",
    "A000110": "Bell numbers",
    "A000670": "Fubini numbers (ordered Bell)",
    "A001006": "Motzkin numbers",
    "A000984": "Central binomial coefficients",
    "A000041": "Partitions of n",
    "A000129": "Pell numbers",
    # Recursive & Self-referential
    "A005132": "Recamán's sequence",
    "A006577": "Steps to reach 1 in Collatz (3n+1)",
    "A003215": "Hex (centered hexagonal) numbers",
    "A007318": "Pascal's triangle read by rows",
    "A000120": "Binary weight of n",
    "A000002": "Kolakoski sequence",
    "A001462": "Golomb's sequence",
    # Chaotic & Pseudorandom
    "A000796": "Decimal expansion of Pi",
    "A001113": "Decimal expansion of e",
    "A002193": "Decimal expansion of sqrt(2)",
    "A000583": "Fourth powers",
    "A001511": "2-ruler sequence",
    "A005408": "The odd numbers",
    "A005843": "The even numbers",
    # Oscillating & Periodic
    "A000035": "Period 2: 0,1,0,1,...",
    "A000034": "Period 2: 1,2,1,2,...",
    "A011655": "Period 3: 0,1,2,0,1,2,...",
    "A010060": "Thue-Morse sequence",
    "A001285": "Thue-Morse (1,2 version)",
    "A014577": "Regular paper-folding (dragon curve)",
    "A005614": "Fibonacci word (binary)",
    # Sparse & Explosive
    "A000244": "Powers of 3",
    "A000400": "Powers of 6",
    "A001146": "2^(2^n)",
    "A007953": "Digital sum of n",
    "A055642": "Number of digits of n",
    # Number Theory Exotica
    "A000169": "n^(n-1)",
    "A001333": "Numerators of convergents to sqrt(2)",
    "A000225": "2^n - 1 (Mersenne numbers)",
    "A000051": "2^n + 1",
    "A000396": "Perfect numbers",
    "A005100": "Deficient numbers",
    "A005101": "Abundant numbers",
    # Geometry & Space
    "A000124": "Lazy caterer's sequence",
    "A000127": "Regions of a circle",
    "A006003": "Centered octahedral numbers",
    "A000330": "Sum of squares",
    "A000537": "Sum of cubes",
    "A002378": "Oblong (pronic) numbers",
    # Music & Signal
    "A005187": "a(n) = a(floor(n/2)) + n",
    "A000201": "Beatty sequence for sqrt(2)",
    "A001950": "Upper Wythoff sequence",
    # Sequences about sequences
    "A007947": "Largest squarefree factor of n",
    "A001221": "omega(n) — number of prime factors",
    "A001222": "Omega(n) — with multiplicity",
    "A003418": "lcm(1,...,n)",
    "A000793": "Landau's function",
    # Famous Constants
    "A007376": "Digits of Champernowne constant",
    "A010815": "Related to Jacobi theta",
    # Deceptively simple
    "A000012": "The all 1's sequence",
    "A000004": "The all 0's sequence",
    "A000007": "Characteristic function of 0",
    "A005117": "Squarefree numbers",
}

# Deduplicate
SEQUENCES = dict(sorted(set(SEQUENCES.items())))


# ── OEIS fetch ──────────────────────────────────────────────────────────────

def fetch_oeis_sequence(seq_id, cache_dir):
    """Fetch a sequence from OEIS, caching locally."""
    cache_file = cache_dir / f"{seq_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"https://oeis.org/search?q=id:{seq_id}&fmt=json"
    try:
        r = subprocess.run(
            ["curl", "-s", url],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        if not data or not isinstance(data, dict):
            # OEIS returns {"results": [...]} or {"count": 0, ...}
            results = data.get("results", [])
            if not results:
                return None
            seq = results[0]
        elif isinstance(data, list):
            if not data:
                return None
            seq = data[0]
        else:
            seq = data
        with open(cache_file, "w") as f:
            json.dump(seq, f, indent=2)
        return seq
    except Exception as e:
        print(f"  OEIS fetch error for {seq_id}: {e}")
        return None


def get_first_terms(seq_data, n=16):
    """Extract first n terms from OEIS data string."""
    terms_str = seq_data.get("data", "")
    terms = []
    for t in terms_str.split(","):
        t = t.strip()
        if t == "":
            continue
        try:
            terms.append(int(t))
        except ValueError:
            break
        if len(terms) >= n:
            break
    return terms


# ── LLM prompt ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are a mitochondrial medicine specialist who sees mathematical patterns "
    "as metaphors for biological processes.\n\n"
    "I will show you a mathematical integer sequence. Translate its character "
    "— its growth rate, rhythm, regularity or chaos — into a personalized "
    "intervention protocol for a mitochondrial aging patient.\n\n"
    "Sequence: {seq_id} — {seq_name}\n"
    "First 16 terms: {terms}\n\n"
    "Mapping intuition:\n"
    "  - Explosive growth (factorials, powers) → aggressive intervention\n"
    "  - Steady rhythm (Fibonacci, triangular) → balanced, moderate protocol\n"
    "  - Periodic/oscillating → pulsed or cycled treatment\n"
    "  - Chaotic/irregular → experimental combinations\n"
    "  - Constant/minimal → minimal intervention (watchful waiting)\n"
    "  - Prime-like (sparse, unpredictable) → targeted single-agent\n\n"
    "PARAMETERS (output a JSON object with ALL 12 keys):\n"
    "  Intervention (0.0-1.0 each):\n"
    "    rapamycin_dose, nad_supplement, senolytic_dose,\n"
    "    yamanaka_intensity, transplant_rate, exercise_level\n"
    "  Patient (derive from the sequence's character):\n"
    "    baseline_age (20-90), baseline_heteroplasmy (0.0-0.95),\n"
    "    baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0),\n"
    "    metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)\n\n"
    "Values are snapped to grids: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] for doses.\n\n"
    "In 1-2 sentences, describe the treatment philosophy this sequence inspires. "
    "Then output ONLY the JSON object. Keep reasoning SHORT."
)


# ── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(path, results, metadata):
    completed_keys = {f"{r['seq_id']}|{r['model']}" for r in results}
    with open(path, "w") as f:
        json.dump({
            "metadata": metadata,
            "completed_keys": list(completed_keys),
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    cache_dir = PROJECT / "artifacts" / "oeis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = PROJECT / "artifacts" / "oeis_seed_experiment.json"
    checkpoint_path = PROJECT / "artifacts" / "oeis_seed_experiment_checkpoint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch all sequences from OEIS
    print(f"Fetching {len(SEQUENCES)} sequences from OEIS...")
    seq_data = {}
    for seq_id in sorted(SEQUENCES.keys()):
        data = fetch_oeis_sequence(seq_id, cache_dir)
        if data:
            terms = get_first_terms(data)
            if len(terms) >= 6:
                seq_data[seq_id] = {
                    "id": seq_id,
                    "name": data.get("name", SEQUENCES[seq_id]),
                    "terms": terms,
                    "description": SEQUENCES[seq_id],
                }
            else:
                print(f"  {seq_id}: too few terms ({len(terms)}), skipping")
        else:
            print(f"  {seq_id}: fetch failed, skipping")
        time.sleep(0.3)  # Be polite to OEIS

    print(f"Fetched {len(seq_data)} sequences successfully")

    # Load checkpoint
    results = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        completed_keys = set(ckpt.get("completed_keys", []))
        print(f"Resumed from checkpoint: {len(results)} trials already done")

    n_total = len(seq_data) * len(MODELS)
    n_remaining = n_total - len(completed_keys)
    print(f"\nOEIS Seed Experiment (mitochondrial): "
          f"{len(seq_data)} sequences × {len(MODELS)} models = {n_total} trials")
    print(f"Remaining: {n_remaining} trials")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print()

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "oeis_seed_mitochondrial",
        "n_sequences": len(seq_data),
        "n_models": len(MODELS),
        "models": [m["name"] for m in MODELS],
    }

    start_time = time.time()
    trial_num = len(completed_keys)
    failures = 0
    checkpoint_interval = 50

    for seq_id in sorted(seq_data.keys()):
        seq = seq_data[seq_id]
        terms_str = ", ".join(str(t) for t in seq["terms"][:16])

        prompt = PROMPT_TEMPLATE.format(
            seq_id=seq_id,
            seq_name=seq["name"],
            terms=terms_str,
        )

        for model_info in MODELS:
            model_name = model_info["name"]
            key = f"{seq_id}|{model_name}"

            if key in completed_keys:
                continue

            trial_num += 1
            print(f"[{trial_num}/{n_total}] {model_name} | {seq_id} {seq['name'][:50]}",
                  end=" ", flush=True)

            vector, raw_resp = query_ollama(model_name, prompt)

            if vector is None:
                failures += 1
                print("-> PARSE FAIL")
                results.append({
                    "seq_id": seq_id, "seq_name": seq["name"],
                    "seq_terms": seq["terms"][:16],
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
                    "seq_id": seq_id, "seq_name": seq["name"],
                    "seq_terms": seq["terms"][:16],
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
                "seq_id": seq_id, "seq_name": seq["name"],
                "seq_terms": seq["terms"][:16],
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
    print(f"OEIS SEED EXPERIMENT (MITOCHONDRIAL) COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
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

        # Top 10 most effective sequences
        from collections import defaultdict
        seq_best = defaultdict(lambda: -999)
        seq_info = {}
        for r in successes:
            benefit = r["analytics"]["intervention"]["atp_benefit_terminal"]
            if benefit > seq_best[r["seq_id"]]:
                seq_best[r["seq_id"]] = benefit
                seq_info[r["seq_id"]] = r

        print(f"\nTop 10 most effective sequences (by ATP benefit):")
        sorted_seqs = sorted(seq_best.items(), key=lambda x: x[1], reverse=True)
        for sid, best_benefit in sorted_seqs[:10]:
            r = seq_info[sid]
            print(f"  {sid} {r['seq_name'][:40]:40s}: benefit={best_benefit:+.3f} ({r['model']})")

        # Top 10 most harmful
        print(f"\nTop 10 most harmful sequences:")
        for sid, best_benefit in sorted_seqs[-10:]:
            r = seq_info[sid]
            print(f"  {sid} {r['seq_name'][:40]:40s}: benefit={best_benefit:+.3f} ({r['model']})")

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
