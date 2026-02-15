#!/usr/bin/env python3
"""
competing_evaluators.py — D5: Competing Evaluators

Find protocols robust across 4 competing clinical criteria — the TI
"transaction" that resonates across all "absorbers."

TIQM analogy (Cramer 2025, The Quantum Handshake):
    In the Transactional Interpretation, a quantum transaction only completes
    when the offer wave resonates with ALL absorbers. Here, each evaluator
    represents a different clinical "absorber" (stakeholder perspective).
    A "transaction" protocol must resonate with all 4 simultaneously — it is
    the rare intervention that satisfies every clinical priority at once.

Evaluators map to distinct clinical priorities from Cramer's model:
  1. ATP_Guardian  — "Keep the lights on": energy preservation is the proximate
     concern (Cramer Ch. V.K p.66: ATP collapse = cellular death)
  2. Het_Hunter    — "Fix the root cause": aggressively reduce heteroplasmy,
     the upstream driver of all downstream pathology (Cramer Ch. II-III)
  3. Crisis_Delayer — "Buy time": maximize years before the heteroplasmy cliff
     triggers irreversible energy collapse (Cramer Ch. V.K)
  4. Efficiency_Auditor — "First, do no harm": minimize intervention burden
     per unit benefit, reflecting real clinical constraints on dosing

A "transaction" protocol is one in top-25% for ALL 4 evaluators.
Robustness score = harmonic mean of percentile ranks (penalizes any
single weak dimension, matching the weakest-link nature of clinical success).

Discovery potential:
  - Whether "universal good" protocols exist or goals inherently conflict
  - Which evaluator pairs agree/disagree most (fundamental bio trade-offs)
  - How rare transactions are (if <1%, biology forces hard choices)
  - Whether efficiency and effectiveness correlate or anti-correlate

Scale: ~500 candidates × 1 patient + baselines = ~1000 sims
Estimated time: ~2 minutes

Reference:
    Cramer, J.G. (2025). "How to Live Much Longer: The Mitochondrial DNA
    Connection." ISBN 979-8-9928220-0-4.
"""

import json
import time
from pathlib import Path

import numpy as np

from constants import (
    DEFAULT_INTERVENTION,
    INTERVENTION_PARAMS,
    INTERVENTION_NAMES,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder

PROJECT = Path(__file__).resolve().parent

# ── Test patient ─────────────────────────────────────────────────────────────

TEST_PATIENT = {
    "baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
    "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0, "inflammation_level": 0.25,
}


# ── Evaluator functions ─────────────────────────────────────────────────────

def atp_guardian(analytics):
    """Maximize energy preservation: the proximate clinical objective.

    Score = atp_benefit_mean - 0.5 * energy_cost_per_year

    The 0.5 penalty on energy cost reflects that some interventions (especially
    Yamanaka at 3-5 MU/day; Cramer Ch. VIII.A Table 3 p.100) consume significant
    ATP themselves. A protocol that boosts ATP by 0.3 but costs 0.6/year in energy
    is net harmful. The 0.5 weight balances "ATP gained" vs "ATP spent" — set
    below 1.0 because intervention costs are temporary but benefits compound.

    Args:
        analytics: Dict from compute_all() with energy/damage/intervention keys.

    Returns:
        Float score (higher = better energy preservation).
    """
    intv = analytics["intervention"]
    return intv["atp_benefit_mean"] - 0.5 * intv["energy_cost_per_year"]


def het_hunter(analytics):
    """Minimize heteroplasmy aggressively: the upstream biological goal.

    Score = het_benefit_terminal - 0.3 * frac_above_cliff

    Rewards heteroplasmy reduction (the root cause of aging per Cramer Ch. II-III).
    Penalizes protocols that leave significant time above the cliff (het > 0.70;
    Cramer Ch. V.K p.66), even if terminal het looks good — because time spent
    above the cliff causes irreversible damage (bistability; fix C4 in simulator).

    The 0.3 penalty weight is moderate because frac_above_cliff is a fraction
    (0-1) while het_benefit is typically ±0.1-0.3, so 0.3 makes them comparable.

    Args:
        analytics: Dict from compute_all().

    Returns:
        Float score (higher = more damage reversal).
    """
    intv = analytics["intervention"]
    dmg = analytics["damage"]
    return intv["het_benefit_terminal"] - 0.3 * dmg["frac_above_cliff"]


def crisis_delayer(analytics):
    """Maximize time until energy crisis: the clinical timeline objective.

    Score = crisis_delay/30 + 0.3 * time_to_crisis/30

    crisis_delay = how many additional years the intervention buys before crisis
    time_to_crisis = absolute years until ATP drops below survival threshold

    Both normalized to [0,1] via /30 (the simulation horizon). The 0.3 weight
    on absolute time reflects that delaying crisis matters more than the absolute
    timeline — a patient at 80yo with crisis at 85 benefits more from +5 years
    than one at 25yo with crisis at 55 benefits from knowing their timeline.

    Caps at 30 years to prevent infinite-horizon artifacts.

    Args:
        analytics: Dict from compute_all().

    Returns:
        Float score (higher = more time bought).
    """
    intv = analytics["intervention"]
    eng = analytics["energy"]
    crisis_delay = min(intv["crisis_delay_years"], 30.0)
    time_to_crisis = min(eng["time_to_crisis_years"], 30.0)
    return crisis_delay / 30.0 + 0.3 * time_to_crisis / 30.0


def efficiency_auditor(analytics):
    """Maximize benefit per unit cost: the clinical feasibility objective.

    Score = benefit_cost_ratio/10 - 0.2 * total_dose/6

    Rewards high benefit-cost ratio (how much health improvement per unit of
    intervention intensity). Penalizes heavy total dosing, reflecting real-world
    constraints: polypharmacy risks, patient compliance, treatment burden.

    benefit_cost_ratio capped at 10 to prevent outlier dominance (a single
    highly efficient low-dose protocol shouldn't overwhelm all others).
    total_dose normalized by 6 (max possible: 6 params × 1.0 each).
    The 0.2 weight makes the dose penalty a tiebreaker, not the primary driver.

    Args:
        analytics: Dict from compute_all().

    Returns:
        Float score (higher = more efficient protocol).
    """
    intv = analytics["intervention"]
    bcr = min(intv["benefit_cost_ratio"], 10.0)
    return bcr / 10.0 - 0.2 * intv["total_dose"] / 6.0


EVALUATORS = {
    "ATP_Guardian": atp_guardian,
    "Het_Hunter": het_hunter,
    "Crisis_Delayer": crisis_delayer,
    "Efficiency_Auditor": efficiency_auditor,
}


# ── Candidate generation ────────────────────────────────────────────────────

def _latin_hypercube(n, d, rng):
    """Pure numpy LHS in [0,1]^d."""
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        for i in range(n):
            lo = perm[i] / n
            hi = (perm[i] + 1) / n
            result[i, j] = lo + rng.random() * (hi - lo)
    return result


def generate_candidates(n_random=300, pareto_path=None, synergy_path=None, rng=None):
    """Generate candidate intervention pool from multiple sources.

    Sources:
      - n_random: LHS random samples
      - pareto_path: Pareto frontier from reachable_set.py (D1)
      - synergy_path: Top synergies from interaction_mapper.py (D4)
      - Named protocols: known cocktails from existing tests

    Returns:
        List of (intervention_dict, source_label) tuples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    candidates = []

    # Source 1: Random LHS
    samples = _latin_hypercube(n_random, len(INTERVENTION_NAMES), rng)
    for row in samples:
        intv = {}
        for j, name in enumerate(INTERVENTION_NAMES):
            lo, hi = INTERVENTION_PARAMS[name]["range"]
            intv[name] = float(lo + row[j] * (hi - lo))
        candidates.append((intv, "random_lhs"))

    # Source 2: Pareto frontier (if available)
    if pareto_path:
        ppath = Path(pareto_path)
        if ppath.exists():
            with open(ppath) as f:
                pdata = json.load(f)
            for pat_data in pdata.get("results", {}).values():
                for point in pat_data.get("pareto_frontier", []):
                    if point.get("intervention"):
                        candidates.append((point["intervention"], "pareto_d1"))

    # Source 3: Top synergies (if available)
    if synergy_path:
        spath = Path(synergy_path)
        if spath.exists():
            with open(spath) as f:
                sdata = json.load(f)
            for pat_data in sdata.get("results", {}).values():
                for syn in pat_data.get("top_synergies", [])[:3]:
                    # Build intervention from synergy pair at optimal doses
                    intv = dict(DEFAULT_INTERVENTION)
                    pair = syn.get("pair", [])
                    doses = syn.get("doses", [0.5, 0.5])
                    if len(pair) == 2 and len(doses) == 2:
                        intv[pair[0]] = doses[0]
                        intv[pair[1]] = doses[1]
                        candidates.append((intv, "synergy_d4"))

    # Source 4: Named protocols
    named_protocols = [
        ("cocktail_balanced", {
            "rapamycin_dose": 0.5, "nad_supplement": 0.5,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5,
        }),
        ("rapamycin_heavy", {
            "rapamycin_dose": 0.75, "nad_supplement": 0.25,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.25,
        }),
        ("transplant_focused", {
            "rapamycin_dose": 0.1, "nad_supplement": 0.25,
            "senolytic_dose": 0.1, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.75, "exercise_level": 0.1,
        }),
        ("nad_heavy", {
            "rapamycin_dose": 0.25, "nad_supplement": 0.75,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.25,
        }),
        ("exercise_only", {
            "rapamycin_dose": 0.0, "nad_supplement": 0.0,
            "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 1.0,
        }),
        ("kitchen_sink", {
            "rapamycin_dose": 0.5, "nad_supplement": 0.5,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.25,
            "transplant_rate": 0.5, "exercise_level": 0.5,
        }),
        ("transplant_plus_rapa", {
            "rapamycin_dose": 0.5, "nad_supplement": 0.25,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.75, "exercise_level": 0.25,
        }),
        ("minimal_effective", {
            "rapamycin_dose": 0.25, "nad_supplement": 0.25,
            "senolytic_dose": 0.1, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5,
        }),
        ("aggressive_reversal", {
            "rapamycin_dose": 0.75, "nad_supplement": 0.75,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 1.0, "exercise_level": 0.5,
        }),
        ("yamanaka_cautious", {
            "rapamycin_dose": 0.25, "nad_supplement": 0.5,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.25,
            "transplant_rate": 0.0, "exercise_level": 0.25,
        }),
    ]
    for name, intv in named_protocols:
        candidates.append((intv, f"named:{name}"))

    return candidates


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_candidate(intervention, patient, baseline):
    """Simulate and compute all 4 evaluator scores.

    Returns:
        (analytics_dict, scores_dict) or None on simulation failure.
    """
    result = simulate(intervention=intervention, patient=patient)
    analytics = compute_all(result, baseline)

    scores = {}
    for name, func in EVALUATORS.items():
        scores[name] = func(analytics)

    return analytics, scores


def compute_percentile_ranks(all_scores):
    """Compute per-evaluator percentile ranking for all candidates.

    Args:
        all_scores: List of score dicts.

    Returns:
        List of percentile rank dicts (0-100, higher = better).
    """
    n = len(all_scores)
    ranks = [{} for _ in range(n)]

    for eval_name in EVALUATORS:
        values = np.array([s[eval_name] for s in all_scores])
        # Rank: higher value → higher percentile
        order = np.argsort(values)
        percentiles = np.zeros(n)
        percentiles[order] = np.linspace(0, 100, n)
        for i in range(n):
            ranks[i][eval_name] = float(percentiles[i])

    return ranks


def find_transactions(ranks, threshold=75):
    """Find protocols in top-25% for ALL evaluators.

    Returns:
        List of indices meeting the threshold on all evaluators.
    """
    transactions = []
    for i, rank in enumerate(ranks):
        if all(v >= threshold for v in rank.values()):
            transactions.append(i)
    return transactions


def compute_pareto_4d(scores_list):
    """Compute 4D non-dominated set via iterative dominance checking.

    Returns:
        List of indices on the 4D Pareto frontier.
    """
    n = len(scores_list)
    eval_names = list(EVALUATORS.keys())
    matrix = np.array([[s[e] for e in eval_names] for s in scores_list])

    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if (np.all(matrix[j] >= matrix[i]) and
                    np.any(matrix[j] > matrix[i])):
                is_dominated[i] = True
                break

    return list(np.where(~is_dominated)[0])


def agreement_matrix(ranks):
    """Compute 4x4 evaluator rank correlation matrix.

    Returns:
        (matrix, eval_names) — Spearman rank correlation between evaluators.
    """
    eval_names = list(EVALUATORS.keys())
    n_eval = len(eval_names)
    n = len(ranks)

    # Extract rank arrays per evaluator
    rank_arrays = {}
    for eval_name in eval_names:
        rank_arrays[eval_name] = np.array([r[eval_name] for r in ranks])

    matrix = np.zeros((n_eval, n_eval))
    for i, name_i in enumerate(eval_names):
        for j, name_j in enumerate(eval_names):
            if i == j:
                matrix[i, j] = 1.0
            else:
                # Pearson correlation on percentile ranks ≈ Spearman
                ri = rank_arrays[name_i]
                rj = rank_arrays[name_j]
                if np.std(ri) > 1e-9 and np.std(rj) > 1e-9:
                    matrix[i, j] = float(np.corrcoef(ri, rj)[0, 1])

    return matrix, eval_names


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(n_random=300, rng_seed=42):
    out_path = PROJECT / "artifacts" / "competing_evaluators.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(rng_seed)
    start_time = time.time()

    print("=" * 70)
    print("D5: COMPETING EVALUATORS — Multi-Criteria Protocol Search")
    print("=" * 70)
    print(f"Evaluators: {list(EVALUATORS.keys())}")
    print(f"Random candidates: {n_random}")
    print()

    # Generate candidates
    pareto_path = PROJECT / "artifacts" / "reachable_set.json"
    synergy_path = PROJECT / "artifacts" / "interaction_mapper.json"
    candidates = generate_candidates(
        n_random=n_random,
        pareto_path=str(pareto_path),
        synergy_path=str(synergy_path),
        rng=rng,
    )

    # Count sources
    from collections import Counter
    source_counts = Counter(src for _, src in candidates)
    print(f"Candidates: {len(candidates)} total")
    for src, cnt in source_counts.most_common():
        print(f"  {src}: {cnt}")

    # Compute baseline
    baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=TEST_PATIENT)

    # Score all candidates
    print(f"\nScoring {len(candidates)} candidates...")
    all_scores = []
    all_analytics = []
    all_interventions = []
    all_sources = []

    for i, (intv, source) in enumerate(candidates):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(candidates)}] ({elapsed:.0f}s)")

        analytics, scores = score_candidate(intv, TEST_PATIENT, baseline)
        all_scores.append(scores)
        all_analytics.append(analytics)
        all_interventions.append(intv)
        all_sources.append(source)

    sim_count = len(candidates) + 1  # +1 for baseline

    # Compute percentile ranks
    ranks = compute_percentile_ranks(all_scores)

    # Find transactions
    transactions = find_transactions(ranks, threshold=75)

    # Robustness scores (harmonic mean of percentile ranks)
    robustness = []
    for rank in ranks:
        values = list(rank.values())
        if all(v > 0 for v in values):
            hm = len(values) / sum(1.0 / v for v in values)
        else:
            hm = 0.0
        robustness.append(hm)

    # 4D Pareto frontier
    pareto_4d_idx = compute_pareto_4d(all_scores)

    # Agreement matrix
    agree_matrix, eval_names = agreement_matrix(ranks)

    elapsed = time.time() - start_time

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"COMPETING EVALUATORS COMPLETE — {elapsed:.1f}s ({sim_count} sims)")
    print(f"{'=' * 70}")

    print(f"\nTransactions (top-25% on ALL evaluators): "
          f"{len(transactions)}/{len(candidates)} "
          f"({100*len(transactions)/len(candidates):.1f}%)")

    if transactions:
        # Sort by robustness
        sorted_trans = sorted(transactions, key=lambda i: robustness[i], reverse=True)
        print("\nTop 5 transaction protocols:")
        for rank, idx in enumerate(sorted_trans[:5]):
            intv = all_interventions[idx]
            print(f"  #{rank+1} robustness={robustness[idx]:.1f} "
                  f"[{all_sources[idx]}]")
            print(f"    rapa={intv['rapamycin_dose']:.2f} "
                  f"nad={intv['nad_supplement']:.2f} "
                  f"seno={intv['senolytic_dose']:.2f} "
                  f"yama={intv['yamanaka_intensity']:.2f} "
                  f"trans={intv['transplant_rate']:.2f} "
                  f"ex={intv['exercise_level']:.2f}")
            print(f"    scores: " + "  ".join(
                f"{k}={all_scores[idx][k]:.3f}" for k in eval_names))

    print(f"\n4D Pareto frontier: {len(pareto_4d_idx)} protocols")

    print(f"\nEvaluator agreement matrix (Spearman rho):")
    print(f"  {'':20s}", end="")
    for name in eval_names:
        print(f"  {name[:8]:>8s}", end="")
    print()
    for i, name_i in enumerate(eval_names):
        print(f"  {name_i:20s}", end="")
        for j in range(len(eval_names)):
            print(f"  {agree_matrix[i,j]:8.3f}", end="")
        print()

    # Per-evaluator top 3
    print("\nPer-evaluator top 3:")
    for eval_name in eval_names:
        sorted_idx = sorted(range(len(all_scores)),
                            key=lambda i: all_scores[i][eval_name],
                            reverse=True)
        print(f"  {eval_name}:")
        for rank, idx in enumerate(sorted_idx[:3]):
            print(f"    #{rank+1} score={all_scores[idx][eval_name]:.3f} "
                  f"[{all_sources[idx]}] "
                  f"dose={all_interventions[idx].get('transplant_rate', 0):.2f}t "
                  f"rapa={all_interventions[idx].get('rapamycin_dose', 0):.2f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    # Build transaction details
    transaction_details = []
    if transactions:
        sorted_trans = sorted(transactions, key=lambda i: robustness[i], reverse=True)
        for idx in sorted_trans:
            transaction_details.append({
                "intervention": all_interventions[idx],
                "source": all_sources[idx],
                "scores": all_scores[idx],
                "percentile_ranks": ranks[idx],
                "robustness": robustness[idx],
            })

    # Build 4D Pareto details
    pareto_details = []
    for idx in pareto_4d_idx:
        pareto_details.append({
            "intervention": all_interventions[idx],
            "source": all_sources[idx],
            "scores": all_scores[idx],
        })

    output = {
        "experiment": "competing_evaluators",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_sims": sim_count,
        "n_candidates": len(candidates),
        "test_patient": TEST_PATIENT,
        "evaluator_names": eval_names,
        "source_counts": dict(source_counts),
        "transactions": {
            "count": len(transactions),
            "fraction": len(transactions) / len(candidates),
            "protocols": transaction_details,
        },
        "pareto_4d": {
            "count": len(pareto_4d_idx),
            "protocols": pareto_details,
        },
        "agreement_matrix": agree_matrix.tolist(),
        "per_evaluator_rankings": {
            eval_name: {
                "top_3": [
                    {"intervention": all_interventions[idx],
                     "score": all_scores[idx][eval_name],
                     "source": all_sources[idx]}
                    for idx in sorted(range(len(all_scores)),
                                      key=lambda i: all_scores[i][eval_name],
                                      reverse=True)[:3]
                ],
                "bottom_3": [
                    {"intervention": all_interventions[idx],
                     "score": all_scores[idx][eval_name],
                     "source": all_sources[idx]}
                    for idx in sorted(range(len(all_scores)),
                                      key=lambda i: all_scores[i][eval_name])[:3]
                ],
            }
            for eval_name in eval_names
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
