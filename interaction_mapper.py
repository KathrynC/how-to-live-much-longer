#!/usr/bin/env python3
"""
interaction_mapper.py — D4: Intervention Interaction Mapper

Discover synergistic and antagonistic intervention combinations by running
2D grid sweeps of all 15 intervention pairs (C(6,2)) across 3 patient types.

Biological motivation (Cramer 2025):
    Mitochondrial interventions act through distinct biochemical pathways
    (Ch. VI-VIII), so combining them may produce super-additive effects:
      - Rapamycin upregulates mitophagy via PINK1/Parkin (Ch. VI.B p.75)
      - NAD+ precursors restore cofactor levels gated by CD38 (Ch. VI.A.3 p.73)
      - Senolytics remove senescent cells that produce SASP (Ch. VII.A pp.89-92)
      - Transplant adds healthy mtDNA copies + displaces damaged (Ch. VIII.G pp.104-107)
      - Exercise triggers hormetic ROS signaling (Ch. VI.B p.76)
      - Yamanaka partial reprogramming at enormous ATP cost (Ch. VIII.A Table 3 p.100)

    However, pathway overlap creates potential antagonisms:
      - Yamanaka + exercise both consume ATP → energy crisis if patient is weak
      - High rapamycin + transplant: autophagy may clear transplanted mitochondria
      - NAD + exercise: exercise-induced ROS may deplete NAD faster

    The synergy measure (Bliss independence model) quantifies whether combined
    fitness exceeds the sum of individual effects. This is standard in
    pharmacological combination screening (Bliss 1939, Loewe 1953).

Synergy = actual_fitness(A,B) - (fitness_A_alone + fitness_B_alone)
    Positive → super-additive (synergy: pathways complement each other)
    Negative → sub-additive (antagonism: pathways compete for same resource)

Discovery potential:
  - Whether rapamycin + transplant synergize (complementary: clear damaged + add healthy)
  - Whether yamanaka + exercise antagonize (both cost ATP; Cramer Ch. VIII.A Table 3)
  - Whether synergy patterns reverse between young and near-cliff patients
    (nonlinear cliff dynamics mean interactions change qualitatively near het=0.7)
  - Whether CD38 nonlinearity creates unexpected NAD interaction effects
    (low NAD dose mostly destroyed by CD38; Cramer Ch. VI.A.3 p.73)

Scale: 15 pairs × 6×6 grid × 3 patients = ~1620 sims + ~540 singles = ~2160 total
Estimated time: ~3 minutes

Reference:
    Cramer, J.G. (2025). "How to Live Much Longer: The Mitochondrial DNA
    Connection." ISBN 979-8-9928220-0-4.
"""

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from constants import (
    DEFAULT_INTERVENTION,
    INTERVENTION_PARAMS,
    INTERVENTION_NAMES,
)
from simulator import simulate
from analytics import NumpyEncoder

PROJECT = Path(__file__).resolve().parent

# ── Patient profiles ─────────────────────────────────────────────────────────
# Three patient archetypes spanning the disease trajectory:
#   - young_25: Pre-cliff, low damage, NAD still high — tests whether early
#     intervention combos provide preventive synergy
#   - moderate_70: Post-AGE_TRANSITION (65; Cramer Appendix 2 p.155), accelerated
#     deletion rate, moderate damage — the typical "aging patient" scenario
#   - near_cliff_80: Heteroplasmy 0.65, dangerously close to the 0.70 cliff
#     (Cramer Ch. V.K p.66) — tests whether combos can rescue near-collapse
#
# The key question is whether synergy patterns REVERSE between patients:
# combinations that help young patients may harm near-cliff patients (and vice
# versa) because the sigmoid cliff dynamics change the landscape qualitatively.

PATIENTS = {
    "young_25": {
        "label": "Young prevention (25yo, 10% het)",
        "params": {
            "baseline_age": 25.0, "baseline_heteroplasmy": 0.10,
            "baseline_nad_level": 0.9, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.1,
        },
    },
    "moderate_70": {
        "label": "Moderate aging (70yo, 30% het)",
        "params": {
            "baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
            "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.25,
        },
    },
    "near_cliff_80": {
        "label": "Near-cliff (80yo, 65% het)",
        "params": {
            "baseline_age": 80.0, "baseline_heteroplasmy": 0.65,
            "baseline_nad_level": 0.4, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.5,
        },
    },
}

# Dose grid matching INTERVENTION_PARAMS in constants.py — the 6-level grid
# spans no-treatment (0.0) through maximum clinical dose (1.0).
# Intermediate levels (0.1, 0.25, 0.5, 0.75) are log-spaced to better resolve
# the nonlinear dose-response curves typical of biological systems.
DOSE_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


# ── Fitness function ─────────────────────────────────────────────────────────

def compute_fitness(result, baseline):
    """Scalar fitness combining ATP preservation and heteroplasmy reduction.

    Fitness = atp_benefit + 0.5 * het_benefit

    The 0.5 weight on het_benefit reflects that ATP is the proximate cause of
    cellular dysfunction (Cramer Ch. V.K p.66: "energy crisis"), while
    heteroplasmy is the upstream driver. Weighting ATP higher ensures we
    favor interventions that maintain energy output NOW, even if heteroplasmy
    reduction is slower. This matches the fitness function in
    llm_seeded_evolution.py for consistency across the analysis pipeline.

    Args:
        result: Simulation result dict from simulate().
        baseline: No-treatment simulation result for same patient.

    Returns:
        Float fitness score (higher = better).
    """
    final_atp = float(result["states"][-1, 2])
    base_atp = float(baseline["states"][-1, 2])
    final_het = float(result["heteroplasmy"][-1])
    base_het = float(baseline["heteroplasmy"][-1])
    atp_benefit = final_atp - base_atp
    het_benefit = base_het - final_het  # positive = improvement (less damage)
    return atp_benefit + 0.5 * het_benefit


# ── Single-parameter sweep ───────────────────────────────────────────────────

def evaluate_single(param, dose, patient, baseline):
    """Evaluate a single intervention parameter at a given dose, others at zero.

    This isolates the marginal effect of one intervention, providing the
    "individual contribution" baseline needed for Bliss-independence synergy
    calculation: synergy = f(A,B) - [f(A) + f(B)].

    Args:
        param: Intervention parameter name (e.g., "rapamycin_dose").
        dose: Dose level (0.0–1.0).
        patient: Patient parameter dict.
        baseline: No-treatment simulation result.

    Returns:
        Float fitness score for single-param intervention.
    """
    intervention = dict(DEFAULT_INTERVENTION)
    intervention[param] = dose
    result = simulate(intervention=intervention, patient=patient)
    return compute_fitness(result, baseline)


# ── Pair evaluation ──────────────────────────────────────────────────────────

def evaluate_pair(param_a, dose_a, param_b, dose_b, patient, baseline):
    """Evaluate two intervention parameters applied simultaneously.

    The combined simulation captures all ODE-mediated interactions between
    the two pathways (e.g., NAD restoration improving mitophagy efficiency,
    or Yamanaka competing with exercise for ATP reserves).

    Args:
        param_a: First intervention parameter name.
        dose_a: First parameter dose (0.0–1.0).
        param_b: Second intervention parameter name.
        dose_b: Second parameter dose (0.0–1.0).
        patient: Patient parameter dict.
        baseline: No-treatment simulation result.

    Returns:
        Float fitness score for the combination.
    """
    intervention = dict(DEFAULT_INTERVENTION)
    intervention[param_a] = dose_a
    intervention[param_b] = dose_b
    result = simulate(intervention=intervention, patient=patient)
    return compute_fitness(result, baseline)


# ── Synergy computation ─────────────────────────────────────────────────────

def compute_synergy(actual, a_alone, b_alone):
    """Bliss-independence synergy: actual - (a_alone + b_alone).

    Positive values indicate the interventions complement each other through
    distinct biological mechanisms (e.g., rapamycin clears damaged mitos via
    PINK1/Parkin while transplant adds fresh copies — Cramer Ch. VI.B + VIII.G).

    Negative values indicate antagonism: pathways compete for a shared resource
    (e.g., both Yamanaka and exercise consume ATP — Ch. VIII.A Table 3).

    Args:
        actual: Fitness of combined intervention.
        a_alone: Fitness of first intervention alone.
        b_alone: Fitness of second intervention alone.

    Returns:
        Float synergy score (positive = synergistic, negative = antagonistic).
    """
    return actual - (a_alone + b_alone)


# ── Synergy matrix builder ──────────────────────────────────────────────────

def build_synergy_matrix(pair_results):
    """Build 6x6 symmetric matrix of max synergies per intervention pair.

    Returns:
        (matrix, param_names) — 6x6 numpy array + ordered param names.
    """
    n = len(INTERVENTION_NAMES)
    matrix = np.zeros((n, n))

    for entry in pair_results:
        i = INTERVENTION_NAMES.index(entry["param_a"])
        j = INTERVENTION_NAMES.index(entry["param_b"])
        matrix[i, j] = entry["max_synergy"]
        matrix[j, i] = entry["max_synergy"]

    return matrix, INTERVENTION_NAMES


# ── Sobol comparison ────────────────────────────────────────────────────────

def compare_to_sobol(matrix, sobol_path):
    """Rank-correlate synergy matrix with Sobol ST-S1 interaction indices.

    Sobol total-order (ST) minus first-order (S1) captures the fraction of
    variance attributable to parameter interactions — a global measure from
    variance decomposition. This synergy mapper provides a complementary
    LOCAL measure (super-additivity at specific dose combinations).

    If the two rankings agree (high Spearman rho), it confirms that the
    interaction structure is robust across methodologies. If they disagree,
    it suggests the synergy landscape is dose-dependent in ways that
    global sensitivity analysis cannot resolve.

    Args:
        matrix: 6x6 synergy matrix (max synergy per pair).
        sobol_path: Path to sobol_sensitivity.json artifact.

    Returns:
        Dict with spearman_rho and per-parameter rank comparison, or None
        if Sobol data is unavailable.
    """
    sobol_file = Path(sobol_path)
    if not sobol_file.exists():
        return None

    with open(sobol_file) as f:
        sobol = json.load(f)

    # Extract interaction indices (ST - S1) for intervention params only
    het_interact = sobol.get("heteroplasmy", {}).get("interaction", {})
    sobol_values = []
    synergy_values = []

    for i, name_i in enumerate(INTERVENTION_NAMES):
        sobol_val = het_interact.get(name_i, 0.0)
        # Sum of absolute synergies for this parameter (row sum of matrix)
        synergy_sum = float(np.sum(np.abs(matrix[i, :])))
        sobol_values.append(sobol_val)
        synergy_values.append(synergy_sum)

    # Spearman rank correlation (numpy-only)
    n = len(sobol_values)
    if n < 3:
        return None

    def _rank(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        return ranks

    rank_sobol = _rank(np.array(sobol_values))
    rank_synergy = _rank(np.array(synergy_values))
    d = rank_sobol - rank_synergy
    rho = 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))

    return {
        "spearman_rho": float(rho),
        "sobol_interaction_ranks": {name: float(rank_sobol[i])
                                     for i, name in enumerate(INTERVENTION_NAMES)},
        "synergy_sum_ranks": {name: float(rank_synergy[i])
                               for i, name in enumerate(INTERVENTION_NAMES)},
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    out_path = PROJECT / "artifacts" / "interaction_mapper.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    sim_count = 0

    pairs = list(combinations(INTERVENTION_NAMES, 2))
    n_pairs = len(pairs)
    n_grid = len(DOSE_GRID)

    print("=" * 70)
    print("D4: INTERACTION MAPPER — Synergy/Antagonism Discovery")
    print("=" * 70)
    print(f"Intervention pairs: {n_pairs}")
    print(f"Dose grid: {DOSE_GRID} ({n_grid} levels)")
    print(f"Patients: {list(PATIENTS.keys())}")
    print(f"Estimated sims: ~{n_pairs * n_grid * n_grid * len(PATIENTS) + len(INTERVENTION_NAMES) * n_grid * len(PATIENTS)}")
    print()

    all_patient_results = {}

    for pat_id, pat_info in PATIENTS.items():
        patient = pat_info["params"]
        print(f"\n--- {pat_info['label']} ---")

        # Compute baseline
        baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
        sim_count += 1

        # Pre-compute single-parameter fitness for all params × all doses
        print("  Computing single-parameter sweeps...", end="", flush=True)
        single_fitness = {}  # (param, dose) → fitness
        for param in INTERVENTION_NAMES:
            for dose in DOSE_GRID:
                if dose == 0.0:
                    single_fitness[(param, dose)] = 0.0
                else:
                    single_fitness[(param, dose)] = evaluate_single(
                        param, dose, patient, baseline)
                    sim_count += 1
        print(f" done ({len(INTERVENTION_NAMES) * (n_grid - 1)} sims)")

        # Sweep all pairs
        pair_results = []

        for pair_idx, (param_a, param_b) in enumerate(pairs):
            print(f"  [{pair_idx+1}/{n_pairs}] {param_a} × {param_b}", end="", flush=True)

            grid_data = []
            max_synergy = -999.0
            max_synergy_doses = (0.0, 0.0)
            min_synergy = 999.0
            min_synergy_doses = (0.0, 0.0)

            for dose_a in DOSE_GRID:
                for dose_b in DOSE_GRID:
                    if dose_a == 0.0 and dose_b == 0.0:
                        actual = 0.0
                    elif dose_a == 0.0:
                        actual = single_fitness[(param_b, dose_b)]
                    elif dose_b == 0.0:
                        actual = single_fitness[(param_a, dose_a)]
                    else:
                        actual = evaluate_pair(
                            param_a, dose_a, param_b, dose_b, patient, baseline)
                        sim_count += 1

                    a_alone = single_fitness[(param_a, dose_a)]
                    b_alone = single_fitness[(param_b, dose_b)]
                    synergy = compute_synergy(actual, a_alone, b_alone)

                    grid_data.append({
                        "dose_a": dose_a,
                        "dose_b": dose_b,
                        "fitness_combined": round(actual, 6),
                        "fitness_a_alone": round(a_alone, 6),
                        "fitness_b_alone": round(b_alone, 6),
                        "synergy": round(synergy, 6),
                    })

                    if synergy > max_synergy:
                        max_synergy = synergy
                        max_synergy_doses = (dose_a, dose_b)
                    if synergy < min_synergy:
                        min_synergy = synergy
                        min_synergy_doses = (dose_a, dose_b)

            pair_entry = {
                "param_a": param_a,
                "param_b": param_b,
                "max_synergy": round(max_synergy, 6),
                "max_synergy_doses": list(max_synergy_doses),
                "min_synergy": round(min_synergy, 6),
                "min_synergy_doses": list(min_synergy_doses),
                "grid": grid_data,
            }
            pair_results.append(pair_entry)

            label = "SYNERGY" if max_synergy > 0.01 else ("ANTAGONISM" if min_synergy < -0.01 else "neutral")
            print(f"  max={max_synergy:+.4f} min={min_synergy:+.4f} [{label}]")

        # Build synergy matrix
        matrix, _ = build_synergy_matrix(pair_results)

        # Rankings
        ranked_synergies = sorted(pair_results, key=lambda p: p["max_synergy"], reverse=True)
        ranked_antagonisms = sorted(pair_results, key=lambda p: p["min_synergy"])

        all_patient_results[pat_id] = {
            "label": pat_info["label"],
            "pair_results": pair_results,
            "synergy_matrix": matrix.tolist(),
            "top_synergies": [
                {"pair": [p["param_a"], p["param_b"]],
                 "synergy": p["max_synergy"],
                 "doses": p["max_synergy_doses"]}
                for p in ranked_synergies[:5]
            ],
            "top_antagonisms": [
                {"pair": [p["param_a"], p["param_b"]],
                 "antagonism": p["min_synergy"],
                 "doses": p["min_synergy_doses"]}
                for p in ranked_antagonisms[:5]
            ],
        }

    elapsed = time.time() - start_time

    # ── Cross-patient analysis ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"INTERACTION MAPPER COMPLETE — {elapsed:.1f}s ({sim_count} sims)")
    print(f"{'=' * 70}")

    # Do synergy patterns reverse between patients?
    print("\nTop 3 synergies per patient:")
    for pat_id, data in all_patient_results.items():
        print(f"  {pat_id}:")
        for s in data["top_synergies"][:3]:
            print(f"    {s['pair'][0]} × {s['pair'][1]}: {s['synergy']:+.4f} "
                  f"at doses {s['doses']}")

    print("\nTop 3 antagonisms per patient:")
    for pat_id, data in all_patient_results.items():
        print(f"  {pat_id}:")
        for s in data["top_antagonisms"][:3]:
            print(f"    {s['pair'][0]} × {s['pair'][1]}: {s['antagonism']:+.4f} "
                  f"at doses {s['doses']}")

    # Sobol comparison
    sobol_path = PROJECT / "artifacts" / "sobol_sensitivity.json"
    sobol_comparison = {}
    for pat_id, data in all_patient_results.items():
        matrix = np.array(data["synergy_matrix"])
        comp = compare_to_sobol(matrix, sobol_path)
        if comp:
            sobol_comparison[pat_id] = comp
            print(f"\n  Sobol vs synergy rank correlation ({pat_id}): "
                  f"rho={comp['spearman_rho']:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "interaction_mapper",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_sims": sim_count,
        "dose_grid": DOSE_GRID,
        "intervention_names": INTERVENTION_NAMES,
        "patients": {pid: pinfo["label"] for pid, pinfo in PATIENTS.items()},
        "results": all_patient_results,
        "sobol_comparison": sobol_comparison if sobol_comparison else None,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
