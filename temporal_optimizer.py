#!/usr/bin/env python3
"""
temporal_optimizer.py — D2: Temporal Protocol Optimizer

Discover optimal intervention timelines — not just what to give, but when.

Biological motivation (Cramer 2025):
    Mitochondrial aging is highly nonlinear in time. Two key transitions
    create opportunities for temporal optimization:

    1. AGE_TRANSITION at 65 (Cramer Appendix 2 p.155, Va23 data): the mtDNA
       deletion doubling time drops from 11.8yr to 3.06yr, accelerating damage
       accumulation 3.9×. Interventions may need to intensify around this age.

    2. The heteroplasmy cliff at ~0.70 (Cramer Ch. V.K p.66): once crossed,
       bistability (fix C4) makes return difficult. Interventions timed BEFORE
       crossing are far more effective than identical protocols applied after.

    These nonlinearities mean that a time-varying protocol (e.g., "aggressive
    early, taper late" or "escalate before the cliff") can outperform a
    constant-dose protocol with the same total drug exposure.

Uses a (1+lambda) evolutionary strategy over InterventionSchedule space.
Representation: N_PHASES phases = (N_PHASES-1) boundary years + N_PHASES × 6
intervention params = 20D genotype (for 3 phases).

Mutation: 50% chance mutate a boundary (Gaussian, sigma=2yr, clamp+sort),
50% chance mutate one intervention param in one phase (±1 grid step).

Compares optimal phased schedule to equivalent constant protocol to
quantify "timing importance" — how much temporal structure matters.

Discovery potential:
  - Whether front-loading or delayed onset is optimal
  - Whether optimal timing differs across patient types
  - The timing importance score (temporal structure vs constant dosing)
  - Whether pulsed patterns emerge naturally
  - Phase transitions corresponding to AGE_TRANSITION=65

Scale: 3 patients × (50 gen × 10 lambda + 1 constant) = ~1530 sims + extras ≈ ~3000
Estimated time: ~7 minutes

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
    SIM_YEARS,
)
from simulator import simulate, InterventionSchedule
from analytics import compute_all, NumpyEncoder

PROJECT = Path(__file__).resolve().parent

# ── Configuration ───────────────────────────────────────────────────────────

N_PHASES = 3           # 3 phases allows "early/middle/late" temporal structure,
                       # sufficient to capture AGE_TRANSITION effects without
                       # overfitting the 30-year trajectory
N_GENERATIONS = 50     # (1+lambda) ES generations — typically converges by gen 30-40
LAMBDA = 10            # children per generation — 10 provides adequate mutation
                       # sampling for the 20D genotype (standard for (1+lambda) ES)
BOUNDARY_SIGMA = 2.0   # mutation sigma for phase boundaries (years).
                       # 2.0 years is ~7% of the 30yr horizon — large enough to
                       # explore different temporal structures but small enough
                       # for fine-grained boundary placement once a good region
                       # is found. Chosen to roughly match the timescale of
                       # deletion doubling (3.06yr post-transition; Cramer App.2)

# ── Patient profiles ────────────────────────────────────────────────────────

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


# ── Genotype encoding ──────────────────────────────────────────────────────

def encode_schedule(boundaries, phase_interventions):
    """Encode schedule as flat genotype array.

    Args:
        boundaries: List of N_PHASES-1 boundary years.
        phase_interventions: List of N_PHASES intervention dicts.

    Returns:
        np.array of length (N_PHASES-1) + N_PHASES*6 = 20 (for 3 phases).
    """
    n_intv = len(INTERVENTION_NAMES)
    genotype = np.zeros(len(boundaries) + len(phase_interventions) * n_intv)

    # Boundaries
    for i, b in enumerate(boundaries):
        genotype[i] = b

    # Intervention params per phase
    offset = len(boundaries)
    for p, intv in enumerate(phase_interventions):
        for j, name in enumerate(INTERVENTION_NAMES):
            genotype[offset + p * n_intv + j] = intv.get(name, 0.0)

    return genotype


def decode_schedule(genotype, n_phases):
    """Decode flat genotype to boundaries + intervention dicts.

    Returns:
        (boundaries, phase_interventions)
    """
    n_intv = len(INTERVENTION_NAMES)
    n_boundaries = n_phases - 1

    boundaries = sorted(genotype[:n_boundaries].tolist())

    phase_interventions = []
    offset = n_boundaries
    for p in range(n_phases):
        intv = {}
        for j, name in enumerate(INTERVENTION_NAMES):
            intv[name] = float(np.clip(genotype[offset + p * n_intv + j], 0.0, 1.0))
        phase_interventions.append(intv)

    return boundaries, phase_interventions


def genotype_to_schedule(genotype, n_phases):
    """Build InterventionSchedule from genotype.

    Returns:
        InterventionSchedule instance.
    """
    boundaries, phase_interventions = decode_schedule(genotype, n_phases)

    # Build phase list: [(0, intv0), (boundary1, intv1), ...]
    phases = [(0.0, phase_interventions[0])]
    for i, b in enumerate(boundaries):
        phases.append((b, phase_interventions[i + 1]))

    return InterventionSchedule(phases)


# ── Mutation ─────────────────────────────────────────────────────────────────

def mutate_genotype(genotype, n_phases, rng):
    """Mutate genotype: 50% boundary, 50% param.

    Boundary mutation: Gaussian perturbation, clamp to [1, SIM_YEARS-1], re-sort.
    Param mutation: ±1 grid step for a random param in a random phase.

    Returns:
        New genotype (copy).
    """
    child = genotype.copy()
    n_intv = len(INTERVENTION_NAMES)
    n_boundaries = n_phases - 1

    if rng.random() < 0.5 and n_boundaries > 0:
        # Mutate a boundary
        b_idx = rng.integers(0, n_boundaries)
        child[b_idx] += rng.normal(0, BOUNDARY_SIGMA)
        child[b_idx] = np.clip(child[b_idx], 1.0, SIM_YEARS - 1.0)
        # Re-sort boundaries
        child[:n_boundaries] = np.sort(child[:n_boundaries])
    else:
        # Mutate an intervention param
        phase = rng.integers(0, n_phases)
        param_idx = rng.integers(0, n_intv)
        param_name = INTERVENTION_NAMES[param_idx]
        grid = sorted(INTERVENTION_PARAMS[param_name]["grid"])

        offset = n_boundaries + phase * n_intv + param_idx
        current_val = child[offset]

        # Find nearest grid point
        nearest = min(grid, key=lambda g: abs(g - current_val))
        grid_idx = grid.index(nearest)

        # Step ±1
        if grid_idx == 0:
            new_grid_idx = 1
        elif grid_idx == len(grid) - 1:
            new_grid_idx = len(grid) - 2
        else:
            new_grid_idx = grid_idx + rng.choice([-1, 1])

        child[offset] = grid[new_grid_idx]

    return child


# ── Fitness function ─────────────────────────────────────────────────────────

def evaluate_fitness(genotype, n_phases, patient, baseline):
    """Simulate schedule, compute analytics, return weighted scalar.

    Fitness = atp_benefit_mean + 0.5 * het_benefit_terminal
              + 0.3 * (crisis_delay/30) - 0.2 * energy_cost

    Weight rationale (all weights relative to atp_benefit_mean = 1.0):
      - 0.5 for het_benefit: heteroplasmy reduction is important but secondary
        to ATP (the proximate clinical outcome; Cramer Ch. V.K p.66)
      - 0.3 for crisis_delay: time-buying is valuable but less than direct
        health improvement; normalized by 30yr to make units comparable
      - -0.2 for energy_cost: penalizes metabolically expensive protocols
        (especially Yamanaka at 3-5 MU/day; Cramer Ch. VIII.A Table 3) but
        as a secondary consideration — we're optimizing outcomes, not minimizing
        treatment burden

    These weights match the fitness function in llm_seeded_evolution.py and
    the D5 evaluator composite for cross-tool consistency.

    Args:
        genotype: Flat numpy array encoding phased intervention schedule.
        n_phases: Number of temporal phases.
        patient: Patient parameter dict.
        baseline: No-treatment simulation result.

    Returns:
        (fitness, analytics) tuple.
    """
    schedule = genotype_to_schedule(genotype, n_phases)
    result = simulate(intervention=schedule, patient=patient)
    analytics = compute_all(result, baseline)

    intv = analytics["intervention"]
    crisis_delay = min(intv["crisis_delay_years"], 30.0)

    fitness = (intv["atp_benefit_mean"]
               + 0.5 * intv["het_benefit_terminal"]
               + 0.3 * (crisis_delay / 30.0)
               - 0.2 * intv["energy_cost_per_year"])

    return fitness, analytics


# ── (1+lambda) Evolution Strategy ───────────────────────────────────────────

def evolve(patient, n_phases=N_PHASES, n_gen=N_GENERATIONS, lam=LAMBDA,
           rng_seed=42):
    """Run (1+lambda) ES over phased intervention schedules.

    Returns:
        Dict with best genotype, fitness trajectory, analytics.
    """
    rng = np.random.default_rng(rng_seed)
    n_intv = len(INTERVENTION_NAMES)

    # Compute baseline
    baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)

    # Initialize: random genotype
    n_boundaries = n_phases - 1
    genotype_len = n_boundaries + n_phases * n_intv

    # Start with evenly-spaced boundaries and moderate doses
    boundaries = np.linspace(SIM_YEARS / n_phases, SIM_YEARS * (n_phases - 1) / n_phases, n_boundaries)
    parent = np.zeros(genotype_len)
    parent[:n_boundaries] = boundaries

    # Initialize each phase with random grid values
    for p in range(n_phases):
        for j, name in enumerate(INTERVENTION_NAMES):
            grid = INTERVENTION_PARAMS[name]["grid"]
            parent[n_boundaries + p * n_intv + j] = float(rng.choice(grid))

    parent_fit, parent_analytics = evaluate_fitness(parent, n_phases, patient, baseline)
    trajectory = [parent_fit]
    sim_count = 1

    for gen in range(n_gen):
        best_child = None
        best_child_fit = -999.0

        for _ in range(lam):
            child = mutate_genotype(parent, n_phases, rng)
            child_fit, child_analytics = evaluate_fitness(child, n_phases, patient, baseline)
            sim_count += 1

            if child_fit > best_child_fit:
                best_child = child
                best_child_fit = child_fit
                best_analytics = child_analytics

        # (1+lambda): keep better of parent and best child
        if best_child_fit > parent_fit:
            parent = best_child
            parent_fit = best_child_fit
            parent_analytics = best_analytics

        trajectory.append(parent_fit)

        if (gen + 1) % 10 == 0:
            print(f"    gen {gen+1}/{n_gen}: fitness={parent_fit:.4f} ({sim_count} sims)")

    return {
        "genotype": parent,
        "fitness": parent_fit,
        "trajectory": trajectory,
        "analytics": parent_analytics,
        "sim_count": sim_count,
        "baseline": baseline,
    }


# ── Constant protocol comparison ────────────────────────────────────────────

def compare_to_constant(best_genotype, n_phases, patient, baseline):
    """Flatten phased schedule to average constant protocol.

    Returns timing_importance = (phased_fitness - constant_fitness) / phased_fitness.
    """
    boundaries, phase_interventions = decode_schedule(best_genotype, n_phases)

    # Compute phase durations
    all_boundaries = [0.0] + boundaries + [SIM_YEARS]
    durations = [all_boundaries[i + 1] - all_boundaries[i]
                 for i in range(len(all_boundaries) - 1)]
    total = sum(durations)

    # Weighted average of interventions
    avg_intv = {}
    for name in INTERVENTION_NAMES:
        weighted_sum = sum(
            phase_interventions[p].get(name, 0.0) * durations[p]
            for p in range(n_phases)
        )
        avg_intv[name] = float(np.clip(weighted_sum / total, 0.0, 1.0))

    # Evaluate constant
    result_const = simulate(intervention=avg_intv, patient=patient)
    analytics_const = compute_all(result_const, baseline)
    intv_metrics = analytics_const["intervention"]
    crisis_delay = min(intv_metrics["crisis_delay_years"], 30.0)

    constant_fitness = (intv_metrics["atp_benefit_mean"]
                        + 0.5 * intv_metrics["het_benefit_terminal"]
                        + 0.3 * (crisis_delay / 30.0)
                        - 0.2 * intv_metrics["energy_cost_per_year"])

    return {
        "constant_intervention": avg_intv,
        "constant_fitness": constant_fitness,
        "constant_analytics": analytics_const,
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    out_path = PROJECT / "artifacts" / "temporal_optimizer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_sims = 0

    print("=" * 70)
    print("D2: TEMPORAL PROTOCOL OPTIMIZER — Optimal Intervention Timelines")
    print("=" * 70)
    print(f"Phases: {N_PHASES}")
    print(f"Generations: {N_GENERATIONS}, Lambda: {LAMBDA}")
    print(f"Patients: {list(PATIENTS.keys())}")
    print(f"Estimated sims: ~{len(PATIENTS) * (N_GENERATIONS * LAMBDA + 1 + 1)}")
    print()

    all_results = {}

    for pat_id, pat_info in PATIENTS.items():
        patient = pat_info["params"]
        print(f"\n--- {pat_info['label']} ---")

        # Evolve
        evo_result = evolve(patient, n_phases=N_PHASES,
                            rng_seed=42 + hash(pat_id) % 1000)
        total_sims += evo_result["sim_count"]

        # Decode best
        boundaries, phase_interventions = decode_schedule(
            evo_result["genotype"], N_PHASES)

        print(f"  Best fitness: {evo_result['fitness']:.4f}")
        print(f"  Phase boundaries: {[f'{b:.1f}yr' for b in boundaries]}")
        for p, intv in enumerate(phase_interventions):
            phase_start = 0.0 if p == 0 else boundaries[p - 1]
            phase_end = SIM_YEARS if p == N_PHASES - 1 else boundaries[p]
            top_params = sorted(intv.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_params)
            print(f"    Phase {p+1} (yr {phase_start:.1f}-{phase_end:.1f}): {top_str}")

        # Compare to constant
        const_result = compare_to_constant(
            evo_result["genotype"], N_PHASES, patient, evo_result["baseline"])
        total_sims += 1

        phased_fit = evo_result["fitness"]
        const_fit = const_result["constant_fitness"]
        if abs(phased_fit) > 1e-6:
            timing_importance = (phased_fit - const_fit) / abs(phased_fit)
        else:
            timing_importance = 0.0

        print(f"  Constant equivalent: fitness={const_fit:.4f}")
        print(f"  Timing importance: {timing_importance:.1%}")

        # Characterize pattern
        pattern = _characterize_pattern(phase_interventions)
        print(f"  Pattern: {pattern}")

        all_results[pat_id] = {
            "label": pat_info["label"],
            "best_fitness": evo_result["fitness"],
            "phase_boundaries": boundaries,
            "phase_interventions": phase_interventions,
            "fitness_trajectory": evo_result["trajectory"],
            "constant_comparison": {
                "constant_intervention": const_result["constant_intervention"],
                "constant_fitness": const_fit,
                "timing_importance": timing_importance,
            },
            "pattern": pattern,
            "analytics": {
                "energy": evo_result["analytics"]["energy"],
                "damage": evo_result["analytics"]["damage"],
                "intervention": evo_result["analytics"]["intervention"],
            },
        }

    elapsed = time.time() - start_time

    # ── Cross-patient analysis ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"TEMPORAL OPTIMIZER COMPLETE — {elapsed:.1f}s ({total_sims} sims)")
    print(f"{'=' * 70}")

    print("\nCross-patient summary:")
    print(f"  {'Patient':20s}  {'Phased':>8s}  {'Constant':>8s}  {'Timing':>10s}  {'Pattern':>20s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*20}")
    for pat_id, data in all_results.items():
        print(f"  {pat_id:20s}  "
              f"{data['best_fitness']:8.4f}  "
              f"{data['constant_comparison']['constant_fitness']:8.4f}  "
              f"{data['constant_comparison']['timing_importance']:9.1%}  "
              f"{data['pattern']:>20s}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "temporal_optimizer",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_sims": total_sims,
        "n_phases": N_PHASES,
        "n_generations": N_GENERATIONS,
        "lambda": LAMBDA,
        "intervention_names": INTERVENTION_NAMES,
        "patients": {pid: pinfo["label"] for pid, pinfo in PATIENTS.items()},
        "results": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


def _characterize_pattern(phase_interventions):
    """Characterize temporal pattern of a phased schedule.

    Biologically meaningful patterns:
      - "front-loaded": aggressive early, taper late — addresses damage before
        it compounds past the cliff (preventive strategy)
      - "escalating": ramp up over time — matches accelerating damage rate
        post-AGE_TRANSITION (65; Cramer Appendix 2 p.155)
      - "pulsed": high-low-high — may reflect drug holidays or cycling
      - "delayed-onset": low-high-low — waits for damage to accumulate before
        intervening (reactive strategy)
      - "constant": minimal temporal variation (<0.3 total dose difference)

    Threshold rationale:
      - 0.3 for "constant" detection: total dose sum ranges 0-6, so 0.3 is
        ~5% of max — below clinical significance for temporal variation
      - 0.5 for front-loaded/escalating: a meaningful shift in treatment
        intensity (~8% of max total dose) between phases

    Returns one of: "front-loaded", "delayed-onset", "tapered",
    "escalating", "pulsed", "constant", "complex".
    """
    n = len(phase_interventions)
    if n < 2:
        return "constant"

    # Compute total dose per phase (sum of all 6 intervention params)
    doses = []
    for intv in phase_interventions:
        doses.append(sum(intv.values()))

    if max(doses) - min(doses) < 0.3:
        return "constant"
    if doses[0] > doses[-1] + 0.5:
        return "front-loaded"
    if doses[-1] > doses[0] + 0.5:
        return "escalating"
    if n >= 3 and doses[0] < doses[1] and doses[1] > doses[2]:
        return "pulsed"
    if n >= 3 and doses[0] > doses[1] and doses[1] < doses[2]:
        return "delayed-onset"
    return "complex"


if __name__ == "__main__":
    run_experiment()
