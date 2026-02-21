#!/usr/bin/env python3
"""
reachable_set.py — D1: Reachable Set Mapper

Map the achievable outcome space — the "advanced wave" working backward from
desired futures. Uses Latin hypercube sampling of the 6D intervention space
to discover what outcomes are achievable per patient type.

TIQM analogy (Cramer, forthcoming 2026; cf. The Quantum Handshake):
    In the Transactional Interpretation, the "advanced wave" propagates
    backward from the absorber to constrain which transactions are possible.
    Here, we work backward from desired health targets (e.g., het < 0.3) to
    discover which intervention combinations can reach them. The reachable
    set IS the space of possible "transactions" between interventions and
    outcomes.

Computes:
  1. Reachable region boundary in (heteroplasmy, ATP) space — the envelope
     of all achievable outcomes, showing what biology permits
  2. Pareto frontier (minimize het, maximize ATP) — the optimal trade-off
     curve: no protocol on the frontier can improve one metric without
     worsening the other
  3. Minimum-intervention paths to predefined health targets — the least
     aggressive protocol that still achieves the clinical goal

Biological grounding for health targets:
  - maintain_health (het<0.4, ATP>0.7): staying well below the cliff (0.70;
    Cramer Ch. V.K p.66) with healthy energy reserves
  - significant_reversal (het<0.3, ATP>0.6): rolling back decades of damage
    below typical 70-year-old levels
  - aggressive_reversal (het<0.2, ATP>0.5): approaching youthful heteroplasmy
    at the cost of some energy
  - cliff_escape (het<0.6, ATP>0.4): minimal survival goal for near-cliff
    patients — pull back from the precipice

Discovery potential:
  - Whether aggressive reversal (het < 0.2) is achievable for any patient
  - How fast the reachable set shrinks with baseline damage (aging narrows options)
  - Whether the Pareto frontier has a sharp elbow (natural "best achievable" point)
  - Whether there are isolated reachable islands requiring non-obvious combos

Scale: 400 samples × 3 patients = 1200 sims + target searches ~1200 = ~2400 total
Estimated time: ~5 minutes

Reference:
    Cramer, J.G. (forthcoming 2026). "How to Live Much Longer: The Mitochondrial DNA
    Connection." Springer. ISBN 978-3-032-17740-7.
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
from analytics import NumpyEncoder

PROJECT = Path(__file__).resolve().parent

# ── Patient profiles ─────────────────────────────────────────────────────────

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

# ── Health targets ───────────────────────────────────────────────────────────
# These represent clinically meaningful outcome categories. The heteroplasmy
# cliff at 0.70 (Cramer Ch. V.K p.66, Rossignol 2003) is the key biological
# boundary — above it, ATP production collapses via a steep sigmoid.
#
# het_max thresholds chosen relative to the cliff:
#   0.4 = comfortable safety margin (>40% below cliff)
#   0.3 = typical healthy adult level
#   0.2 = youthful level (Cramer Appendix 2: young adults ~5-15% het)
#   0.6 = danger zone but still functional (14% below cliff)
#
# atp_min thresholds relative to baseline (1.0 MU/day; Cramer Ch. VIII.A Table 3):
#   0.7 = healthy function maintained
#   0.6 = mildly compromised but viable
#   0.5 = symptomatic energy deficit
#   0.4 = minimal survival threshold

TARGETS = {
    "maintain_health": {
        "description": "Maintain healthy state",
        "het_max": 0.4,
        "atp_min": 0.7,
    },
    "significant_reversal": {
        "description": "Significant damage reversal",
        "het_max": 0.3,
        "atp_min": 0.6,
    },
    "aggressive_reversal": {
        "description": "Aggressive reversal",
        "het_max": 0.2,
        "atp_min": 0.5,
    },
    "cliff_escape": {
        "description": "Escape from near-cliff",
        "het_max": 0.6,
        "atp_min": 0.4,
    },
}


# ── Latin hypercube sampling ─────────────────────────────────────────────────

def latin_hypercube(n, d, rng):
    """Pure numpy stratified sampling in [0,1]^d.

    Latin hypercube sampling (LHS) ensures each dimension is uniformly covered
    by dividing each axis into n equal strata and placing exactly one sample per
    stratum. This provides far better space-filling than pure random sampling
    for the same sample count — critical when exploring the 6D intervention
    space where exhaustive grid search (6^6 = 46,656 points) is too expensive.

    With n=400 samples, LHS achieves coverage comparable to ~2000 random samples
    (McKay et al. 1979), sufficient to resolve the major features of the
    reachable set without excessive simulation cost.

    Args:
        n: Number of samples (400 per patient by default).
        d: Number of dimensions (6 intervention params).
        rng: numpy random generator for reproducibility.

    Returns:
        np.array of shape (n, d) with values in [0, 1].
    """
    result = np.zeros((n, d))
    for j in range(d):
        # Create n equally-spaced intervals, sample within each
        perm = rng.permutation(n)
        for i in range(n):
            lo = perm[i] / n
            hi = (perm[i] + 1) / n
            result[i, j] = lo + rng.random() * (hi - lo)
    return result


def rescale_to_intervention(samples_01):
    """Map [0,1]^6 to intervention parameter ranges.

    Args:
        samples_01: np.array of shape (n, 6), values in [0, 1].

    Returns:
        List of intervention dicts.
    """
    interventions = []
    for row in samples_01:
        intv = {}
        for j, name in enumerate(INTERVENTION_NAMES):
            lo, hi = INTERVENTION_PARAMS[name]["range"]
            intv[name] = float(lo + row[j] * (hi - lo))
        interventions.append(intv)
    return interventions


# ── Batch evaluation ─────────────────────────────────────────────────────────

def evaluate_batch(interventions, patient, baseline):
    """Simulate a batch of interventions and return outcome arrays.

    Returns:
        Dict with arrays: het_final, atp_final, total_dose, interventions.
    """
    n = len(interventions)
    het_final = np.zeros(n)
    atp_final = np.zeros(n)
    total_dose = np.zeros(n)

    for i, intv in enumerate(interventions):
        result = simulate(intervention=intv, patient=patient)
        het_final[i] = result["heteroplasmy"][-1]
        atp_final[i] = result["states"][-1, 2]
        total_dose[i] = sum(intv.values())

    return {
        "het_final": het_final,
        "atp_final": atp_final,
        "total_dose": total_dose,
        "interventions": interventions,
    }


# ── Pareto frontier ──────────────────────────────────────────────────────────

def compute_pareto_frontier(het_vals, atp_vals):
    """Non-dominated sorting: minimize het, maximize ATP.

    Args:
        het_vals: np.array of heteroplasmy values.
        atp_vals: np.array of ATP values.

    Returns:
        np.array of indices on the Pareto frontier.
    """
    n = len(het_vals)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i if: het_j <= het_i AND atp_j >= atp_i (strict on at least one)
            if (het_vals[j] <= het_vals[i] and atp_vals[j] >= atp_vals[i]
                    and (het_vals[j] < het_vals[i] or atp_vals[j] > atp_vals[i])):
                is_dominated[i] = True
                break

    frontier_idx = np.where(~is_dominated)[0]
    # Sort by het for clean boundary
    order = np.argsort(het_vals[frontier_idx])
    return frontier_idx[order]


# ── Reachable boundary ──────────────────────────────────────────────────────

def compute_reachable_boundary(het_vals, atp_vals, n_bins=20):
    """Trace boundary of achievable region in (het, ATP) space.

    For each het bin, find min/max ATP. This outlines the reachable set.

    Returns:
        Dict with het_bins, atp_upper, atp_lower arrays.
    """
    het_min, het_max = float(np.min(het_vals)), float(np.max(het_vals))
    if het_max - het_min < 1e-6:
        return {"het_bins": [], "atp_upper": [], "atp_lower": []}

    bin_edges = np.linspace(het_min, het_max, n_bins + 1)
    het_centers = []
    atp_upper = []
    atp_lower = []

    for k in range(n_bins):
        mask = (het_vals >= bin_edges[k]) & (het_vals < bin_edges[k + 1])
        if k == n_bins - 1:  # include right edge
            mask = mask | (het_vals == bin_edges[k + 1])
        if np.any(mask):
            het_centers.append(float((bin_edges[k] + bin_edges[k + 1]) / 2))
            atp_upper.append(float(np.max(atp_vals[mask])))
            atp_lower.append(float(np.min(atp_vals[mask])))

    return {
        "het_bins": het_centers,
        "atp_upper": atp_upper,
        "atp_lower": atp_lower,
    }


# ── Minimum-intervention path to target ─────────────────────────────────────

def find_minimum_intervention(target, patient, batch_cache):
    """Find minimum-dose intervention that reaches a health target.

    Args:
        target: Dict with het_max and atp_min.
        patient: Patient dict.
        batch_cache: Dict from evaluate_batch().

    Returns:
        Dict with reachable (bool), best intervention, dose.
    """
    het = batch_cache["het_final"]
    atp = batch_cache["atp_final"]
    doses = batch_cache["total_dose"]
    interventions = batch_cache["interventions"]

    # Find all samples that meet the target
    meets_target = (het <= target["het_max"]) & (atp >= target["atp_min"])

    if not np.any(meets_target):
        return {
            "reachable": False,
            "n_meeting_target": 0,
            "fraction_meeting_target": 0.0,
        }

    # Among those meeting target, find minimum total dose
    meeting_indices = np.where(meets_target)[0]
    meeting_doses = doses[meeting_indices]
    best_idx = meeting_indices[np.argmin(meeting_doses)]

    return {
        "reachable": True,
        "n_meeting_target": int(np.sum(meets_target)),
        "fraction_meeting_target": float(np.mean(meets_target)),
        "min_dose": float(doses[best_idx]),
        "best_intervention": interventions[best_idx],
        "achieved_het": float(het[best_idx]),
        "achieved_atp": float(atp[best_idx]),
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(n_samples=400, rng_seed=42):
    out_path = PROJECT / "artifacts" / "reachable_set.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(rng_seed)
    start_time = time.time()
    sim_count = 0
    d = len(INTERVENTION_NAMES)

    print("=" * 70)
    print("D1: REACHABLE SET MAPPER — Achievable Outcome Space")
    print("=" * 70)
    print(f"LHS samples per patient: {n_samples}")
    print(f"Intervention dimensions: {d}")
    print(f"Patients: {list(PATIENTS.keys())}")
    print(f"Targets: {list(TARGETS.keys())}")
    print(f"Estimated sims: ~{n_samples * len(PATIENTS)}")
    print()

    # Generate LHS samples (shared across patients for comparability)
    samples_01 = latin_hypercube(n_samples, d, rng)
    interventions = rescale_to_intervention(samples_01)

    all_results = {}
    pareto_areas = {}

    for pat_id, pat_info in PATIENTS.items():
        patient = pat_info["params"]
        print(f"\n--- {pat_info['label']} ---")

        # Compute baseline
        baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
        base_het = float(baseline["heteroplasmy"][-1])
        base_atp = float(baseline["states"][-1, 2])
        sim_count += 1
        print(f"  Baseline: het={base_het:.4f}, ATP={base_atp:.4f}")

        # Evaluate all samples
        print(f"  Evaluating {n_samples} LHS samples...", end="", flush=True)
        batch = evaluate_batch(interventions, patient, baseline)
        sim_count += n_samples
        print(f" done")

        het = batch["het_final"]
        atp = batch["atp_final"]

        print(f"  Het range: [{np.min(het):.4f}, {np.max(het):.4f}]")
        print(f"  ATP range: [{np.min(atp):.4f}, {np.max(atp):.4f}]")

        # Pareto frontier
        pareto_idx = compute_pareto_frontier(het, atp)
        pareto_het = het[pareto_idx]
        pareto_atp = atp[pareto_idx]

        # Pareto area (approximate area under frontier)
        if len(pareto_idx) > 1:
            # Trapezoidal approximation (maximize ATP = area above x-axis)
            sorted_order = np.argsort(pareto_het)
            sorted_het = pareto_het[sorted_order]
            sorted_atp = pareto_atp[sorted_order]
            area = float(np.trapezoid(sorted_atp, sorted_het))
        else:
            area = 0.0
        pareto_areas[pat_id] = area

        print(f"  Pareto frontier: {len(pareto_idx)} points, area={area:.4f}")

        # Detect elbow (point of maximum curvature on Pareto front).
        # The "elbow" is the point where the frontier bends most sharply —
        # biologically, this is where diminishing returns kick in hardest.
        # Past the elbow, large additional intervention doses yield only
        # marginal improvement in one metric while substantially sacrificing
        # the other. This identifies the natural "best achievable" protocol:
        # pushing harder wastes resources for minimal gain.
        #
        # Method: normalize frontier to [0,1]^2, find point farthest from the
        # diagonal (maximum L1 distance from the line connecting endpoints).
        # This is a standard Pareto elbow detection heuristic.
        elbow = None
        if len(pareto_idx) >= 3:
            sorted_order = np.argsort(pareto_het)
            ph = pareto_het[sorted_order]
            pa = pareto_atp[sorted_order]
            # Normalize to [0,1] for curvature
            h_norm = (ph - ph[0]) / max(ph[-1] - ph[0], 1e-9)
            a_norm = (pa - pa[-1]) / max(pa[0] - pa[-1], 1e-9)
            # Distance from diagonal = curvature proxy
            distances = np.abs(h_norm + a_norm - 1.0) / np.sqrt(2)
            elbow_local_idx = int(np.argmax(distances))
            elbow_global_idx = pareto_idx[sorted_order[elbow_local_idx]]
            elbow = {
                "het": float(het[elbow_global_idx]),
                "atp": float(atp[elbow_global_idx]),
                "intervention": interventions[elbow_global_idx],
                "total_dose": float(batch["total_dose"][elbow_global_idx]),
            }
            print(f"  Elbow point: het={elbow['het']:.4f}, ATP={elbow['atp']:.4f}")

        # Reachable boundary
        boundary = compute_reachable_boundary(het, atp)

        # Target reachability
        target_results = {}
        for target_id, target in TARGETS.items():
            tr = find_minimum_intervention(target, patient, batch)
            target_results[target_id] = tr
            status = "REACHABLE" if tr["reachable"] else "UNREACHABLE"
            detail = ""
            if tr["reachable"]:
                detail = (f" ({tr['n_meeting_target']}/{n_samples} samples, "
                         f"min dose={tr['min_dose']:.2f})")
            print(f"  Target '{target_id}': {status}{detail}")

        # Pareto frontier details
        pareto_details = []
        for idx in pareto_idx:
            pareto_details.append({
                "het": float(het[idx]),
                "atp": float(atp[idx]),
                "total_dose": float(batch["total_dose"][idx]),
                "intervention": interventions[idx],
            })

        all_results[pat_id] = {
            "label": pat_info["label"],
            "baseline": {"het": base_het, "atp": base_atp},
            "outcome_range": {
                "het": [float(np.min(het)), float(np.max(het))],
                "atp": [float(np.min(atp)), float(np.max(atp))],
            },
            "pareto_frontier": pareto_details,
            "pareto_area": area,
            "elbow": elbow,
            "boundary": boundary,
            "targets": target_results,
        }

    elapsed = time.time() - start_time

    # ── Cross-patient analysis ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"REACHABLE SET MAPPER COMPLETE — {elapsed:.1f}s ({sim_count} sims)")
    print(f"{'=' * 70}")

    print("\nPareto area by patient (larger = more achievable outcomes):")
    for pat_id, area in pareto_areas.items():
        print(f"  {pat_id}: {area:.4f}")

    # Reachable set shrinkage
    areas = list(pareto_areas.values())
    if len(areas) >= 2 and areas[0] > 0:
        shrinkage = (areas[0] - areas[-1]) / areas[0]
        print(f"\nReachable set shrinkage (young → near-cliff): {shrinkage:.1%}")

    # Target reachability summary
    print("\nTarget reachability matrix:")
    print(f"  {'Target':25s}", end="")
    for pat_id in PATIENTS:
        print(f"  {pat_id:>15s}", end="")
    print()
    for target_id in TARGETS:
        print(f"  {target_id:25s}", end="")
        for pat_id in PATIENTS:
            tr = all_results[pat_id]["targets"][target_id]
            if tr["reachable"]:
                print(f"  {'YES':>15s}", end="")
            else:
                print(f"  {'NO':>15s}", end="")
        print()

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "reachable_set",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_sims": sim_count,
        "n_samples_per_patient": n_samples,
        "intervention_names": INTERVENTION_NAMES,
        "patients": {pid: pinfo["label"] for pid, pinfo in PATIENTS.items()},
        "targets": {tid: t["description"] for tid, t in TARGETS.items()},
        "results": all_results,
        "cross_patient": {
            "pareto_areas": pareto_areas,
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
