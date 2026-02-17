#!/usr/bin/env python3
"""
multi_tissue_sim.py — D3: Multi-Tissue Simulator

Discover systemic trade-offs by coupling brain + muscle + cardiac tissues
with shared resources. Wraps existing derivatives() function, calling it
3 times per RK4 step (once per tissue). State vector: 8 × 3 = 24 states.

Biological motivation (Cramer 2025):
    The single-tissue ODE in simulator.py models one "average" tissue, but
    real aging involves differential tissue vulnerability:

    - Brain: highest metabolic demand (20% of body's O2 for 2% of mass;
      Cramer Ch. IV p.46), most sensitive to ROS, produces the most damaging
      neuroinflammation when senescent (microglia SASP cascades)
    - Cardiac: the pump that delivers all interventions via blood flow.
      If cardiac tissue fails, ALL other tissues lose delivery efficiency —
      creating a catastrophic cascade (Cramer Ch. IV pp.46-47: heart as
      high-demand tissue)
    - Muscle: most exercise-responsive tissue (biogenesis upregulation;
      Cramer Ch. VI.B p.76), moderate demand, relatively resilient

    These tissues share limited NAD+ pool and are coupled by systemic
    inflammation (SASP from any tissue affects all) and cardiac output.

Coupling model:
  - NAD+ sharing: global pool distributed by metabolic demand weighting.
    Brain consumes 2× because of its extreme oxidative metabolism
  - Systemic inflammation: SASP from any tissue raises inflammation for all.
    Brain SASP weighted 1.5× (neuroinflammation cascades to systemic via
    blood-brain barrier breakdown in aging; not directly from Cramer but
    consistent with inflammaging discussion in Ch. VII.A pp.89-92)
  - Cardiac blood flow: cardiac_ATP / BASELINE_ATP modulates intervention
    delivery to ALL tissues (if heart fails, drugs can't reach targets)

Does NOT modify simulator.py — wraps existing derivatives() for 3-tissue coupling.

Discovery potential:
  - Cardiac cascade (heart failure → blood flow reduction → systemic collapse)
  - Resource allocation trade-offs (brain priority vs cardiac safety)
  - Whether "worst-first" dynamic allocation outperforms static
  - Neuroinflammation cross-tissue coupling
  - Systemic NAD+ competition (brain depleting pool for other tissues)

Scale: 5 protocols × 4 allocations + 10 allocation sweep = ~30 sims (but each
       is 3× computation). Estimated time: ~2 minutes

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
    INTERVENTION_NAMES,
    TISSUE_PROFILES,
    BASELINE_ATP,
    SIM_YEARS, DT, N_STATES,
)
from simulator import derivatives, initial_state
from analytics import NumpyEncoder

PROJECT = Path(__file__).resolve().parent

# ── Tissue configuration ────────────────────────────────────────────────────

TISSUE_NAMES = ["brain", "muscle", "cardiac"]

# NAD demand weights: how much of the shared NAD+ pool each tissue draws.
# Brain = 2.0: consumes ~20% of body's oxygen for ~2% of mass (Cramer Ch. IV
#   p.46), implying roughly 10× higher metabolic rate per unit mass. We use 2.0
#   (not 10) because the ODE already models metabolic_demand in tissue profiles,
#   and NAD demand scales sublinearly with total oxidative metabolism.
# Cardiac = 1.5: the heart beats continuously, demanding substantial NAD+ for
#   ATP production via oxidative phosphorylation (Cramer Ch. IV pp.46-47).
# Muscle = 1.0: baseline demand, modifiable by exercise_level.
NAD_DEMAND = {"brain": 2.0, "muscle": 1.0, "cardiac": 1.5}

# SASP inflammation weights: how much each tissue's senescent cell burden
# contributes to systemic inflammation.
# Brain = 1.5: neuroinflammation from senescent microglia cascades beyond the
#   CNS via cytokine release and blood-brain barrier dysfunction. This is
#   consistent with Cramer's inflammaging discussion (Ch. VII.A pp.89-92) and
#   the observation that neurodegeneration drives systemic decline.
# Muscle and cardiac = 1.0: standard SASP contribution to circulating cytokines.
SASP_WEIGHT = {"brain": 1.5, "muscle": 1.0, "cardiac": 1.0}

# ── Transplant allocation strategies ────────────────────────────────────────

ALLOCATION_STRATEGIES = {
    "equal": {
        "description": "Equal allocation across tissues",
        "brain": 1.0 / 3.0,
        "muscle": 1.0 / 3.0,
        "cardiac": 1.0 / 3.0,
    },
    "brain_priority": {
        "description": "Brain priority (60/20/20)",
        "brain": 0.60,
        "muscle": 0.20,
        "cardiac": 0.20,
    },
    "cardiac_first": {
        "description": "Cardiac first (20/20/60)",
        "brain": 0.20,
        "muscle": 0.20,
        "cardiac": 0.60,
    },
    "worst_first": {
        "description": "Dynamic: most to lowest-ATP tissue",
        "brain": None,  # computed dynamically
        "muscle": None,
        "cardiac": None,
    },
}

# ── Intervention profiles ───────────────────────────────────────────────────

INTERVENTION_PROFILES = {
    "no_treatment": {
        "description": "No treatment",
        "params": dict(DEFAULT_INTERVENTION),
    },
    "cocktail": {
        "description": "Balanced cocktail (rapa+NAD+seno+exercise)",
        "params": {
            "rapamycin_dose": 0.5, "nad_supplement": 0.5,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5,
        },
    },
    "transplant_heavy": {
        "description": "Transplant-heavy protocol",
        "params": {
            "rapamycin_dose": 0.25, "nad_supplement": 0.25,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.75, "exercise_level": 0.25,
        },
    },
    "nad_heavy": {
        "description": "NAD-heavy protocol",
        "params": {
            "rapamycin_dose": 0.25, "nad_supplement": 0.75,
            "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.25,
        },
    },
    "balanced": {
        "description": "Balanced all-modality protocol",
        "params": {
            "rapamycin_dose": 0.5, "nad_supplement": 0.5,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.5, "exercise_level": 0.5,
        },
    },
}

# Default patient
DEFAULT_PATIENT = {
    "baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
    "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0, "inflammation_level": 0.25,
}


# ── Global pool computation ─────────────────────────────────────────────────

def _compute_global_pools(tissue_states, profiles):
    """Compute shared resource pools from current tissue states.

    Args:
        tissue_states: Dict mapping tissue name → state vector (8,).
            8D per tissue: [N_h, N_del, ATP, ROS, NAD, Sen, ΔΨ, N_pt].
        profiles: Dict mapping tissue name → TISSUE_PROFILES entry.

    Returns:
        Dict with:
            global_nad: NAD pool adjusted for cross-tissue competition
            systemic_inflammation: combined SASP from all tissues
            blood_flow: cardiac ATP → delivery factor for all tissues
    """
    # NAD+ sharing: global pool = average NAD weighted by inverse demand
    # More demanding tissues draw more from the pool.
    total_nad = 0.0
    total_demand = 0.0
    for tissue in TISSUE_NAMES:
        state = tissue_states[tissue]
        nad = max(state[4], 0.0)
        demand = NAD_DEMAND[tissue]
        total_nad += nad * demand
        total_demand += demand
    global_nad = total_nad / total_demand if total_demand > 0 else 0.5

    # Systemic inflammation: SASP from any tissue raises inflammation for all
    # Brain SASP weighted 1.5× (neuroinflammation cascades)
    total_sasp = 0.0
    for tissue in TISSUE_NAMES:
        state = tissue_states[tissue]
        sen = max(state[5], 0.0)
        total_sasp += sen * SASP_WEIGHT[tissue]
    systemic_inflammation = min(total_sasp / len(TISSUE_NAMES), 1.0)

    # Cardiac blood flow: cardiac ATP determines systemic delivery efficiency.
    # Healthy cardiac output (ATP ≈ BASELINE_ATP = 1.0) → blood_flow = 1.0.
    # If cardiac ATP drops (heart failure), intervention delivery degrades.
    #   Lower bound 0.3: even severe heart failure maintains ~30% of normal
    #   cardiac output (cardiogenic shock threshold; prevents division-by-zero
    #   and models that some delivery always occurs even in extremis).
    #   Upper bound 1.2: mild supranormal output possible with exercise-enhanced
    #   cardiac function, but physiologically capped.
    cardiac_atp = max(tissue_states["cardiac"][2], 0.0)
    blood_flow = np.clip(cardiac_atp / BASELINE_ATP, 0.3, 1.2)

    return {
        "global_nad": global_nad,
        "systemic_inflammation": systemic_inflammation,
        "blood_flow": blood_flow,
    }


# ── Coupling application ────────────────────────────────────────────────────

def _apply_coupling(tissue, patient, intervention, pools, allocation):
    """Modify local params with global coupling effects.

    Args:
        tissue: Tissue name ("brain", "muscle", "cardiac").
        patient: Patient dict (will be modified copy).
        intervention: Intervention dict (will be modified copy).
        pools: Global pools dict from _compute_global_pools.
        allocation: Dict with tissue → fraction of transplant resources.

    Returns:
        (modified_patient, modified_intervention)
    """
    patient = dict(patient)
    intervention = dict(intervention)

    # Systemic inflammation modifies patient inflammation level
    patient["inflammation_level"] = min(
        patient.get("inflammation_level", 0.0) + pools["systemic_inflammation"] * 0.5,
        1.0)

    # Blood flow modulates intervention delivery
    # Low cardiac output → interventions reach tissues less effectively
    delivery = pools["blood_flow"]
    for name in INTERVENTION_NAMES:
        if name != "exercise_level":  # exercise is self-administered
            intervention[name] = intervention[name] * delivery

    # Transplant allocation: split transplant_rate among tissues.
    # The × len(TISSUE_NAMES) factor ensures that equal allocation (1/3 each)
    # recovers the same per-tissue transplant rate as single-tissue simulation
    # (i.e., equal split doesn't dilute: 0.5 * 1/3 * 3 = 0.5 per tissue).
    # Brain-priority (0.6) gives that tissue 0.5 * 0.6 * 3 = 0.9, nearly
    # double the single-tissue rate, representing concentrated resource focus.
    # This models the clinical reality that transplanted mitochondria (mitlets;
    # Cramer Ch. VIII.G pp.104-107) must be directed to specific tissues.
    if allocation.get(tissue) is not None:
        intervention["transplant_rate"] = (
            intervention["transplant_rate"] * allocation[tissue] * len(TISSUE_NAMES))

    return patient, intervention


# ── Multi-tissue RK4 step ───────────────────────────────────────────────────

def multi_tissue_step(tissue_states, t, dt, intervention, patient,
                      profiles, pools, allocation):
    """One RK4 step for all 3 tissues with coupling.

    Args:
        tissue_states: Dict mapping tissue name → np.array(8,).
            8D state vector per tissue (C11):
              [0] N_healthy, [1] N_deletion, [2] ATP, [3] ROS,
              [4] NAD, [5] Senescent_fraction, [6] Membrane_potential,
              [7] N_point
        t: Current time in years.
        dt: Timestep.
        intervention: Base intervention dict.
        patient: Base patient dict.
        profiles: TISSUE_PROFILES.
        pools: Current global pools.
        allocation: Current allocation dict.

    Returns:
        Dict mapping tissue name → new state np.array(8,).
    """
    new_states = {}

    for tissue in TISSUE_NAMES:
        state = tissue_states[tissue]
        profile = profiles[tissue]
        tissue_mods = {
            "ros_sensitivity": profile["ros_sensitivity"],
            "biogenesis_rate": profile["biogenesis_rate"],
        }

        # Apply coupling
        coupled_patient, coupled_intervention = _apply_coupling(
            tissue, patient, intervention, pools, allocation)
        coupled_patient["metabolic_demand"] = profile["metabolic_demand"]

        # RK4 step using existing derivatives()
        k1 = derivatives(state, t, coupled_intervention, coupled_patient, tissue_mods)
        k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt,
                         coupled_intervention, coupled_patient, tissue_mods)
        k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt,
                         coupled_intervention, coupled_patient, tissue_mods)
        k4 = derivatives(state + dt * k3, t + dt,
                         coupled_intervention, coupled_patient, tissue_mods)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce constraints
        new_state = np.maximum(new_state, 0.0)
        new_state[5] = min(new_state[5], 1.0)

        new_states[tissue] = new_state

    return new_states


# ── Full coupled simulation ─────────────────────────────────────────────────

def multi_tissue_simulate(intervention, patient, sim_years=None,
                          allocation_strategy="equal"):
    """Run coupled 3-tissue simulation.

    Args:
        intervention: Base intervention dict.
        patient: Base patient dict.
        sim_years: Override simulation horizon.
        allocation_strategy: Key into ALLOCATION_STRATEGIES or a dict.

    Returns:
        Dict with per-tissue trajectories and systemic metrics.
    """
    if sim_years is None:
        sim_years = SIM_YEARS
    dt = DT
    n_steps = int(sim_years / dt)

    # Resolve allocation strategy
    if isinstance(allocation_strategy, str):
        alloc_template = ALLOCATION_STRATEGIES[allocation_strategy]
        is_dynamic = allocation_strategy == "worst_first"
    else:
        alloc_template = allocation_strategy
        is_dynamic = False

    # Initialize per-tissue states
    profiles = TISSUE_PROFILES
    tissue_states = {}
    for tissue in TISSUE_NAMES:
        p = dict(patient)
        p["metabolic_demand"] = profiles[tissue]["metabolic_demand"]
        tissue_states[tissue] = initial_state(p)

    # Pre-allocate trajectories
    time_arr = np.zeros(n_steps + 1)
    trajectories = {tissue: np.zeros((n_steps + 1, N_STATES))
                    for tissue in TISSUE_NAMES}
    het_trajectories = {tissue: np.zeros(n_steps + 1) for tissue in TISSUE_NAMES}
    pool_history = {
        "global_nad": np.zeros(n_steps + 1),
        "systemic_inflammation": np.zeros(n_steps + 1),
        "blood_flow": np.zeros(n_steps + 1),
    }

    # Record initial states
    for tissue in TISSUE_NAMES:
        trajectories[tissue][0] = tissue_states[tissue]
        # C11 8D state: 3-pool copy number = N_h[0] + N_del[1] + N_pt[7].
        # Total heteroplasmy = (N_del + N_pt) / total. Uses total het (not
        # deletion-only) because cross-tissue inflammation coupling depends on
        # ALL defective mitochondria: both deletions and point mutations produce
        # defective ETC complexes that leak ROS, contributing to SASP and the
        # systemic inflammation pool. The cliff itself is deletion-driven, but
        # the inter-tissue coupling pathway is ROS→inflammation→SASP, which
        # does not require threshold behavior.
        total = tissue_states[tissue][0] + tissue_states[tissue][1] + tissue_states[tissue][7]
        het_trajectories[tissue][0] = (
            (tissue_states[tissue][1] + tissue_states[tissue][7]) / max(total, 1e-12))

    pools = _compute_global_pools(tissue_states, profiles)
    pool_history["global_nad"][0] = pools["global_nad"]
    pool_history["systemic_inflammation"][0] = pools["systemic_inflammation"]
    pool_history["blood_flow"][0] = pools["blood_flow"]

    # Main integration loop
    for i in range(n_steps):
        t = i * dt

        # Update global pools
        pools = _compute_global_pools(tissue_states, profiles)
        pool_history["global_nad"][i] = pools["global_nad"]
        pool_history["systemic_inflammation"][i] = pools["systemic_inflammation"]
        pool_history["blood_flow"][i] = pools["blood_flow"]

        # Compute allocation for this step
        if is_dynamic:
            # Dynamic: most resources to lowest-ATP tissue
            atps = {tissue: max(tissue_states[tissue][2], 0.01)
                    for tissue in TISSUE_NAMES}
            inv_atps = {t: 1.0 / a for t, a in atps.items()}
            total_inv = sum(inv_atps.values())
            allocation = {t: inv_atps[t] / total_inv for t in TISSUE_NAMES}
        else:
            allocation = {t: alloc_template[t] for t in TISSUE_NAMES}

        # Step all tissues
        tissue_states = multi_tissue_step(
            tissue_states, t, dt, intervention, patient,
            profiles, pools, allocation)

        # Record
        time_arr[i + 1] = (i + 1) * dt
        for tissue in TISSUE_NAMES:
            trajectories[tissue][i + 1] = tissue_states[tissue]
            # 3-pool total het for cross-tissue coupling (see initial recording comment)
            total = tissue_states[tissue][0] + tissue_states[tissue][1] + tissue_states[tissue][7]
            het_trajectories[tissue][i + 1] = (
                (tissue_states[tissue][1] + tissue_states[tissue][7]) / max(total, 1e-12))

    # Final pool values
    pools = _compute_global_pools(tissue_states, profiles)
    pool_history["global_nad"][n_steps] = pools["global_nad"]
    pool_history["systemic_inflammation"][n_steps] = pools["systemic_inflammation"]
    pool_history["blood_flow"][n_steps] = pools["blood_flow"]

    return {
        "time": time_arr,
        "trajectories": trajectories,
        "heteroplasmy": het_trajectories,
        "pool_history": pool_history,
        "intervention": intervention,
        "patient": patient,
        "allocation_strategy": (allocation_strategy if isinstance(allocation_strategy, str)
                                else "custom"),
    }


# ── Systemic metrics ────────────────────────────────────────────────────────

def compute_systemic_metrics(result):
    """Compute aggregate metrics from multi-tissue simulation.

    Returns:
        Dict with systemic health indicators.
    """
    metrics = {}
    tissue_atps = {}
    tissue_hets = {}

    for tissue in TISSUE_NAMES:
        final_state = result["trajectories"][tissue][-1]
        final_het = result["heteroplasmy"][tissue][-1]
        tissue_atps[tissue] = float(final_state[2])
        tissue_hets[tissue] = float(final_het)

        atp_traj = result["trajectories"][tissue][:, 2]
        metrics[f"{tissue}_atp_final"] = float(final_state[2])
        metrics[f"{tissue}_atp_min"] = float(np.min(atp_traj))
        metrics[f"{tissue}_het_final"] = float(final_het)
        metrics[f"{tissue}_sen_final"] = float(final_state[5])

    # Systemic indicators
    atp_values = list(tissue_atps.values())
    metrics["worst_tissue_atp"] = min(atp_values)
    metrics["worst_tissue"] = min(tissue_atps, key=tissue_atps.get)
    metrics["average_het"] = float(np.mean(list(tissue_hets.values())))
    metrics["max_het"] = max(tissue_hets.values())

    # Systemic health score: harmonic mean of tissue ATPs.
    # The harmonic mean (not arithmetic) penalizes any single tissue failing —
    # reflecting that a person with healthy brain and muscle but cardiac failure
    # is NOT "2/3 healthy." The weakest tissue dominates clinical outcome.
    # This mirrors the "weakest link" principle in Cramer's model where the
    # heteroplasmy cliff in ANY critical tissue triggers systemic collapse.
    if all(a > 0 for a in atp_values):
        metrics["systemic_health"] = float(
            len(atp_values) / sum(1.0 / a for a in atp_values))
    else:
        metrics["systemic_health"] = 0.0

    # Total senescence
    total_sen = sum(
        float(result["trajectories"][t][-1, 5]) for t in TISSUE_NAMES)
    metrics["total_senescence"] = total_sen

    # Blood flow at end
    metrics["final_blood_flow"] = float(result["pool_history"]["blood_flow"][-1])

    # Cardiac cascade indicator: did cardiac ATP drop below 0.5?
    cardiac_atp = result["trajectories"]["cardiac"][:, 2]
    metrics["cardiac_crisis"] = bool(np.any(cardiac_atp < 0.5 * BASELINE_ATP))

    return metrics


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    out_path = PROJECT / "artifacts" / "multi_tissue_sim.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    sim_count = 0

    print("=" * 70)
    print("D3: MULTI-TISSUE SIMULATOR — Systemic Trade-Off Discovery")
    print("=" * 70)
    print(f"Tissues: {TISSUE_NAMES}")
    print(f"Protocols: {list(INTERVENTION_PROFILES.keys())}")
    print(f"Allocation strategies: {list(ALLOCATION_STRATEGIES.keys())}")
    print()

    # ── Experiment 1: All protocol × allocation combinations ─────────────
    print("--- Experiment 1: Protocol × Allocation Grid ---")
    grid_results = []

    for prof_id, prof_info in INTERVENTION_PROFILES.items():
        for alloc_id in ALLOCATION_STRATEGIES:
            print(f"  {prof_id:20s} × {alloc_id:15s}", end="", flush=True)

            result = multi_tissue_simulate(
                intervention=prof_info["params"],
                patient=DEFAULT_PATIENT,
                allocation_strategy=alloc_id,
            )
            metrics = compute_systemic_metrics(result)
            sim_count += 1

            # Compact trajectory storage (subsample to ~100 points)
            n_total = len(result["time"])
            step = max(1, n_total // 100)
            compact_trajectories = {}
            compact_het = {}
            for tissue in TISSUE_NAMES:
                compact_trajectories[tissue] = {
                    "atp": result["trajectories"][tissue][::step, 2].tolist(),
                    "het": result["heteroplasmy"][tissue][::step].tolist(),
                    "sen": result["trajectories"][tissue][::step, 5].tolist(),
                }
                compact_het[tissue] = result["heteroplasmy"][tissue][::step].tolist()

            entry = {
                "protocol": prof_id,
                "allocation": alloc_id,
                "metrics": metrics,
                "compact_trajectories": compact_trajectories,
                "compact_time": result["time"][::step].tolist(),
                "compact_pools": {
                    "blood_flow": result["pool_history"]["blood_flow"][::step].tolist(),
                    "inflammation": result["pool_history"]["systemic_inflammation"][::step].tolist(),
                    "nad": result["pool_history"]["global_nad"][::step].tolist(),
                },
            }
            grid_results.append(entry)

            print(f"  sys_health={metrics['systemic_health']:.3f} "
                  f"worst={metrics['worst_tissue']}({metrics['worst_tissue_atp']:.3f}) "
                  f"blood={metrics['final_blood_flow']:.3f}")

    # ── Experiment 2: Transplant allocation sweep ────────────────────────
    print("\n--- Experiment 2: Brain Allocation Sweep (cocktail protocol) ---")
    sweep_results = []
    brain_fractions = np.linspace(0.0, 1.0, 11)

    for brain_frac in brain_fractions:
        remaining = 1.0 - brain_frac
        allocation = {
            "brain": float(brain_frac),
            "muscle": float(remaining / 2),
            "cardiac": float(remaining / 2),
        }

        result = multi_tissue_simulate(
            intervention=INTERVENTION_PROFILES["transplant_heavy"]["params"],
            patient=DEFAULT_PATIENT,
            allocation_strategy=allocation,
        )
        metrics = compute_systemic_metrics(result)
        sim_count += 1

        sweep_results.append({
            "brain_fraction": float(brain_frac),
            "allocation": allocation,
            "metrics": metrics,
        })

        print(f"  brain={brain_frac:.0%}: "
              f"sys={metrics['systemic_health']:.3f} "
              f"brain_atp={metrics['brain_atp_final']:.3f} "
              f"cardiac_atp={metrics['cardiac_atp_final']:.3f} "
              f"{'CARDIAC CRISIS' if metrics['cardiac_crisis'] else ''}")

    elapsed = time.time() - start_time

    # ── Analysis ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"MULTI-TISSUE SIMULATOR COMPLETE — {elapsed:.1f}s ({sim_count} sims)")
    print(f"{'=' * 70}")

    # Best allocation per objective
    print("\nBest configurations:")

    # Best systemic health
    best_systemic = max(grid_results, key=lambda r: r["metrics"]["systemic_health"])
    print(f"  Best systemic health: {best_systemic['protocol']} × "
          f"{best_systemic['allocation']} "
          f"(score={best_systemic['metrics']['systemic_health']:.3f})")

    # Best brain
    best_brain = max(grid_results, key=lambda r: r["metrics"]["brain_atp_final"])
    print(f"  Best brain ATP: {best_brain['protocol']} × "
          f"{best_brain['allocation']} "
          f"(ATP={best_brain['metrics']['brain_atp_final']:.3f})")

    # Least cardiac crisis
    no_crisis = [r for r in grid_results if not r["metrics"]["cardiac_crisis"]]
    if no_crisis:
        best_safe = max(no_crisis, key=lambda r: r["metrics"]["systemic_health"])
        print(f"  Best without cardiac crisis: {best_safe['protocol']} × "
              f"{best_safe['allocation']} "
              f"(score={best_safe['metrics']['systemic_health']:.3f})")

    # Allocation sweep analysis
    best_sweep = max(sweep_results, key=lambda r: r["metrics"]["systemic_health"])
    print(f"\n  Optimal brain allocation (transplant-heavy): "
          f"{best_sweep['brain_fraction']:.0%} "
          f"(systemic={best_sweep['metrics']['systemic_health']:.3f})")

    # Cardiac cascade detection
    cardiac_crises = [r for r in grid_results if r["metrics"]["cardiac_crisis"]]
    print(f"\n  Cardiac cascade detected in: {len(cardiac_crises)}/{len(grid_results)} scenarios")
    if cardiac_crises:
        for r in cardiac_crises[:3]:
            print(f"    {r['protocol']} × {r['allocation']}: "
                  f"cardiac_atp={r['metrics']['cardiac_atp_final']:.3f}")

    # Trade-off description
    print("\nKey trade-offs discovered:")
    # Compare brain_priority vs cardiac_first for transplant_heavy
    brain_prio = next((r for r in grid_results
                       if r["protocol"] == "transplant_heavy"
                       and r["allocation"] == "brain_priority"), None)
    cardiac_prio = next((r for r in grid_results
                         if r["protocol"] == "transplant_heavy"
                         and r["allocation"] == "cardiac_first"), None)
    if brain_prio and cardiac_prio:
        bp = brain_prio["metrics"]
        cp = cardiac_prio["metrics"]
        print(f"  Brain priority:   brain_atp={bp['brain_atp_final']:.3f} "
              f"cardiac_atp={bp['cardiac_atp_final']:.3f} "
              f"systemic={bp['systemic_health']:.3f}")
        print(f"  Cardiac first:    brain_atp={cp['brain_atp_final']:.3f} "
              f"cardiac_atp={cp['cardiac_atp_final']:.3f} "
              f"systemic={cp['systemic_health']:.3f}")

    # Worst-first vs equal
    worst_first_results = [r for r in grid_results if r["allocation"] == "worst_first"]
    equal_results = [r for r in grid_results if r["allocation"] == "equal"]
    if worst_first_results and equal_results:
        wf_avg = np.mean([r["metrics"]["systemic_health"] for r in worst_first_results])
        eq_avg = np.mean([r["metrics"]["systemic_health"] for r in equal_results])
        better = "worst-first" if wf_avg > eq_avg else "equal"
        print(f"  Dynamic allocation: worst-first avg={wf_avg:.3f} "
              f"vs equal avg={eq_avg:.3f} → {better} wins")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "multi_tissue_sim",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_sims": sim_count,
        "tissue_names": TISSUE_NAMES,
        "nad_demand": NAD_DEMAND,
        "sasp_weights": SASP_WEIGHT,
        "patient": DEFAULT_PATIENT,
        "grid_results": grid_results,
        "allocation_sweep": sweep_results,
        "summary": {
            "best_systemic": {
                "protocol": best_systemic["protocol"],
                "allocation": best_systemic["allocation"],
                "score": best_systemic["metrics"]["systemic_health"],
            },
            "best_brain": {
                "protocol": best_brain["protocol"],
                "allocation": best_brain["allocation"],
                "brain_atp": best_brain["metrics"]["brain_atp_final"],
            },
            "optimal_brain_allocation": best_sweep["brain_fraction"],
            "cardiac_crisis_count": len(cardiac_crises),
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
