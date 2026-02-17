#!/usr/bin/env python3
"""
causal_surgery.py

Mid-simulation intervention switching to find the point of no return.

Adapted from causal_surgery.py in the parent Evolutionary-Robotics project,
which did mid-simulation brain transplants (weight switching at specific
timesteps). Here we switch from no-treatment to full intervention at various
time points to answer: "When is it too late to start treatment?"

Experiments:
  1. Forward surgery: no treatment → intervention at year T
  2. Reverse surgery: intervention → no treatment at year T
  3. Protocol switching: switch from one protocol to another at year T

For each, we sweep T across [0, 2, 5, 8, 10, 15, 20, 25] years and record
the final outcome (ATP, heteroplasmy, copy number).

Scale: 3 patients × 4 interventions × 8 switch times × 2 directions = 192 sims
Estimated time: ~30 seconds (pure numpy, no LLM)
"""

import json
import time
from pathlib import Path

import numpy as np

from constants import (
    SIM_YEARS, DT, N_STATES,
    DEFAULT_INTERVENTION,
)
from simulator import initial_state, _rk4_step, _total_heteroplasmy
from analytics import compute_energy, compute_damage, NumpyEncoder

PROJECT = Path(__file__).resolve().parent


# ── Patients ────────────────────────────────────────────────────────────────

PATIENTS = {
    "healthy_30": {
        "label": "Healthy 30yo (10% het)",
        "params": {
            "baseline_age": 30.0, "baseline_heteroplasmy": 0.10,
            "baseline_nad_level": 0.9, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.1,
        },
    },
    "moderate_60": {
        "label": "Moderate 60yo (40% het)",
        "params": {
            "baseline_age": 60.0, "baseline_heteroplasmy": 0.40,
            "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.3,
        },
    },
    "near_cliff_75": {
        "label": "Near-cliff 75yo (60% het)",
        "params": {
            "baseline_age": 75.0, "baseline_heteroplasmy": 0.60,
            "baseline_nad_level": 0.4, "genetic_vulnerability": 1.25,
            "metabolic_demand": 1.0, "inflammation_level": 0.5,
        },
    },
}


# ── Interventions ───────────────────────────────────────────────────────────

INTERVENTIONS = {
    "rapamycin_only": {
        "label": "Rapamycin only",
        "params": {**DEFAULT_INTERVENTION, "rapamycin_dose": 0.75},
    },
    "nad_only": {
        "label": "NAD+ only",
        "params": {**DEFAULT_INTERVENTION, "nad_supplement": 0.75},
    },
    "full_cocktail": {
        "label": "Full cocktail",
        "params": {
            "rapamycin_dose": 0.5, "nad_supplement": 0.75,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5,
        },
    },
    "transplant_only": {
        "label": "Transplant only",
        "params": {**DEFAULT_INTERVENTION, "transplant_rate": 0.75},
    },
}


# ── Switch times (years from start) ────────────────────────────────────────

SWITCH_TIMES = [0, 2, 5, 8, 10, 15, 20, 25]


# ── Two-phase simulation ───────────────────────────────────────────────────

def simulate_with_surgery(patient, phase1_intervention, phase2_intervention,
                          switch_year, sim_years=SIM_YEARS, dt=DT):
    """Run simulation with intervention switching at switch_year.

    Phase 1: use phase1_intervention for t < switch_year
    Phase 2: use phase2_intervention for t >= switch_year

    Returns:
        Dict matching simulate() output format.
    """
    n_steps = int(sim_years / dt)
    state = initial_state(patient)

    time_arr = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, N_STATES))
    het_arr = np.zeros(n_steps + 1)

    states[0] = state
    # C11: Total heteroplasmy = (N_del + N_pt) / (N_h + N_del + N_pt).
    # Uses 3-pool _total_heteroplasmy() instead of the old 2-arg _heteroplasmy()
    # because the state vector is now 8D:
    #   state[0] = N_healthy     — undamaged mtDNA copies
    #   state[1] = N_deletion    — large-deletion mtDNA (exponential growth, drives cliff)
    #   state[7] = N_point       — point-mutated mtDNA (linear growth, no cliff effect)
    #
    # Total het (not deletion-only) is appropriate here because causal_surgery
    # tracks the patient's overall damage burden for treatment timing analysis.
    # The "point of no return" depends on cumulative damage from BOTH mutation
    # types, not just deletions, since reversal must address all damaged copies.
    het_arr[0] = _total_heteroplasmy(state[0], state[1], state[7])

    for i in range(n_steps):
        t = i * dt
        intervention = phase1_intervention if t < switch_year else phase2_intervention
        state = _rk4_step(state, t, dt, intervention, patient)
        state = np.maximum(state, 0.0)
        state[5] = min(state[5], 1.0)

        time_arr[i + 1] = (i + 1) * dt
        states[i + 1] = state
        # 3-pool total het: both deletion and point mutations count toward
        # the overall damage fraction reported for treatment timing analysis.
        het_arr[i + 1] = _total_heteroplasmy(state[0], state[1], state[7])

    return {
        "time": time_arr,
        "states": states,
        "heteroplasmy": het_arr,
        "intervention": phase2_intervention,
        "patient": patient,
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    out_path = PROJECT / "artifacts" / "causal_surgery.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()
    trial_num = 0
    n_total = (len(PATIENTS) * len(INTERVENTIONS) * len(SWITCH_TIMES) * 2
               + len(PATIENTS))  # +baselines

    print(f"{'=' * 70}")
    print(f"CAUSAL SURGERY EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Patients: {list(PATIENTS.keys())}")
    print(f"Interventions: {list(INTERVENTIONS.keys())}")
    print(f"Switch times: {SWITCH_TIMES}")
    print(f"Total sims: ~{n_total}")
    print()

    for pat_id, pat_info in PATIENTS.items():
        patient = pat_info["params"]

        # Baseline (no treatment ever)
        trial_num += 1
        print(f"[{trial_num}] {pat_id} | baseline (no treatment)", flush=True)
        from simulator import simulate
        baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
        energy = compute_energy(baseline)
        damage = compute_damage(baseline)
        results.append({
            "patient": pat_id,
            "patient_label": pat_info["label"],
            "intervention": "none",
            "direction": "baseline",
            "switch_year": None,
            "final_atp": float(baseline["states"][-1, 2]),
            "final_het": float(baseline["heteroplasmy"][-1]),
            # C11: 3-pool copy number = N_h + N_del + N_pt (was N_h + N_d pre-C11)
            "final_copy_number": float(baseline["states"][-1, 0] + baseline["states"][-1, 1] + baseline["states"][-1, 7]),
            "time_to_crisis": energy["time_to_crisis_years"],
            "time_to_cliff": damage["time_to_cliff_years"],
            "atp_trajectory": baseline["states"][:, 2].tolist(),
            "het_trajectory": baseline["heteroplasmy"].tolist(),
        })

        for intv_id, intv_info in INTERVENTIONS.items():
            intervention = intv_info["params"]

            for switch_year in SWITCH_TIMES:
                # Forward: no treatment → intervention at switch_year
                trial_num += 1
                print(f"[{trial_num}] {pat_id} | {intv_id} | forward @ year {switch_year}",
                      end=" ", flush=True)

                result_fwd = simulate_with_surgery(
                    patient, DEFAULT_INTERVENTION, intervention, switch_year)
                e_fwd = compute_energy(result_fwd)
                d_fwd = compute_damage(result_fwd)

                final_atp = float(result_fwd["states"][-1, 2])
                final_het = float(result_fwd["heteroplasmy"][-1])
                print(f"-> ATP={final_atp:.3f} het={final_het:.3f}")

                results.append({
                    "patient": pat_id,
                    "patient_label": pat_info["label"],
                    "intervention": intv_id,
                    "intervention_label": intv_info["label"],
                    "direction": "forward",
                    "switch_year": switch_year,
                    "final_atp": final_atp,
                    "final_het": final_het,
                    # C11: 3-pool copy number = N_h + N_del + N_pt
                    "final_copy_number": float(
                        result_fwd["states"][-1, 0] + result_fwd["states"][-1, 1] + result_fwd["states"][-1, 7]),
                    "time_to_crisis": e_fwd["time_to_crisis_years"],
                    "time_to_cliff": d_fwd["time_to_cliff_years"],
                })

                # Reverse: intervention → no treatment at switch_year
                trial_num += 1
                print(f"[{trial_num}] {pat_id} | {intv_id} | reverse @ year {switch_year}",
                      end=" ", flush=True)

                result_rev = simulate_with_surgery(
                    patient, intervention, DEFAULT_INTERVENTION, switch_year)
                e_rev = compute_energy(result_rev)
                d_rev = compute_damage(result_rev)

                final_atp_r = float(result_rev["states"][-1, 2])
                final_het_r = float(result_rev["heteroplasmy"][-1])
                print(f"-> ATP={final_atp_r:.3f} het={final_het_r:.3f}")

                results.append({
                    "patient": pat_id,
                    "patient_label": pat_info["label"],
                    "intervention": intv_id,
                    "intervention_label": intv_info["label"],
                    "direction": "reverse",
                    "switch_year": switch_year,
                    "final_atp": final_atp_r,
                    "final_het": final_het_r,
                    # C11: 3-pool copy number = N_h + N_del + N_pt
                    "final_copy_number": float(
                        result_rev["states"][-1, 0] + result_rev["states"][-1, 1] + result_rev["states"][-1, 7]),
                    "time_to_crisis": e_rev["time_to_crisis_years"],
                    "time_to_cliff": d_rev["time_to_cliff_years"],
                })

    elapsed = time.time() - start_time

    # ── Analysis ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"CAUSAL SURGERY COMPLETE — {elapsed:.1f}s ({len(results)} sims)")
    print(f"{'=' * 70}")

    # Point of no return analysis: for each patient × intervention,
    # find the latest switch_year where forward surgery still helps
    print("\nPOINT OF NO RETURN (forward surgery: no-treatment → intervention)")
    print(f"{'Patient':20s} {'Intervention':20s} {'Latest helpful switch':>22s} {'Final ATP':>10s}")
    print("-" * 74)

    for pat_id, pat_info in PATIENTS.items():
        # Get baseline final values
        baseline_row = [r for r in results if r["patient"] == pat_id
                        and r["direction"] == "baseline"][0]
        baseline_atp = baseline_row["final_atp"]
        baseline_het = baseline_row["final_het"]

        for intv_id, intv_info in INTERVENTIONS.items():
            forward_rows = [r for r in results
                           if r["patient"] == pat_id
                           and r["intervention"] == intv_id
                           and r["direction"] == "forward"]
            forward_rows.sort(key=lambda r: r["switch_year"])

            # Find latest switch where final ATP > baseline ATP
            latest_helpful = None
            for row in forward_rows:
                if row["final_atp"] > baseline_atp + 0.01:
                    latest_helpful = row["switch_year"]

            ponr = f"year {latest_helpful}" if latest_helpful is not None else "never helps"
            best_row = max(forward_rows, key=lambda r: r["final_atp"])
            print(f"{pat_info['label']:20s} {intv_info['label']:20s} "
                  f"{ponr:>22s} {best_row['final_atp']:>10.3f}")

    # Reverse surgery: how long must treatment run to have lasting effect?
    print("\nTREATMENT DURATION THRESHOLD (reverse surgery: intervention → stop)")
    print(f"{'Patient':20s} {'Intervention':20s} {'Min duration for lasting benefit':>32s}")
    print("-" * 74)

    for pat_id, pat_info in PATIENTS.items():
        baseline_row = [r for r in results if r["patient"] == pat_id
                        and r["direction"] == "baseline"][0]
        baseline_atp = baseline_row["final_atp"]

        for intv_id, intv_info in INTERVENTIONS.items():
            reverse_rows = [r for r in results
                           if r["patient"] == pat_id
                           and r["intervention"] == intv_id
                           and r["direction"] == "reverse"]
            reverse_rows.sort(key=lambda r: r["switch_year"])

            # Find minimum treatment duration where stopping still beats baseline
            min_duration = None
            for row in reverse_rows:
                if row["final_atp"] > baseline_atp + 0.01:
                    min_duration = row["switch_year"]
                    break

            dur_str = f"{min_duration} years" if min_duration is not None else "permanent needed"
            print(f"{pat_info['label']:20s} {intv_info['label']:20s} {dur_str:>32s}")

    # Save
    output = {
        "experiment": "causal_surgery",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "n_results": len(results),
        "patients": {k: v["label"] for k, v in PATIENTS.items()},
        "interventions": {k: v["label"] for k, v in INTERVENTIONS.items()},
        "switch_times": SWITCH_TIMES,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
