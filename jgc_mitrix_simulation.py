#!/usr/bin/env python3
"""Simulate John G. Cramer's Mitrix Bio mitochondrial transplant treatment.

Based on: Marcus, A.D. (2026, Feb 20). "Longevity Treatments for a 91-Year-Old?
A Bold Bet in Silicon Valley's Immortality Race." The Information (Weekend).

Three analyses:
  1. JGC's case: no treatment vs transplant protocol vs full cocktail
  2. "Battery first" hypothesis: Yamanaka alone vs transplant-first vs combination
  3. Age sweep: when does "fix the battery first" become essential?

Usage:
    python jgc_mitrix_simulation.py              # Run all analyses, save plots
    python jgc_mitrix_simulation.py --no-plots   # Print results only
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator import simulate
from analytics import compute_all
from constants import DEFAULT_INTERVENTION


# ── Output directory ─────────────────────────────────────────────────────────

OUTPUT_DIR = "output/jgc_mitrix"


# ── Patient profile ──────────────────────────────────────────────────────────

# John G. Cramer, age 91, February 2026
# From the article: "His hearing isn't great, and he has already had cataract
# surgery and a knee replacement. A cardiologist monitors his irregular
# heartbeat." But: no debilitating illnesses, active researcher, traveling
# cross-country for treatments, planning to celebrate with Scotch.
#
# This translates to: moderate age-related damage but no cliff crisis,
# low-normal inflammation, average genetic vulnerability.

JGC_PATIENT = {
    "baseline_age": 91.0,
    "baseline_heteroplasmy": 0.30,    # moderate damage, well below cliff
    "baseline_nad_level": 0.4,        # age 91 → significant NAD decline
    "genetic_vulnerability": 1.0,     # no known haplogroup risk
    "metabolic_demand": 1.0,          # average somatic
    "inflammation_level": 0.25,       # mild inflammaging (cardiac monitoring)
}

# ── Intervention protocols ───────────────────────────────────────────────────

NO_TREATMENT = dict(DEFAULT_INTERVENTION)

# Mitrix protocol: transplant-focused with conservative support
# Cramer receives IV mitochondria from 26-year-old granddaughter Selena Shea
MITRIX_TRANSPLANT = {
    "rapamycin_dose": 0.25,           # mild mitophagy support
    "nad_supplement": 0.5,            # moderate NAD restoration
    "senolytic_dose": 0.25,           # mild senescent clearance
    "yamanaka_intensity": 0.0,        # NOT part of Mitrix's approach
    "transplant_rate": 0.75,          # primary therapy
    "exercise_level": 0.5,            # moderate activity
}

# Full cocktail: transplant + everything except Yamanaka
FULL_COCKTAIL = {
    "rapamycin_dose": 0.75,
    "nad_supplement": 0.75,
    "senolytic_dose": 0.5,
    "yamanaka_intensity": 0.0,
    "transplant_rate": 0.75,
    "exercise_level": 0.5,
}

# "Billionaire" approach: Yamanaka reprogramming only (Retro/Altos/NewLimit)
YAMANAKA_ONLY = {
    "rapamycin_dose": 0.0,
    "nad_supplement": 0.0,
    "senolytic_dose": 0.0,
    "yamanaka_intensity": 0.75,       # aggressive reprogramming
    "transplant_rate": 0.0,
    "exercise_level": 0.0,
}

# Battery first: transplant to restore energy, THEN Yamanaka
BATTERY_FIRST = {
    "rapamycin_dose": 0.5,
    "nad_supplement": 0.75,
    "senolytic_dose": 0.25,
    "yamanaka_intensity": 0.5,        # moderate Yamanaka (after energy restored)
    "transplant_rate": 0.75,          # transplant supports the energy cost
    "exercise_level": 0.5,
}

# Conservative no-transplant: what the big companies might offer someday
CONSERVATIVE_NO_TRANSPLANT = {
    "rapamycin_dose": 0.75,
    "nad_supplement": 0.75,
    "senolytic_dose": 0.5,
    "yamanaka_intensity": 0.0,
    "transplant_rate": 0.0,
    "exercise_level": 0.75,
}


# ── Analysis 1: JGC's case ──────────────────────────────────────────────────

def run_jgc_case(sim_years=15):
    """Simulate JGC with no treatment vs Mitrix transplant vs full cocktail."""
    print("\n" + "=" * 72)
    print("ANALYSIS 1: John G. Cramer's Mitochondrial Transplant Treatment")
    print(f"Patient: 91-year-old male, good general health")
    print(f"Simulation horizon: {sim_years} years (ages 91-{91 + sim_years})")
    print("=" * 72)

    protocols = {
        "No treatment": NO_TREATMENT,
        "Mitrix transplant": MITRIX_TRANSPLANT,
        "Full cocktail": FULL_COCKTAIL,
        "Conservative (no transplant)": CONSERVATIVE_NO_TRANSPLANT,
    }

    results = {}
    baseline_result = simulate(
        intervention=NO_TREATMENT, patient=JGC_PATIENT, sim_years=sim_years
    )

    for name, protocol in protocols.items():
        result = simulate(
            intervention=protocol, patient=JGC_PATIENT, sim_years=sim_years
        )
        analytics = compute_all(result, baseline_result=baseline_result)
        results[name] = {"result": result, "analytics": analytics}

    # Print summary table
    print(f"\n{'Protocol':<30} {'ATP final':>10} {'Het final':>10} "
          f"{'Del het':>10} {'Cliff dist':>10} {'Sen frac':>10}")
    print("-" * 82)
    for name, data in results.items():
        r = data["result"]
        atp = r["states"][-1, 2]
        het = r["heteroplasmy"][-1]
        del_het = r["deletion_heteroplasmy"][-1]
        cliff_dist = 0.50 - del_het
        sen = r["states"][-1, 5]
        print(f"{name:<30} {atp:10.4f} {het:10.4f} {del_het:10.4f} "
              f"{cliff_dist:10.4f} {sen:10.4f}")

    # ATP trajectory at key timepoints
    print(f"\n{'Protocol':<30} {'Year 0':>8} {'Year 5':>8} "
          f"{'Year 10':>8} {'Year 15':>8}")
    print("-" * 66)
    for name, data in results.items():
        r = data["result"]
        time = r["time"]
        atp = r["states"][:, 2]
        vals = []
        for yr in [0, 5, 10, 15]:
            if yr <= sim_years:
                idx = int(yr / sim_years * (len(time) - 1))
                vals.append(f"{atp[idx]:8.4f}")
            else:
                vals.append(f"{'--':>8}")
        print(f"{name:<30} {'  '.join(vals)}")

    return results


# ── Analysis 2: Battery first hypothesis ─────────────────────────────────────

def run_battery_first(sim_years=15):
    """Test: does fixing mitochondria first make Yamanaka more effective?"""
    print("\n" + "=" * 72)
    print('ANALYSIS 2: "Battery First" Hypothesis')
    print('Mitrix thesis: "If the battery is not working, nothing works."')
    print("Can Yamanaka reprogramming work without mitochondrial energy?")
    print("=" * 72)

    protocols = {
        "No treatment": NO_TREATMENT,
        "Yamanaka only (billionaires)": YAMANAKA_ONLY,
        "Transplant only (Mitrix)": MITRIX_TRANSPLANT,
        "Battery first (transplant+Yamanaka)": BATTERY_FIRST,
    }

    baseline_result = simulate(
        intervention=NO_TREATMENT, patient=JGC_PATIENT, sim_years=sim_years
    )

    results = {}
    for name, protocol in protocols.items():
        result = simulate(
            intervention=protocol, patient=JGC_PATIENT, sim_years=sim_years
        )
        analytics = compute_all(result, baseline_result=baseline_result)
        results[name] = {"result": result, "analytics": analytics}

    print(f"\n{'Protocol':<35} {'ATP final':>10} {'ATP benefit':>12} "
          f"{'Het benefit':>12} {'Del het':>10}")
    print("-" * 81)
    for name, data in results.items():
        r = data["result"]
        a = data["analytics"]
        atp = r["states"][-1, 2]
        atp_ben = a["intervention"].get("atp_benefit", 0.0)
        het_ben = a["intervention"].get("het_benefit", 0.0)
        del_het = r["deletion_heteroplasmy"][-1]
        print(f"{name:<35} {atp:10.4f} {atp_ben:12.4f} "
              f"{het_ben:12.4f} {del_het:10.4f}")

    # The key question: does Yamanaka HURT at age 91?
    yam_atp = results["Yamanaka only (billionaires)"]["result"]["states"][-1, 2]
    no_rx_atp = results["No treatment"]["result"]["states"][-1, 2]
    transplant_atp = results["Transplant only (Mitrix)"]["result"]["states"][-1, 2]
    combo_atp = results["Battery first (transplant+Yamanaka)"]["result"]["states"][-1, 2]

    print(f"\nKey finding:")
    if yam_atp < no_rx_atp:
        print(f"  Yamanaka HARMS at age 91: ATP {yam_atp:.4f} < no treatment {no_rx_atp:.4f}")
        print(f"  Energy cost of reprogramming exceeds benefit when battery is weak.")
    else:
        print(f"  Yamanaka helps at age 91: ATP {yam_atp:.4f} > no treatment {no_rx_atp:.4f}")

    if combo_atp > transplant_atp:
        print(f"  Battery-first combo outperforms transplant alone: {combo_atp:.4f} > {transplant_atp:.4f}")
        print(f"  Restoring energy first lets Yamanaka contribute positively.")
    else:
        print(f"  Transplant alone is sufficient: {transplant_atp:.4f} >= {combo_atp:.4f}")

    return results


# ── Analysis 3: Age sweep ────────────────────────────────────────────────────

def run_age_sweep(sim_years=15):
    """At what age does the battery-first strategy become essential?"""
    print("\n" + "=" * 72)
    print("ANALYSIS 3: Age Sweep — When Does Battery-First Become Essential?")
    print("Comparing Yamanaka vs Transplant vs Combination across ages")
    print("=" * 72)

    ages = [40, 50, 60, 70, 80, 90]

    # Age-appropriate patient profiles
    def patient_for_age(age):
        age_frac = (age - 20) / 70.0  # 0 at 20, 1 at 90
        return {
            "baseline_age": float(age),
            "baseline_heteroplasmy": 0.05 + 0.35 * age_frac ** 1.2,
            "baseline_nad_level": max(0.3, 0.95 - 0.55 * age_frac),
            "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0,
            "inflammation_level": 0.05 + 0.30 * age_frac,
        }

    strategies = {
        "No treatment": NO_TREATMENT,
        "Yamanaka only": YAMANAKA_ONLY,
        "Transplant only": MITRIX_TRANSPLANT,
        "Battery first": BATTERY_FIRST,
    }

    sweep_data = {s: {"ages": [], "atp_final": [], "het_final": []}
                  for s in strategies}

    print(f"\n{'Age':>4}  {'No treatment':>14} {'Yamanaka':>14} "
          f"{'Transplant':>14} {'Battery first':>14}")
    print("-" * 76)

    for age in ages:
        patient = patient_for_age(age)
        baseline = simulate(
            intervention=NO_TREATMENT, patient=patient, sim_years=sim_years
        )
        row = [f"{age:4d}"]
        for sname, protocol in strategies.items():
            result = simulate(
                intervention=protocol, patient=patient, sim_years=sim_years
            )
            atp = result["states"][-1, 2]
            sweep_data[sname]["ages"].append(age)
            sweep_data[sname]["atp_final"].append(atp)
            sweep_data[sname]["het_final"].append(result["heteroplasmy"][-1])
            row.append(f"{atp:14.4f}")
        print("  ".join(row))

    # Find crossover: when does Yamanaka start hurting?
    yam_vals = sweep_data["Yamanaka only"]["atp_final"]
    no_rx_vals = sweep_data["No treatment"]["atp_final"]
    print(f"\nYamanaka vs no treatment (positive = helps, negative = harms):")
    for i, age in enumerate(ages):
        diff = yam_vals[i] - no_rx_vals[i]
        marker = "HARMS" if diff < 0 else "helps"
        print(f"  Age {age}: {diff:+.4f} ({marker})")

    return sweep_data, ages


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_jgc_trajectories(results, sim_years=15):
    """4-panel comparison: ATP, deletion het, NAD, senescence over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "John G. Cramer (age 91): Mitochondrial Transplant Simulation",
        fontsize=14, fontweight="bold",
    )

    colors = {
        "No treatment": "#95a5a6",
        "Mitrix transplant": "#e74c3c",
        "Full cocktail": "#2ecc71",
        "Conservative (no transplant)": "#3498db",
    }
    styles = {
        "No treatment": "--",
        "Mitrix transplant": "-",
        "Full cocktail": "-",
        "Conservative (no transplant)": "-.",
    }

    panels = [
        (axes[0, 0], 2, "ATP (MU/day)", "Energy Production"),
        (axes[0, 1], "del_het", "Deletion Heteroplasmy", "Distance from Cliff"),
        (axes[1, 0], 4, "NAD+ Level", "Cofactor Availability"),
        (axes[1, 1], 5, "Senescent Fraction", "Cellular Aging"),
    ]

    for ax, state_idx, ylabel, title in panels:
        for name, data in results.items():
            r = data["result"]
            time = r["time"]
            if state_idx == "del_het":
                y = r["deletion_heteroplasmy"]
            else:
                y = r["states"][:, state_idx]
            ax.plot(time, y, color=colors[name], linestyle=styles[name],
                    linewidth=2, label=name)

        if state_idx == "del_het":
            ax.axhline(y=0.50, color="#e74c3c", linestyle=":", alpha=0.5,
                       label="Cliff (0.50)")

        ax.set_xlabel("Years from start of treatment")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Add age axis labels on top
    for ax in axes[0]:
        ax2 = ax.twiny()
        ticks = [0, 5, 10, 15]
        ticks = [t for t in ticks if t <= sim_years]
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"Age {91 + t}" for t in ticks], fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "jgc_treatment_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_battery_first(results, sim_years=15):
    """Battery-first hypothesis: ATP trajectories for 4 strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        '"If the Battery Is Not Working, Nothing Works"\n'
        'Testing the Mitrix Hypothesis at Age 91',
        fontsize=13, fontweight="bold",
    )

    colors = {
        "No treatment": "#95a5a6",
        "Yamanaka only (billionaires)": "#f39c12",
        "Transplant only (Mitrix)": "#e74c3c",
        "Battery first (transplant+Yamanaka)": "#2ecc71",
    }

    # Left: ATP trajectories
    for name, data in results.items():
        r = data["result"]
        ax1.plot(r["time"], r["states"][:, 2], color=colors[name],
                 linewidth=2, label=name)
    ax1.set_xlabel("Years from start of treatment")
    ax1.set_ylabel("ATP (MU/day)")
    ax1.set_title("Energy Production Over Time")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: deletion heteroplasmy
    for name, data in results.items():
        r = data["result"]
        ax2.plot(r["time"], r["deletion_heteroplasmy"], color=colors[name],
                 linewidth=2, label=name)
    ax2.axhline(y=0.50, color="#e74c3c", linestyle=":", alpha=0.5,
                label="Cliff (0.50)")
    ax2.set_xlabel("Years from start of treatment")
    ax2.set_ylabel("Deletion Heteroplasmy")
    ax2.set_title("Mitochondrial Damage Over Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(OUTPUT_DIR, "battery_first_hypothesis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_age_sweep(sweep_data, ages):
    """Age sweep: final ATP for each strategy across ages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "When Does Battery-First Become Essential?\n"
        "Final ATP After 15 Years by Starting Age",
        fontsize=13, fontweight="bold",
    )

    colors = {
        "No treatment": "#95a5a6",
        "Yamanaka only": "#f39c12",
        "Transplant only": "#e74c3c",
        "Battery first": "#2ecc71",
    }

    # Left: absolute ATP
    for sname, sdata in sweep_data.items():
        ax1.plot(ages, sdata["atp_final"], "o-", color=colors[sname],
                 linewidth=2, markersize=8, label=sname)
    ax1.axvline(x=91, color="#333", linestyle=":", alpha=0.4, label="JGC (91)")
    ax1.set_xlabel("Starting Age")
    ax1.set_ylabel("Final ATP (MU/day)")
    ax1.set_title("Absolute Energy After 15 Years")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Yamanaka benefit (vs no treatment)
    yam_benefit = [y - n for y, n in zip(
        sweep_data["Yamanaka only"]["atp_final"],
        sweep_data["No treatment"]["atp_final"])]
    transplant_benefit = [t - n for t, n in zip(
        sweep_data["Transplant only"]["atp_final"],
        sweep_data["No treatment"]["atp_final"])]
    combo_benefit = [c - n for c, n in zip(
        sweep_data["Battery first"]["atp_final"],
        sweep_data["No treatment"]["atp_final"])]

    ax2.bar([a - 1.5 for a in ages], yam_benefit, width=3,
            color="#f39c12", alpha=0.8, label="Yamanaka only")
    ax2.bar([a + 1.5 for a in ages], transplant_benefit, width=3,
            color="#e74c3c", alpha=0.8, label="Transplant only")
    ax2.plot(ages, combo_benefit, "s-", color="#2ecc71",
             linewidth=2, markersize=8, label="Battery first", zorder=5)
    ax2.axhline(y=0, color="#333", linewidth=0.8)
    ax2.axvline(x=91, color="#333", linestyle=":", alpha=0.4)
    ax2.set_xlabel("Starting Age")
    ax2.set_ylabel("ATP Benefit vs No Treatment")
    ax2.set_title("Treatment Benefit by Age")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = os.path.join(OUTPUT_DIR, "age_sweep_battery_first.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── JSON export ──────────────────────────────────────────────────────────────

def _downsample(arr, n=300):
    """Downsample a numpy array to n points for JSON export."""
    if len(arr) <= n:
        return arr.tolist() if hasattr(arr, "tolist") else list(arr)
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[indices].tolist() if hasattr(arr, "tolist") else [arr[i] for i in indices]


def export_json(jgc_results, battery_results, sweep_data, ages, sim_years=15):
    """Export all simulation data as a single JSON artifact."""
    import json

    payload = {
        "meta": {
            "patient": JGC_PATIENT,
            "sim_years": sim_years,
            "date": "2026-02-20",
            "source": "Marcus, A.D., The Information, 2026-02-20",
        },
        "analysis1": {},
        "analysis2": {},
        "analysis3": {"ages": ages, "strategies": {}},
    }

    # Analysis 1 & 2: trajectory data
    for label, results, key in [
        ("analysis1", jgc_results, "analysis1"),
        ("analysis2", battery_results, "analysis2"),
    ]:
        for name, data in results.items():
            r = data["result"]
            payload[key][name] = {
                "time": _downsample(r["time"]),
                "atp": _downsample(r["states"][:, 2]),
                "deletion_het": _downsample(r["deletion_heteroplasmy"]),
                "heteroplasmy": _downsample(r["heteroplasmy"]),
                "nad": _downsample(r["states"][:, 4]),
                "senescent": _downsample(r["states"][:, 5]),
                "ros": _downsample(r["states"][:, 3]),
                "n_healthy": _downsample(r["states"][:, 0]),
                "final_atp": float(r["states"][-1, 2]),
                "final_het": float(r["heteroplasmy"][-1]),
                "final_del_het": float(r["deletion_heteroplasmy"][-1]),
            }

    # Analysis 3: age sweep
    for sname, sdata in sweep_data.items():
        payload["analysis3"]["strategies"][sname] = {
            "atp_final": sdata["atp_final"],
            "het_final": sdata["het_final"],
        }

    path = os.path.join("artifacts", "jgc_mitrix_simulation.json")
    os.makedirs("artifacts", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {path}")
    return payload


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate JGC's Mitrix Bio mitochondrial transplant treatment"
    )
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--sim-years", type=int, default=15,
                        help="Simulation horizon in years (default: 15)")
    args = parser.parse_args()

    sim_years = args.sim_years

    if not args.no_plots:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Analysis 1: JGC's case
    jgc_results = run_jgc_case(sim_years=sim_years)
    if not args.no_plots:
        plot_jgc_trajectories(jgc_results, sim_years=sim_years)

    # Analysis 2: Battery first hypothesis
    battery_results = run_battery_first(sim_years=sim_years)
    if not args.no_plots:
        plot_battery_first(battery_results, sim_years=sim_years)

    # Analysis 3: Age sweep
    sweep_data, ages = run_age_sweep(sim_years=sim_years)
    if not args.no_plots:
        plot_age_sweep(sweep_data, ages)

    # Export JSON for D3 visualization
    export_json(jgc_results, battery_results, sweep_data, ages,
                sim_years=sim_years)

    print("\n" + "=" * 72)
    print("SIMULATION COMPLETE")
    if not args.no_plots:
        print(f"Plots saved to {OUTPUT_DIR}/")
    print(f"JSON saved to artifacts/jgc_mitrix_simulation.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
