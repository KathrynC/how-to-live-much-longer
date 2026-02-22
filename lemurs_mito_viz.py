"""Visualization for the LEMURS->mitochondrial aging integration.

Side-by-side panels showing LEMURS semester trajectories (top row) and
mitochondrial trajectories (bottom row), with archetype comparisons.

Usage:
    from lemurs_mito_viz import generate_all_plots
    generate_all_plots()           # generates all plots to output/lemurs_mito/

    from lemurs_mito_viz import plot_lemurs_comparison
    plot_lemurs_comparison()       # archetype overlay comparison

Uses Agg backend (headless).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lemurs_bridge import (  # noqa: E402
    LEMURSDisturbance,
    lemurs_trajectory,
    LEMURS_STUDENT_ARCHETYPES,
    LEMURS_DEFAULT_PATIENT,
)
from lemurs_mito_scenarios import LEMURS_PROTOCOLS  # noqa: E402
from disturbances import simulate_with_disturbances  # noqa: E402
from simulator import simulate  # noqa: E402

# Young patient profile for LEMURS integration (college student, age 18).
# Low baseline heteroplasmy and inflammation, high NAD — typical healthy young adult.
_YOUNG_PATIENT: dict[str, float] = {
    "baseline_age": 18.0,
    "baseline_heteroplasmy": 0.05,
    "baseline_nad_level": 0.95,
    "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0,
    "inflammation_level": 0.05,
}


def plot_lemurs_mito_trajectory(
    lemurs_curves: dict,
    mito_result: dict,
    title: str,
    output_path: str,
) -> None:
    """2x3 panel plot: LEMURS semester dynamics (top) + mito response (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    lemurs_t = lemurs_curves["times"]
    mito_t = mito_result["time"]
    mito_s = mito_result["states"]

    # Top row: LEMURS semester dynamics
    axes[0, 0].plot(lemurs_t, lemurs_curves["tst"], color="C0", label="TST")
    axes[0, 0].set_title("LEMURS: Total Sleep Time")
    axes[0, 0].set_xlabel("Week")
    axes[0, 0].set_ylabel("Hours")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(lemurs_t, lemurs_curves["pss"], color="C3", label="PSS")
    axes[0, 1].set_title("LEMURS: Perceived Stress (0-40)")
    axes[0, 1].set_xlabel("Week")
    axes[0, 1].set_ylabel("PSS Score")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(lemurs_t, lemurs_curves["gad7"], color="C1", label="GAD-7")
    axes[0, 2].axhline(y=10.0, color="red", linestyle=":", alpha=0.5, label="Clinical (10)")
    axes[0, 2].set_title("LEMURS: GAD-7 Anxiety (0-21)")
    axes[0, 2].set_xlabel("Week")
    axes[0, 2].set_ylabel("GAD-7 Score")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom row: Mito response
    het = mito_result["heteroplasmy"]
    axes[1, 0].plot(mito_t, het, color="C1", label="Heteroplasmy")
    axes[1, 0].axhline(y=0.50, color="red", linestyle=":", alpha=0.5, label="Cliff (0.50)")
    axes[1, 0].set_title("Mito: Heteroplasmy (Deletion)")
    axes[1, 0].set_xlabel("Time (years)")
    axes[1, 0].set_ylabel("Fraction damaged")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(mito_t, mito_s[:, 2], color="C0", label="ATP")
    axes[1, 1].set_title("Mito: ATP Production")
    axes[1, 1].set_xlabel("Time (years)")
    axes[1, 1].set_ylabel("MU/day")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(mito_t, mito_s[:, 3], color="C4", label="ROS")
    axes[1, 2].plot(mito_t, mito_s[:, 5], color="C8", label="Senescence", linestyle="--")
    axes[1, 2].set_title("Mito: ROS & Senescence")
    axes[1, 2].set_xlabel("Time (years)")
    axes[1, 2].set_ylabel("Normalized")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_lemurs_comparison(
    output_path: str = "output/lemurs_mito/archetype_comparison.png",
) -> None:
    """Overlay ATP/het trajectories for all student archetypes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "LEMURS->Mito: Student Archetype Comparison",
        fontsize=14, fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(LEMURS_STUDENT_ARCHETYPES)))

    for i, archetype in enumerate(LEMURS_STUDENT_ARCHETYPES):
        name = archetype["name"]
        d = LEMURSDisturbance(
            lemurs_patient=archetype["patient"],
            lemurs_intervention=archetype.get("intervention"),
            label=f"lemurs_{name}",
        )
        result = simulate_with_disturbances(
            patient=_YOUNG_PATIENT,
            disturbances=[d],
        )
        t = result["time"]
        atp = result["states"][:, 2]
        het = result["heteroplasmy"]

        axes[0].plot(t, atp, label=name, color=colors[i], linewidth=1.5)
        axes[1].plot(t, het, label=name, color=colors[i], linewidth=1.5)

    axes[0].set_title("ATP Production Over Time")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("ATP (MU/day)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Heteroplasmy Over Time")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("Heteroplasmy")
    axes[1].axhline(y=0.50, color="red", linestyle=":", alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_all_plots(output_dir: str = "output/lemurs_mito") -> None:
    """Generate all LEMURS->mito visualizations.

    1. Individual trajectory plot for each archetype.
    2. Archetype comparison overlay.
    3. Intervention comparison (no support vs full support for vulnerable student).
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Individual trajectory plots for each archetype --------------------------
    for archetype in LEMURS_STUDENT_ARCHETYPES:
        name = archetype["name"]
        desc = archetype["description"]

        lemurs_curves = lemurs_trajectory(
            lemurs_patient=archetype["patient"],
            lemurs_intervention=archetype.get("intervention"),
        )

        d = LEMURSDisturbance(
            lemurs_patient=archetype["patient"],
            lemurs_intervention=archetype.get("intervention"),
            label=f"lemurs_{name}",
        )
        mito_result = simulate_with_disturbances(
            patient=_YOUNG_PATIENT,
            disturbances=[d],
        )

        plot_lemurs_mito_trajectory(
            lemurs_curves,
            mito_result,
            f"{name}: {desc}",
            os.path.join(output_dir, f"lemurs_mito_{name}.png"),
        )

    # -- Archetype comparison overlay --------------------------------------------
    plot_lemurs_comparison(
        output_path=os.path.join(output_dir, "archetype_comparison.png"),
    )

    # -- Intervention comparison: vulnerable student, no support vs full support --
    vulnerable = next(
        a for a in LEMURS_STUDENT_ARCHETYPES
        if a["name"] == "vulnerable_female"
    )
    vulnerable_patient = vulnerable["patient"]

    protocols = {
        "No college stress": None,
        "No support": LEMURS_PROTOCOLS["no_treatment"],
        "Full support": LEMURS_PROTOCOLS["full_support"],
    }

    baseline_mito = simulate(patient=_YOUNG_PATIENT)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Vulnerable Student: LEMURS Interventions -> Mitochondrial Impact",
        fontsize=14, fontweight="bold",
    )

    plot_colors = ["gray", "C3", "C2"]

    final_hets = []
    for i, (label, lemurs_int) in enumerate(protocols.items()):
        if label == "No college stress":
            result = baseline_mito
        else:
            d = LEMURSDisturbance(
                lemurs_patient=vulnerable_patient,
                lemurs_intervention=lemurs_int,
                label=f"lemurs_vulnerable_{label}",
            )
            result = simulate_with_disturbances(
                patient=_YOUNG_PATIENT,
                disturbances=[d],
            )

        t = result["time"]
        het = result["heteroplasmy"]
        atp = result["states"][:, 2]
        final_hets.append(het[-1])

        axes[0].plot(t, het, label=label, color=plot_colors[i],
                     linewidth=2 if label == "No college stress" else 1.5)
        axes[1].plot(t, atp, label=label, color=plot_colors[i],
                     linewidth=2 if label == "No college stress" else 1.5)

    # Bar chart of final heteroplasmy
    axes[2].bar(list(protocols.keys()), final_hets, color=plot_colors)
    axes[2].set_ylabel("Final Heteroplasmy")
    axes[2].set_title("30-Year Mitochondrial Damage")
    axes[2].axhline(y=0.50, color="red", linestyle=":", alpha=0.5)

    axes[0].set_title("Heteroplasmy Over Time")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Heteroplasmy")
    axes[0].axhline(y=0.50, color="red", linestyle=":", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("ATP Production Over Time")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("ATP (MU/day)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    intervention_path = os.path.join(output_dir, "intervention_comparison.png")
    plt.tight_layout()
    fig.savefig(intervention_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {intervention_path}")


if __name__ == "__main__":
    print("Generating LEMURS->mito visualizations...")
    generate_all_plots()
    print("Done.")
