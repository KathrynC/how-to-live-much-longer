"""Visualization for the grief->mitochondrial aging integration.

Side-by-side panels showing grief trajectories (left) and mitochondrial
trajectories (right), with the 5 mapping channels highlighted.

Usage:
    from grief_mito_viz import plot_all_grief_scenarios
    plot_all_grief_scenarios()           # generates all plots to output/grief_mito/

    from grief_mito_viz import plot_intervention_comparison
    plot_intervention_comparison(
        grief_patient={"B": 0.8, "M": 0.8, "age": 65.0, "E_ctx": 0.2},
        output_path="output/grief_mito/comparison.png",
    )

Uses Agg backend (headless).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from grief_bridge import (  # noqa: E402
    GriefDisturbance,
    grief_trajectory,
    GRIEF_CLINICAL_SEEDS,
)
from grief_mito_scenarios import GRIEF_PROTOCOLS  # noqa: E402
from disturbances import simulate_with_disturbances  # noqa: E402
from simulator import simulate  # noqa: E402


def plot_grief_mito_trajectory(
    grief_curves: dict,
    mito_result: dict,
    title: str,
    output_path: str,
) -> None:
    """2x4 panel plot: grief (left column) + mito (right column)."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    grief_t = grief_curves["times"]
    mito_t = mito_result["time"]
    mito_s = mito_result["states"]

    # Left column: Grief signals
    axes[0, 0].plot(grief_t, grief_curves["infl"], color="C5", label="Inflammation")
    axes[0, 0].set_title("Grief: Inflammation")
    axes[0, 0].set_ylabel("Normalized")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(grief_t, grief_curves["cort"], color="C3", label="Cortisol")
    axes[1, 0].set_title("Grief: Cortisol")
    axes[1, 0].set_ylabel("Normalized")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(grief_t, grief_curves["sns"], color="C2", label="SNS")
    axes[2, 0].plot(grief_t, grief_curves["slp"], color="C9", label="Sleep", linestyle="--")
    axes[2, 0].set_title("Grief: SNS & Sleep")
    axes[2, 0].set_ylabel("Normalized")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    axes[3, 0].plot(grief_t, grief_curves["cvd_risk"], color="red", label="CVD risk")
    axes[3, 0].set_title("Grief: CVD Risk (cumulative)")
    axes[3, 0].set_xlabel("Time (years)")
    axes[3, 0].set_ylabel("Cumulative")
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].grid(True, alpha=0.3)

    # Right column: Mito response
    het = mito_result["heteroplasmy"]
    axes[0, 1].plot(mito_t, het, color="C1", label="Heteroplasmy")
    axes[0, 1].axhline(y=0.7, color="red", linestyle=":", alpha=0.5, label="Cliff (0.70)")
    axes[0, 1].set_title("Mito: Heteroplasmy")
    axes[0, 1].set_ylabel("Fraction damaged")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(mito_t, mito_s[:, 2], color="C0", label="ATP")
    axes[1, 1].set_title("Mito: ATP Production")
    axes[1, 1].set_ylabel("MU/day")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 1].plot(mito_t, mito_s[:, 3], color="C4", label="ROS")
    axes[2, 1].plot(mito_t, mito_s[:, 5], color="C8", label="Senescence", linestyle="--")
    axes[2, 1].set_title("Mito: ROS & Senescence")
    axes[2, 1].set_ylabel("Normalized")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    axes[3, 1].plot(mito_t, mito_s[:, 4], color="C6", label="NAD+")
    axes[3, 1].plot(mito_t, mito_s[:, 6], color="C7", label="Membrane Potential", linestyle="--")
    axes[3, 1].set_title("Mito: NAD+ & Membrane Potential")
    axes[3, 1].set_xlabel("Time (years)")
    axes[3, 1].set_ylabel("Normalized")
    axes[3, 1].legend(fontsize=8)
    axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_intervention_comparison(
    grief_patient: dict[str, float] | None = None,
    mito_patient: dict[str, float] | None = None,
    output_path: str = "output/grief_mito/intervention_comparison.png",
    title: str = "Grief Interventions -> Mitochondrial Impact",
) -> None:
    """Compare mito outcomes with and without grief interventions."""
    baseline_mito = simulate(patient=mito_patient)

    protocols = {
        "No grief": None,
        "No support": GRIEF_PROTOCOLS["no_grief_support"],
        "Moderate": GRIEF_PROTOCOLS["moderate_support"],
        "Full support": GRIEF_PROTOCOLS["full_grief_support"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = ["gray", "C3", "C1", "C2"]

    for i, (label, grief_int) in enumerate(protocols.items()):
        if label == "No grief":
            result = baseline_mito
            het = result["heteroplasmy"]
            t = result["time"]
            atp = result["states"][:, 2]
        else:
            d = GriefDisturbance(
                grief_patient=grief_patient,
                grief_intervention=grief_int,
            )
            result = simulate_with_disturbances(
                patient=mito_patient,
                disturbances=[d],
            )
            het = result["heteroplasmy"]
            t = result["time"]
            atp = result["states"][:, 2]

        axes[0].plot(t, het, label=label, color=colors[i],
                     linewidth=2 if label == "No grief" else 1.5)
        axes[1].plot(t, atp, label=label, color=colors[i],
                     linewidth=2 if label == "No grief" else 1.5)

    # Bar chart of final heteroplasmy
    final_hets = []
    for label, grief_int in protocols.items():
        if label == "No grief":
            final_hets.append(baseline_mito["heteroplasmy"][-1])
        else:
            d = GriefDisturbance(grief_patient=grief_patient, grief_intervention=grief_int)
            r = simulate_with_disturbances(patient=mito_patient, disturbances=[d])
            final_hets.append(r["heteroplasmy"][-1])
    axes[2].bar(list(protocols.keys()), final_hets, color=colors)
    axes[2].set_ylabel("Final Heteroplasmy")
    axes[2].set_title("30-Year Mitochondrial Damage")
    axes[2].axhline(y=0.7, color="red", linestyle=":", alpha=0.5)

    axes[0].set_title("Heteroplasmy Over Time")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Heteroplasmy")
    axes[0].axhline(y=0.7, color="red", linestyle=":", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("ATP Production Over Time")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("ATP (MU/day)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_grief_scenarios(
    output_dir: str = "output/grief_mito",
    max_scenarios: int | None = None,
) -> None:
    """Generate trajectory plots for all grief clinical scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    seeds = GRIEF_CLINICAL_SEEDS
    if max_scenarios is not None:
        seeds = seeds[:max_scenarios]

    results_no = []
    results_yes = []
    labels = []

    for seed in seeds:
        labels.append(seed["name"])

        # Without support
        d_no = GriefDisturbance(grief_patient=seed["patient"], grief_intervention=None)
        r_no = simulate_with_disturbances(disturbances=[d_no])
        results_no.append(r_no)

        # With support
        d_yes = GriefDisturbance(
            grief_patient=seed["patient"],
            grief_intervention=GRIEF_PROTOCOLS["full_grief_support"],
        )
        r_yes = simulate_with_disturbances(disturbances=[d_yes])
        results_yes.append(r_yes)

        # Individual trajectory
        grief_curves = grief_trajectory(seed["patient"])
        plot_grief_mito_trajectory(
            grief_curves, r_no,
            f"{seed['name']}: {seed['description']}",
            os.path.join(output_dir, f"grief_mito_{seed['name']}.png"),
        )

    # Comparison overlay: final heteroplasmy with/without support
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grief->Mito Impact: All Scenarios", fontsize=14, fontweight="bold")

    x = np.arange(len(labels))
    hets_no = [r["heteroplasmy"][-1] for r in results_no]
    hets_yes = [r["heteroplasmy"][-1] for r in results_yes]

    axes[0].bar(x - 0.2, hets_no, 0.4, label="No support", color="C3")
    axes[0].bar(x + 0.2, hets_yes, 0.4, label="Full support", color="C2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Final Heteroplasmy")
    axes[0].set_title("30-Year Mitochondrial Damage")
    axes[0].axhline(y=0.7, color="red", linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Protection percentage
    protection = [(no - yes) / no * 100 if no > 0 else 0
                  for no, yes in zip(hets_no, hets_yes)]
    axes[1].bar(x, protection, color="C0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Heteroplasmy Reduction (%)")
    axes[1].set_title("Mitochondrial Protection from Grief Interventions")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "grief_mito_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/grief_mito_comparison.png")


if __name__ == "__main__":
    print("Generating grief->mito visualizations...")
    plot_all_grief_scenarios()
    print("Done.")
