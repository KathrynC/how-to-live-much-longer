"""Side-by-side visualization for transplant-vs-no-transplant optimization runs.

Reads:
  artifacts/transplant_vs_no_transplant_100runs_2026-02-18.json

Writes:
  artifacts/transplant_vs_no_transplant_side_by_side.png
  artifacts/transplant_vs_no_transplant_deltas.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    src = artifacts / "transplant_vs_no_transplant_100runs_2026-02-18.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing artifact: {src}")

    data = json.loads(src.read_text())
    with_tx = data["with_transplant"]
    without_tx = data["without_transplant"]
    deltas = data["deltas_with_minus_without"]

    # ---- Figure 1: side-by-side panels ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Panel A: mean optimal regimen parameters
    params = list(with_tx["mean_optimal_params"].keys())
    a = np.array([with_tx["mean_optimal_params"][k] for k in params], dtype=float)
    b = np.array([without_tx["mean_optimal_params"][k] for k in params], dtype=float)

    x = np.arange(len(params))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, a, width=w, label="With Transplant", color="#2ca02c")
    ax.bar(x + w / 2, b, width=w, label="No Transplant", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean Optimal Dose")
    ax.set_title("Optimal Regimen (100 Patients)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    # Panel B: key outcome means
    metric_keys = [
        "fitness",
        "final_atp",
        "final_het",
        "total_dose",
    ]
    metric_labels = [
        "Fitness",
        "Final ATP",
        "Final Heteroplasmy",
        "Total Dose",
    ]

    ma = np.array([with_tx[k]["mean"] for k in metric_keys], dtype=float)
    mb = np.array([without_tx[k]["mean"] for k in metric_keys], dtype=float)

    x2 = np.arange(len(metric_keys))
    ax = axes[1]
    ax.bar(x2 - w / 2, ma, width=w, label="With Transplant", color="#2ca02c")
    ax.bar(x2 + w / 2, mb, width=w, label="No Transplant", color="#1f77b4")
    ax.set_xticks(x2)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_title("Outcome Means")
    ax.grid(axis="y", alpha=0.25)

    # annotate deltas (with - without)
    delta_map = {
        "Fitness": deltas["fitness_mean_delta"],
        "Final ATP": deltas["final_atp_mean_delta"],
        "Final Heteroplasmy": deltas["final_het_mean_delta"],
        "Total Dose": deltas["dose_mean_delta"],
    }
    for i, label in enumerate(metric_labels):
        ymax = max(ma[i], mb[i])
        ax.text(i, ymax * 1.03 if ymax > 0 else 0.03,
                f"Δ={delta_map[label]:+.3f}",
                ha="center", va="bottom", fontsize=9)

    fig.suptitle("Transplant vs No-Transplant: Side-by-Side Optimization Results", fontsize=14, fontweight="bold")

    out1 = artifacts / "transplant_vs_no_transplant_side_by_side.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: explicit delta bars ----
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    dkeys = [
        "fitness_mean_delta",
        "final_atp_mean_delta",
        "final_het_mean_delta",
        "atp_benefit_mean_delta",
        "het_benefit_mean_delta",
        "dose_mean_delta",
    ]
    dlabels = [
        "Fitness",
        "Final ATP",
        "Final Heteroplasmy",
        "ATP Benefit",
        "Het Benefit",
        "Total Dose",
    ]
    vals = np.array([deltas[k] for k in dkeys], dtype=float)

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    bars = ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(dlabels, rotation=20, ha="right")
    ax.set_ylabel("Delta (With - Without)")
    ax.set_title("Effect of Allowing Transplantation")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, vals):
        y = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, y + (0.01 if y >= 0 else -0.01),
                f"{v:+.3f}", ha="center", va=("bottom" if y >= 0 else "top"), fontsize=9)

    out2 = artifacts / "transplant_vs_no_transplant_deltas.png"
    fig.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
