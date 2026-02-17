"""Matplotlib visualization for mitochondrial aging simulations.

Uses Agg backend (non-interactive) for headless rendering.
Generates publication-quality figures for:
    - 9-panel trajectory subplots (8 states + heteroplasmy)
    - Cliff curve (ATP vs heteroplasmy)
    - 2D cliff heatmaps
    - Intervention comparison overlays
    - TIQM experiment summaries

Usage:
    from visualize import plot_trajectory, plot_cliff_curve
    plot_trajectory(result, "output/trajectory.png")
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import STATE_NAMES, HETEROPLASMY_CLIFF, DEFAULT_PATIENT


# ── Plot style ───────────────────────────────────────────────────────────────

COLORS = {
    "N_healthy": "#2ecc71",
    "N_deletion": "#e74c3c",
    "ATP": "#3498db",
    "ROS": "#e67e22",
    "NAD": "#9b59b6",
    "Senescent_fraction": "#95a5a6",
    "Membrane_potential": "#1abc9c",
    "N_point": "#8e44ad",
    "heteroplasmy": "#c0392b",
    "cliff_line": "#e74c3c",
}

LABELS = {
    "N_healthy": "Healthy mtDNA",
    "N_deletion": "Deletion mtDNA",
    "ATP": "ATP (MU/day)",
    "ROS": "ROS Level",
    "NAD": "NAD+ Level",
    "Senescent_fraction": "Senescent Fraction",
    "Membrane_potential": "Membrane Potential (ΔΨ)",
    "N_point": "Point mtDNA",
}


def _setup_figure(nrows, ncols, figsize=None):
    """Create a figure with standard styling."""
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("white")
    return fig, axes


# ── 9-panel trajectory plot ──────────────────────────────────────────────────

def plot_trajectory(result, output_path="trajectory.png", title=None):
    """Plot all 8 state variables plus heteroplasmy over time.

    Creates a 9-panel subplot (8 states + heteroplasmy).

    Args:
        result: Dict from simulate().
        output_path: Path to save the figure.
        title: Optional suptitle.
    """
    time = result["time"]
    states = result["states"]
    het = result["heteroplasmy"]

    fig, axes = _setup_figure(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    # Plot 8 state variables
    for i, name in enumerate(STATE_NAMES):
        ax = axes[i]
        color = COLORS.get(name, "#333333")
        ax.plot(time, states[:, i], color=color, linewidth=1.5)
        ax.set_ylabel(LABELS.get(name, name), fontsize=9)
        ax.set_xlabel("Years" if i >= 6 else "")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time[0], time[-1])

    # Plot heteroplasmy in the 9th panel
    ax = axes[8]
    ax.plot(time, het, color=COLORS["heteroplasmy"], linewidth=1.5)
    ax.axhline(y=HETEROPLASMY_CLIFF, color=COLORS["cliff_line"],
               linestyle="--", alpha=0.7, label=f"Cliff ({HETEROPLASMY_CLIFF})")
    ax.set_ylabel("Heteroplasmy", fontsize=9)
    ax.set_xlabel("Years")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Cliff curve ──────────────────────────────────────────────────────────────

def plot_cliff_curve(sweep_result, output_path="cliff_curve.png",
                     cliff_features=None):
    """Plot ATP vs heteroplasmy (the cliff curve).

    Args:
        sweep_result: Dict from sweep_heteroplasmy() with "het_values"
            and "terminal_atp".
        output_path: Path to save.
        cliff_features: Optional dict from extract_cliff_features().
    """
    het = sweep_result["het_values"]
    atp = sweep_result["terminal_atp"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(het, atp, color=COLORS["ATP"], linewidth=2.5, label="Terminal ATP")
    ax.axvline(x=HETEROPLASMY_CLIFF, color=COLORS["cliff_line"],
               linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Cramer Cliff ({HETEROPLASMY_CLIFF})")

    if cliff_features:
        thresh = cliff_features.get("threshold", HETEROPLASMY_CLIFF)
        ax.axvline(x=thresh, color="#f39c12", linestyle=":",
                   alpha=0.7, linewidth=1.5,
                   label=f"Measured Cliff ({thresh:.3f})")

    ax.set_xlabel("Baseline Heteroplasmy", fontsize=12)
    ax.set_ylabel("Terminal ATP (MU/day)", fontsize=12)
    ax.set_title("Heteroplasmy Cliff: ATP Collapse Threshold", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── 2D heatmaps ─────────────────────────────────────────────────────────────

def plot_heatmap(heatmap_result, x_label, y_label, x_key, y_key,
                 output_path="heatmap.png", title=""):
    """Plot a 2D heatmap of terminal ATP.

    Args:
        heatmap_result: Dict with axis arrays and "atp_grid".
        x_label, y_label: Axis labels.
        x_key, y_key: Keys in heatmap_result for the axes.
        output_path: Path to save.
        title: Figure title.
    """
    x_axis = heatmap_result[x_key]
    y_axis = heatmap_result[y_key]
    grid = heatmap_result["atp_grid"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[y_axis[0], y_axis[-1], x_axis[0], x_axis[-1]],
                   cmap="RdYlGn", interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax, label="Terminal ATP (MU/day)")
    ax.set_xlabel(y_label, fontsize=12)
    ax.set_ylabel(x_label, fontsize=12)
    ax.set_title(title or "Terminal ATP Heatmap", fontsize=14)

    # Mark the cliff line
    ax.axhline(y=HETEROPLASMY_CLIFF, color="white", linestyle="--",
               alpha=0.8, linewidth=1.5)
    ax.text(y_axis[0] + 0.02 * (y_axis[-1] - y_axis[0]),
            HETEROPLASMY_CLIFF + 0.02,
            f"Cliff ({HETEROPLASMY_CLIFF})", color="white", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Intervention comparison overlay ──────────────────────────────────────────

def plot_intervention_comparison(results_dict, output_path="comparison.png",
                                 variable_idx=2, variable_name="ATP"):
    """Overlay trajectories from multiple interventions.

    Args:
        results_dict: Dict mapping intervention name → simulate() result.
        output_path: Path to save.
        variable_idx: State variable index to plot (default 2 = ATP).
        variable_name: Display name for the variable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    # Left panel: chosen state variable
    ax = axes[0]
    for (name, result), color in zip(results_dict.items(), colors):
        time = result["time"]
        values = result["states"][:, variable_idx]
        ax.plot(time, values, color=color, linewidth=1.5, label=name)

    ax.set_xlabel("Years", fontsize=12)
    ax.set_ylabel(variable_name, fontsize=12)
    ax.set_title(f"{variable_name} Trajectories", fontsize=14)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Right panel: heteroplasmy
    ax = axes[1]
    for (name, result), color in zip(results_dict.items(), colors):
        time = result["time"]
        het = result["heteroplasmy"]
        ax.plot(time, het, color=color, linewidth=1.5, label=name)

    ax.axhline(y=HETEROPLASMY_CLIFF, color=COLORS["cliff_line"],
               linestyle="--", alpha=0.7)
    ax.set_xlabel("Years", fontsize=12)
    ax.set_ylabel("Heteroplasmy", fontsize=12)
    ax.set_title("Heteroplasmy Trajectories", fontsize=14)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── README hero: fork-in-the-road phase portrait ───────────────────────────

def plot_readme_fork_in_the_road(output_path="output/readme_fork_in_the_road.png"):
    """Plot one-starting-point, four-futures story figure for README.

    Layout:
      - ATP over time (easy clinical read)
      - Heteroplasmy over time with cliff threshold
      - Simplified ATP-vs-heteroplasmy phase map
    """
    from simulator import simulate

    patient = {
        **DEFAULT_PATIENT,
        "baseline_age": 75.0,
        "baseline_heteroplasmy": 0.62,
        "baseline_nad_level": 0.45,
        "inflammation_level": 0.45,
    }

    protocols = {
        "No treatment": {
            "rapamycin_dose": 0.0, "nad_supplement": 0.0,
            "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.0,
        },
        "Slowing protocol": {
            "rapamycin_dose": 0.35, "nad_supplement": 0.35,
            "senolytic_dose": 0.15, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.35,
        },
        "Reversal (no transplant)": {
            "rapamycin_dose": 0.7, "nad_supplement": 0.8,
            "senolytic_dose": 0.45, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5,
        },
        "Reversal (+ transplant)": {
            "rapamycin_dose": 0.7, "nad_supplement": 0.8,
            "senolytic_dose": 0.45, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.8, "exercise_level": 0.5,
        },
    }
    colors = {
        "No treatment": "#6c757d",
        "Slowing protocol": "#1f77b4",
        "Reversal (no transplant)": "#ff7f0e",
        "Reversal (+ transplant)": "#2ca02c",
    }

    # Precompute all trajectories once so each panel uses identical data.
    traj = {}
    for name, intervention in protocols.items():
        result = simulate(intervention=intervention, patient=patient)
        traj[name] = {
            "time": result["time"],
            "het": result["heteroplasmy"],
            "atp": result["states"][:, 2],
        }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    fig.suptitle("Fork in the Road: Same Starting Point, Four Futures",
                 fontsize=15, fontweight="bold")

    # Panel 1: ATP over time.
    ax = axes[0]
    for name in protocols:
        d = traj[name]
        ax.plot(d["time"], d["atp"], color=colors[name], linewidth=2.4, label=name)
        ax.scatter(float(d["time"][-1]), float(d["atp"][-1]), color=colors[name], s=28, zorder=4)
    ax.set_xlabel("Years")
    ax.set_ylabel("ATP (MU/day)")
    ax.set_title("1) Energy Trajectory")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower left", fontsize=8)

    # Panel 2: Heteroplasmy over time with cliff.
    ax = axes[1]
    ax.axhspan(HETEROPLASMY_CLIFF, 1.0, color="#b22222", alpha=0.08)
    ax.axhline(HETEROPLASMY_CLIFF, color="#8b0000", linestyle="--", linewidth=1.3)
    ax.text(0.02, HETEROPLASMY_CLIFF + 0.015, "Cliff threshold",
            color="#8b0000", fontsize=9, transform=ax.get_yaxis_transform())
    for name in protocols:
        d = traj[name]
        ax.plot(d["time"], d["het"], color=colors[name], linewidth=2.4)
        ax.scatter(float(d["time"][-1]), float(d["het"][-1]), color=colors[name], s=28, zorder=4)
    ax.set_xlabel("Years")
    ax.set_ylabel("Heteroplasmy")
    ax.set_title("2) Damage Trajectory")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.22)

    # Panel 3: Simplified phase map.
    ax = axes[2]
    ax.axvspan(HETEROPLASMY_CLIFF, 1.0, color="#b22222", alpha=0.08)
    ax.axvline(HETEROPLASMY_CLIFF, color="#8b0000", linestyle="--", linewidth=1.3)
    start_point = None
    for name in protocols:
        d = traj[name]
        het = d["het"]
        atp = d["atp"]
        if start_point is None:
            start_point = (float(het[0]), float(atp[0]))
        ax.plot(het, atp, color=colors[name], linewidth=2.4)
        ax.scatter(float(het[-1]), float(atp[-1]), color=colors[name], s=34, zorder=4)
    if start_point is not None:
        ax.scatter(start_point[0], start_point[1], s=72, color="black", zorder=5)
        ax.annotate("Start", xy=start_point, xytext=(8, 6),
                    textcoords="offset points", fontsize=9, color="black")
    ax.set_xlabel("Heteroplasmy")
    ax.set_ylabel("ATP (MU/day)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title("3) State-Space Outcome Map")
    ax.grid(True, alpha=0.22)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── TIQM summary plot ───────────────────────────────────────────────────────

def plot_tiqm_summary(experiments, output_path="tiqm_summary.png"):
    """Plot TIQM experiment resonance scores.

    Args:
        experiments: List of dicts, each with:
            "seed_id": str
            "resonance_behavior": float (0-1)
            "resonance_trajectory": float (0-1)  (or similar)
        output_path: Path to save.
    """
    if not experiments:
        print("  No experiments to plot.")
        return

    seed_ids = [e.get("seed_id", f"exp_{i}") for i, e in enumerate(experiments)]
    n = len(seed_ids)

    # Collect available resonance scores
    behavior_scores = [e.get("resonance_behavior", 0.0) for e in experiments]
    trajectory_scores = [e.get("resonance_trajectory", 0.0) for e in experiments]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n)
    width = 0.35

    bars1 = ax.bar(x - width / 2, behavior_scores, width, label="Behavior Resonance",
                   color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, trajectory_scores, width, label="Trajectory Resonance",
                   color="#2ecc71", alpha=0.8)

    ax.set_xlabel("Clinical Scenario", fontsize=12)
    ax.set_ylabel("Resonance Score (0-1)", fontsize=12)
    ax.set_title("TIQM Experiment: Resonance Scores by Clinical Scenario", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(seed_ids, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Standalone execution ────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from simulator import simulate
    from cliff_mapping import sweep_heteroplasmy, extract_cliff_features
    from constants import DEFAULT_INTERVENTION

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Visualization — Generating All Plots")
    print("=" * 70)

    # 1. Trajectory plot (no intervention)
    print("\n--- Trajectory: No intervention ---")
    result_none = simulate()
    plot_trajectory(result_none,
                    os.path.join(output_dir, "trajectory_no_intervention.png"),
                    title="Natural Aging (70yo, No Intervention)")

    # 2. Trajectory plot (full cocktail)
    print("\n--- Trajectory: Full cocktail ---")
    cocktail = {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    }
    result_cocktail = simulate(intervention=cocktail)
    plot_trajectory(result_cocktail,
                    os.path.join(output_dir, "trajectory_cocktail.png"),
                    title="Full Cocktail (Rapamycin + NAD+ + Senolytics + Exercise)")

    # 3. Cliff curve
    print("\n--- Cliff curve ---")
    sweep = sweep_heteroplasmy(n_points=50, sim_years=10)
    features = extract_cliff_features(sweep)
    plot_cliff_curve(sweep,
                     os.path.join(output_dir, "cliff_curve.png"),
                     cliff_features=features)

    # 4. Intervention comparison
    print("\n--- Intervention comparison ---")
    results = {
        "No treatment": result_none,
        "Rapamycin only": simulate(intervention={**DEFAULT_INTERVENTION,
                                                  "rapamycin_dose": 0.5}),
        "NAD+ only": simulate(intervention={**DEFAULT_INTERVENTION,
                                             "nad_supplement": 0.75}),
        "Full cocktail": result_cocktail,
    }
    plot_intervention_comparison(
        results,
        os.path.join(output_dir, "intervention_comparison.png"),
    )

    # 5. TIQM summary (mock data for standalone test)
    print("\n--- TIQM summary (mock data) ---")
    mock_experiments = [
        {"seed_id": "cognitive_70", "resonance_behavior": 0.72,
         "resonance_trajectory": 0.68},
        {"seed_id": "runner_45", "resonance_behavior": 0.85,
         "resonance_trajectory": 0.79},
        {"seed_id": "near_cliff_80", "resonance_behavior": 0.55,
         "resonance_trajectory": 0.62},
    ]
    plot_tiqm_summary(mock_experiments,
                      os.path.join(output_dir, "tiqm_summary.png"))

    # 6. README hero phase portrait
    print("\n--- README hero: fork in the road ---")
    plot_readme_fork_in_the_road(
        os.path.join(output_dir, "readme_fork_in_the_road.png")
    )

    print("\n" + "=" * 70)
    print(f"All plots saved to {output_dir}/")
