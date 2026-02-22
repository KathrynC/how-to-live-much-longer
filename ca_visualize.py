"""Visualization suite for the mitochondrial semantic cellular automaton.

Generates headless (Agg backend) Matplotlib plots from CA simulation results:

1. plot_ca_trajectory()   -- 8-variable x N-step heatmap of bin indices
2. plot_rule_timeline()   -- rule firings colored by tier across time
3. plot_ca_fidelity()     -- per-variable CA vs ODE bin agreement bars
4. plot_tissue_grid()     -- 2x2 panel of 4 tissue final states
5. plot_cliff_approach()  -- N_deletion bin trajectory with cliff threshold
6. generate_all_plots()   -- convenience: run simulations, produce all plots

All output is written to PNG files (default: output/ca/).

Mirrors ~/lemurs-simulator/ca_visualize.py architecture.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from ca_schema import BIN_SCHEMA, CA_VAR_ORDER, CA_N_VARS, bin_index
from ca_simulator import run_single_cell, run_tissue_grid, CA_DT, CA_N_STEPS, TISSUE_TYPES
from ca_analytics import compute_ca_analytics, _classify_attractor
from ca_rules import RULE_TABLE


# ── 1. Trajectory heatmap ─────────────────────────────────────────────────


def plot_ca_trajectory(ca_result, title=None, output_path="output/ca/trajectory.png"):
    """8-variable x N-step heatmap. Each row is a state variable, each column
    is a CA step. Color = bin index (0=bad, max=good for positive vars).

    Parameters
    ----------
    ca_result : dict
        Output from ca_simulator.run_single_cell().
    title : str or None
        Plot title (defaults to "CA Trajectory Heatmap").
    output_path : str
        File path for PNG output.
    """
    trajectory = ca_result["trajectory"]
    patient = ca_result.get("patient", {})
    dt = ca_result.get("dt", 0.25)

    # Build heatmap array: (n_vars, n_steps+1)
    data = np.zeros((CA_N_VARS, len(trajectory)))
    for t, state in enumerate(trajectory):
        for i, var_name in enumerate(CA_VAR_ORDER):
            label = state.get(var_name, BIN_SCHEMA[var_name]["labels"][0])
            data[i, t] = bin_index(var_name, label)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", interpolation="nearest")

    # Labels
    ax.set_yticks(range(CA_N_VARS))
    ax.set_yticklabels(CA_VAR_ORDER, fontsize=9)

    # X axis: age
    baseline_age = patient.get("baseline_age", 70.0)
    n_ticks = min(7, len(trajectory))
    tick_positions = np.linspace(0, len(trajectory) - 1, n_ticks, dtype=int)
    tick_labels = [f"{baseline_age + p * dt:.0f}" for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Age (years)")

    plt.colorbar(im, ax=ax, label="Bin index")
    ax.set_title(title or "CA Trajectory Heatmap")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2. Rule firing timeline ──────────────────────────────────────────────


def plot_rule_timeline(ca_result, output_path="output/ca/rule_timeline.png"):
    """Rule firings colored by tier across time.

    Parameters
    ----------
    ca_result : dict
        Output from ca_simulator.run_single_cell().
    output_path : str
        File path for PNG output.
    """
    rule_log = ca_result["rule_log"]
    patient = ca_result.get("patient", {})
    dt = ca_result.get("dt", 0.25)
    baseline_age = patient.get("baseline_age", 70.0)

    # Build tier lookup
    tier_lookup = {r["name"]: r["tier"] for r in RULE_TABLE}
    tier_colors = {0: "C3", 1: "C0", 2: "C1", 3: "C2", 4: "C4", 5: "C5", 6: "C6"}
    tier_labels = {
        0: "Cross-tier", 1: "Energy-Damage", 2: "ROS-Damage",
        3: "Mitophagy", 4: "Senescence", 5: "NAD+", 6: "Interventions",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for step, rules in enumerate(rule_log):
        for rule_name in rules:
            tier = tier_lookup.get(rule_name, 0)
            ax.scatter(step, tier, color=tier_colors.get(tier, "gray"),
                       s=10, alpha=0.6)

    ax.set_yticks(sorted(tier_labels.keys()))
    ax.set_yticklabels([tier_labels[t] for t in sorted(tier_labels.keys())], fontsize=9)
    ax.set_xlabel("CA Step")
    ax.set_ylabel("Rule Tier")
    ax.set_title("Rule Firing Timeline by Tier")
    ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. CA vs ODE fidelity bars ───────────────────────────────────────────


def plot_ca_fidelity(ca_result, ode_result, output_path="output/ca/fidelity.png"):
    """Bar chart of per-variable CA vs ODE agreement.

    Only generates a plot if ode_result is not None. When ode_result is None,
    this function returns without creating a file.

    Parameters
    ----------
    ca_result : dict
        Output from ca_simulator.run_single_cell().
    ode_result : dict or None
        Output from simulator.simulate() with "states" and "time" arrays.
    output_path : str
        File path for PNG output.
    """
    analytics = compute_ca_analytics(ca_result, ode_result)
    fidelity = analytics.get("fidelity_stats")
    if fidelity is None:
        return  # No ODE result to compare against

    per_var = fidelity["per_variable_agreement"]
    vars_list = list(per_var.keys())
    values = [per_var[v] for v in vars_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(vars_list)), values, color="C0", alpha=0.8)
    ax.set_xticks(range(len(vars_list)))
    ax.set_xticklabels(vars_list, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Agreement (fraction)")
    ax.set_title(f"CA vs ODE Bin Agreement (overall: {fidelity['overall_agreement']:.2f})")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=fidelity["overall_agreement"], color="red", linestyle="--",
               alpha=0.5, label="Overall")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 4. 4-tissue final state grid ─────────────────────────────────────────


def plot_tissue_grid(tissue_result, output_path="output/ca/tissue_grid.png"):
    """2x2 panel (brain, muscle, cardiac, skin) showing final state bar charts.

    Each sub-panel shows bin indices for all 8 variables, colored by health
    status (green=best, red=worst, orange=middle).

    Parameters
    ----------
    tissue_result : dict
        Output from ca_simulator.run_tissue_grid().
    output_path : str
        File path for PNG output.
    """
    final_tissues = tissue_result["final_tissues"]
    tissues = list(final_tissues.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("4-Tissue CA Final States", fontsize=14, fontweight="bold")

    for idx, tissue in enumerate(tissues[:4]):
        ax = axes[idx // 2][idx % 2]
        state = final_tissues[tissue]
        attractor = _classify_attractor(state)

        # Bar chart of bin indices
        var_names = CA_VAR_ORDER
        bin_vals = []
        for v in var_names:
            label = state.get(v, BIN_SCHEMA[v]["labels"][0])
            bin_vals.append(bin_index(v, label))

        colors = [
            "green" if bv >= len(BIN_SCHEMA[v]["labels"]) - 1 else
            "red" if bv == 0 else "orange"
            for bv, v in zip(bin_vals, var_names)
        ]

        ax.barh(range(len(var_names)), bin_vals, color=colors, alpha=0.7)
        ax.set_yticks(range(len(var_names)))
        ax.set_yticklabels(var_names, fontsize=8)
        ax.set_title(f"{tissue.capitalize()} ({attractor})", fontsize=10)
        ax.set_xlabel("Bin index")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 5. Cliff approach timeline ───────────────────────────────────────────


def plot_cliff_approach(ca_result, output_path="output/ca/cliff_approach.png"):
    """N_deletion bin trajectory over time with cliff threshold marked.

    Parameters
    ----------
    ca_result : dict
        Output from ca_simulator.run_single_cell().
    output_path : str
        File path for PNG output.
    """
    trajectory = ca_result["trajectory"]
    patient = ca_result.get("patient", {})
    dt = ca_result.get("dt", 0.25)
    baseline_age = patient.get("baseline_age", 70.0)

    del_labels = BIN_SCHEMA["N_deletion"]["labels"]

    steps = range(len(trajectory))
    ages = [baseline_age + s * dt for s in steps]
    del_indices = [
        bin_index("N_deletion", trajectory[s].get("N_deletion", "minimal"))
        for s in steps
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ages, del_indices, color="C3", linewidth=2, label="N_deletion bin")

    # Mark cliff threshold (past_cliff = index 3)
    cliff_idx = del_labels.index("past_cliff")
    ax.axhline(y=cliff_idx - 0.5, color="red", linestyle="--", alpha=0.7,
               label="Cliff threshold")

    ax.set_yticks(range(len(del_labels)))
    ax.set_yticklabels(del_labels, fontsize=10)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("N_deletion bin")
    ax.set_title("Deletion Heteroplasmy: Approach to Cliff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 6. Generate all plots ────────────────────────────────────────────────


def generate_all_plots(output_dir="output/ca"):
    """Generate all CA visualizations.

    Runs single-cell and tissue-grid simulations and produces all available
    plots to the specified output directory.

    Parameters
    ----------
    output_dir : str
        Directory for output PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Single cell
    result = run_single_cell()
    plot_ca_trajectory(result, output_path=os.path.join(output_dir, "trajectory.png"))
    plot_rule_timeline(result, output_path=os.path.join(output_dir, "rule_timeline.png"))
    plot_cliff_approach(result, output_path=os.path.join(output_dir, "cliff_approach.png"))

    # Tissue grid
    tissue_result = run_tissue_grid()
    plot_tissue_grid(tissue_result, output_path=os.path.join(output_dir, "tissue_grid.png"))

    # Young patient (can see transition)
    young_result = run_single_cell(
        patient={"baseline_age": 40.0, "baseline_heteroplasmy": 0.15}
    )
    plot_ca_trajectory(
        young_result, title="Young Patient (Age 40 Start)",
        output_path=os.path.join(output_dir, "trajectory_young.png"),
    )

    print(f"CA plots generated in {output_dir}/")


if __name__ == "__main__":
    print("Generating CA visualizations...")
    generate_all_plots()
    print("Done.")
