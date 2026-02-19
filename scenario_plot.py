"""Scenario plotting — trajectory panels, milestone bars, summary heatmaps.

All functions use matplotlib Agg backend (headless). Outputs saved to
output/scenarios/ when save=True.

Functions:
    plot_trajectories: Multi-panel line plot (het, ATP, memory) per scenario
    plot_milestone_comparison: Horizontal bar chart of milestone ages
    plot_summary_heatmap: imshow heatmap of metric at multiple ages
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scenario_analysis import extract_milestones, summary_table


# ── Output directory ────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'scenarios')


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Color palette ───────────────────────────────────────────────────────────

SCENARIO_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# ── Trajectory plot ─────────────────────────────────────────────────────────

def plot_trajectories(
    results: list[dict],
    metrics: list[str] | None = None,
    save: bool = False,
) -> plt.Figure:
    """Multi-panel trajectory plot with one line per scenario.

    Args:
        results: List of result dicts from scenario_runner.run_scenarios().
        metrics: Which metrics to plot. Default: ['heteroplasmy', 'atp', 'memory_index'].
        save: If True, save to output/scenarios/trajectories.png.

    Returns:
        matplotlib Figure.
    """
    if metrics is None:
        metrics = ['heteroplasmy', 'atp', 'memory_index']

    n_panels = len(metrics)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for r_idx, r in enumerate(results):
        core = r['core']
        downstream = r['downstream']
        time_arr = core['time']
        base_age = r['scenario'].patient_params.get('baseline_age', 70.0)
        age_arr = base_age + time_arr
        color = SCENARIO_COLORS[r_idx % len(SCENARIO_COLORS)]
        label = r['scenario_name']

        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]
            if metric == 'heteroplasmy':
                ax.plot(age_arr, core['heteroplasmy'], color=color, label=label, linewidth=1.5)
                ax.set_ylabel('Total Heteroplasmy')
                ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
            elif metric == 'atp':
                ax.plot(age_arr, core['states'][:, 2], color=color, label=label, linewidth=1.5)
                ax.set_ylabel('ATP (MU/day)')
            elif metric == 'memory_index':
                memory_vals = [d['memory_index'] for d in downstream]
                ax.plot(age_arr, memory_vals, color=color, label=label, linewidth=1.5)
                ax.set_ylabel('Memory Index')
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
            elif metric == 'deletion_heteroplasmy':
                ax.plot(age_arr, core['deletion_heteroplasmy'], color=color, label=label, linewidth=1.5)
                ax.set_ylabel('Deletion Heteroplasmy')
                ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
            elif metric == 'resilience':
                res_vals = [d['resilience'] for d in downstream]
                ax.plot(age_arr, res_vals, color=color, label=label, linewidth=1.5)
                ax.set_ylabel('Resilience')
            elif metric == 'amyloid_burden':
                amy_vals = [d['amyloid_burden'] for d in downstream]
                ax.plot(age_arr, amy_vals, color=color, label=label, linewidth=1.5)
                ax.set_ylabel('Amyloid Burden')

    for ax_idx, metric in enumerate(metrics):
        axes[ax_idx].set_title(metric.replace('_', ' ').title())
        axes[ax_idx].legend(fontsize=8, loc='best')
        axes[ax_idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Age (years)')
    fig.suptitle('Scenario Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, 'trajectories.png'), dpi=150)

    return fig


# ── Milestone comparison ────────────────────────────────────────────────────

def plot_milestone_comparison(
    milestone_list: list[dict],
    save: bool = False,
) -> plt.Figure:
    """Horizontal bar chart of milestone ages across scenarios.

    Args:
        milestone_list: List of milestone dicts from scenario_analysis.compare_scenarios().
        save: If True, save to output/scenarios/milestones.png.

    Returns:
        matplotlib Figure.
    """
    milestone_keys = [
        ('het_below_50_age', 'Het < 50%'),
        ('atp_above_08_age', 'ATP > 0.8'),
        ('dementia_age', 'Dementia (memory < 0.5)'),
        ('amyloid_pathology_age', 'Amyloid > 1.0'),
    ]

    scenario_names = [m['scenario_name'] for m in milestone_list]
    n_scenarios = len(scenario_names)
    n_milestones = len(milestone_keys)

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * n_scenarios * n_milestones))

    y_positions = []
    y_labels = []
    bar_colors = []
    bar_widths = []

    y = 0
    for s_idx, m in enumerate(milestone_list):
        color = SCENARIO_COLORS[s_idx % len(SCENARIO_COLORS)]
        for mk_key, mk_label in milestone_keys:
            val = m.get(mk_key)
            y_positions.append(y)
            y_labels.append(f"{m['scenario_name'][:20]} / {mk_label}")
            bar_colors.append(color)
            bar_widths.append(val if val is not None else 0)
            y += 1
        y += 0.5  # gap between scenarios

    ax.barh(y_positions, bar_widths, color=bar_colors, alpha=0.8, height=0.7)

    # Mark milestones not reached with 'N/A' text
    for i, w in enumerate(bar_widths):
        if w == 0:
            ax.text(5, y_positions[i], 'N/A', va='center', fontsize=8, color='gray')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel('Age (years)')
    ax.set_title('Milestone Ages by Scenario')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, 'milestones.png'), dpi=150)

    return fig


# ── Summary heatmap ─────────────────────────────────────────────────────────

def plot_summary_heatmap(
    results: list[dict],
    metric: str = 'atp',
    ages: list[float] | None = None,
    save: bool = False,
) -> plt.Figure:
    """Heatmap of a metric at specified ages across scenarios.

    Uses matplotlib imshow (no seaborn dependency).

    Args:
        results: List of result dicts from scenario_runner.run_scenarios().
        metric: Which metric to show ('heteroplasmy', 'atp', 'memory_index').
        ages: Target ages. Default: [70, 80, 90, 95].
        save: If True, save to output/scenarios/heatmap_{metric}.png.

    Returns:
        matplotlib Figure.
    """
    if ages is None:
        ages = [70, 80, 90, 95]

    table = summary_table(results, ages=ages)

    scenario_names = [row['scenario_name'] for row in table]
    n_scenarios = len(scenario_names)
    n_ages = len(ages)

    data = np.full((n_scenarios, n_ages), np.nan)
    for s_idx, row in enumerate(table):
        for a_idx, age in enumerate(ages):
            m = row['metrics'].get(age)
            if m is not None:
                data[s_idx, a_idx] = m[metric]

    fig, ax = plt.subplots(figsize=(2 + n_ages * 1.5, 1 + n_scenarios * 0.8))

    # Choose colormap based on metric
    if metric == 'heteroplasmy':
        cmap = 'RdYlGn_r'  # red = bad (high het)
    elif metric == 'atp':
        cmap = 'RdYlGn'    # green = good (high ATP)
    elif metric == 'memory_index':
        cmap = 'RdYlGn'    # green = good (high memory)
    else:
        cmap = 'viridis'

    im = ax.imshow(data, aspect='auto', cmap=cmap)
    fig.colorbar(im, ax=ax, label=metric.replace('_', ' ').title())

    ax.set_xticks(range(n_ages))
    ax.set_xticklabels([str(int(a)) for a in ages])
    ax.set_xlabel('Age (years)')

    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels([s[:30] for s in scenario_names], fontsize=8)

    # Annotate cells with values
    for s_idx in range(n_scenarios):
        for a_idx in range(n_ages):
            val = data[s_idx, a_idx]
            if not np.isnan(val):
                ax.text(a_idx, s_idx, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if 0.3 < val < 0.7 else 'white')

    ax.set_title(f'{metric.replace("_", " ").title()} by Scenario and Age')
    fig.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'heatmap_{metric}.png'), dpi=150)

    return fig
