"""Matplotlib visualizations for Zimmerman toolkit analysis results.

Generates publication-quality figures from zimmerman_analysis.py reports.
Uses Agg backend (headless, non-interactive).

Usage:
    python zimmerman_viz.py                        # from saved reports
    python zimmerman_viz.py --reports-dir artifacts/zimmerman

    # Or from Python:
    from zimmerman_viz import generate_all_visualizations
    generate_all_visualizations(reports)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from constants import INTERVENTION_NAMES, PATIENT_NAMES

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output/zimmerman")

# Color scheme: intervention params vs patient params
COLOR_INTERVENTION = "#3498db"
COLOR_PATIENT = "#e74c3c"
COLOR_HIGHLIGHT = "#f39c12"
COLOR_GRID = "#ecf0f1"


def _ensure_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _param_color(name: str) -> str:
    """Color-code by parameter type."""
    if name in INTERVENTION_NAMES:
        return COLOR_INTERVENTION
    elif name in PATIENT_NAMES:
        return COLOR_PATIENT
    return "#95a5a6"


def _short_name(name: str) -> str:
    """Abbreviate parameter names for plot labels."""
    abbrevs = {
        "rapamycin_dose": "rapamycin",
        "nad_supplement": "NAD+",
        "senolytic_dose": "senolytic",
        "yamanaka_intensity": "yamanaka",
        "transplant_rate": "transplant",
        "exercise_level": "exercise",
        "baseline_age": "age",
        "baseline_heteroplasmy": "het₀",
        "baseline_nad_level": "NAD₀",
        "genetic_vulnerability": "gen_vuln",
        "metabolic_demand": "met_demand",
        "inflammation_level": "inflam",
    }
    return abbrevs.get(name, name[:12])


# ── Sobol Bars ────────────────────────────────────────────────────────────────

def plot_sobol_bars(sobol_report: dict, output_key: str = "final_heteroplasmy",
                     save_path: str | None = None) -> None:
    """Horizontal bar chart of S1 and ST Sobol indices.

    Args:
        sobol_report: Output from sobol_sensitivity().
        output_key: Which output metric to visualize.
        save_path: File path to save (default: output/zimmerman/sobol_bars.png).
    """
    _ensure_dir()
    if save_path is None:
        save_path = str(OUTPUT_DIR / "sobol_bars.png")

    if output_key not in sobol_report:
        # Try first available output key
        available = sobol_report.get("output_keys", [])
        if available:
            output_key = available[0]
        else:
            print(f"  No output key '{output_key}' in Sobol report")
            return

    data = sobol_report[output_key]
    s1 = data.get("S1", {})
    st = data.get("ST", {})

    # Sort by total-order index
    params = sorted(st.keys(), key=lambda k: abs(st.get(k, 0)), reverse=True)
    s1_vals = [s1.get(p, 0) for p in params]
    st_vals = [st.get(p, 0) for p in params]
    labels = [_short_name(p) for p in params]
    colors = [_param_color(p) for p in params]

    fig, ax = plt.subplots(1, 1, figsize=(10, max(4, len(params) * 0.4)))
    fig.patch.set_facecolor("white")

    y = np.arange(len(params))
    bar_height = 0.35

    # ST bars (wider, lighter)
    ax.barh(y - bar_height / 2, st_vals, bar_height, color=colors, alpha=0.4,
            label="Total-order (ST)")
    # S1 bars (narrower, solid)
    ax.barh(y + bar_height / 2, s1_vals, bar_height, color=colors, alpha=0.9,
            label="First-order (S1)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Sobol Index", fontsize=11)
    ax.set_title(f"Sobol Sensitivity — {output_key}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    # Legend for param types
    iv_patch = mpatches.Patch(color=COLOR_INTERVENTION, alpha=0.7,
                               label="Intervention")
    pt_patch = mpatches.Patch(color=COLOR_PATIENT, alpha=0.7, label="Patient")
    ax.legend(handles=[iv_patch, pt_patch,
                        mpatches.Patch(facecolor="gray", alpha=0.4, label="ST"),
                        mpatches.Patch(facecolor="gray", alpha=0.9, label="S1")],
              loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Contrastive Sensitivity ──────────────────────────────────────────────────

def plot_contrastive_sensitivity(contrastive_report: dict,
                                  save_path: str | None = None) -> None:
    """Bar chart: per-param flip involvement from contrastive analysis."""
    _ensure_dir()
    if save_path is None:
        save_path = str(OUTPUT_DIR / "contrastive_sensitivity.png")

    sensitivity = contrastive_report.get("sensitivity", {})
    freq = sensitivity.get("param_flip_frequency", {})
    if not freq:
        print("  No contrastive sensitivity data to plot")
        return

    params = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
    vals = [freq[p] for p in params]
    labels = [_short_name(p) for p in params]
    colors = [_param_color(p) for p in params]

    fig, ax = plt.subplots(1, 1, figsize=(10, max(4, len(params) * 0.35)))
    fig.patch.set_facecolor("white")
    ax.barh(range(len(params)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Flip Frequency", fontsize=11)
    ax.set_title("Contrastive Analysis — Parameter Flip Frequency", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Locality Curves ──────────────────────────────────────────────────────────

def plot_locality_curves(locality_report: dict,
                          save_path: str | None = None) -> None:
    """Decay curves per manipulation type from locality profiling."""
    _ensure_dir()
    if save_path is None:
        save_path = str(OUTPUT_DIR / "locality_curves.png")

    curves = locality_report.get("curves", {})
    if not curves:
        print("  No locality curves to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor("white")

    palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12",
               "#1abc9c", "#e67e22"]

    for i, (manip_name, points) in enumerate(curves.items()):
        if not points:
            continue
        xs = [p[0] if isinstance(p, (list, tuple)) else p.get("value", 0) for p in points]
        ys = [p[1] if isinstance(p, (list, tuple)) else p.get("score", 0) for p in points]
        color = palette[i % len(palette)]
        ax.plot(xs, ys, "o-", color=color, label=manip_name, linewidth=1.5,
                markersize=4)

    ax.set_xlabel("Manipulation Intensity", fontsize=11)
    ax.set_ylabel("Normalized Score", fontsize=11)
    ax.set_title("Locality Profiles — Perturbation Decay", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Relation Graph ───────────────────────────────────────────────────────────

def plot_relation_graph(relation_report: dict,
                         save_path: str | None = None) -> None:
    """Force-directed-style causal layout (pure matplotlib, no networkx)."""
    _ensure_dir()
    if save_path is None:
        save_path = str(OUTPUT_DIR / "relation_graph.png")

    edges = relation_report.get("edges", {}).get("causal", [])
    if not edges:
        print("  No causal edges to plot")
        return

    # Collect node positions
    params = set()
    outputs = set()
    for e in edges:
        params.add(e["from"])
        outputs.add(e["to"])

    params = sorted(params)
    outputs = sorted(outputs)[:15]  # limit output nodes

    # Layout: params on left, outputs on right
    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, max(len(params), len(outputs)) * 0.4)))
    fig.patch.set_facecolor("white")

    # Position nodes
    param_y = {p: i for i, p in enumerate(params)}
    output_y = {o: i * (len(params) / max(len(outputs), 1))
                for i, o in enumerate(outputs)}

    x_param = 0.0
    x_output = 1.0

    # Draw edges with width proportional to weight
    max_weight = max((abs(e.get("weight", 0)) for e in edges), default=1.0)
    for e in edges:
        frm, to = e["from"], e["to"]
        if frm not in param_y or to not in output_y:
            continue
        weight = abs(e.get("weight", 0))
        sign = e.get("sign", 1)
        lw = max(0.3, 3.0 * weight / max(max_weight, 1e-6))
        color = "#2ecc71" if sign > 0 else "#e74c3c"
        alpha = min(0.9, 0.2 + 0.7 * weight / max(max_weight, 1e-6))
        ax.plot([x_param, x_output], [param_y[frm], output_y[to]],
                color=color, linewidth=lw, alpha=alpha)

    # Draw nodes
    for p, y in param_y.items():
        ax.plot(x_param, y, "o", color=_param_color(p), markersize=10, zorder=5)
        ax.text(x_param - 0.05, y, _short_name(p), ha="right", va="center",
                fontsize=8)

    for o, y in output_y.items():
        ax.plot(x_output, y, "s", color="#95a5a6", markersize=8, zorder=5)
        label = o.replace("final_", "").replace("energy_", "E:").\
            replace("damage_", "D:").replace("dynamics_", "Dy:").\
            replace("intervention_", "I:")
        ax.text(x_output + 0.05, y, label[:20], ha="left", va="center",
                fontsize=7)

    ax.set_xlim(-0.3, 1.4)
    ax.set_title("Causal Relation Graph — Parameter → Output", fontsize=13)
    ax.axis("off")

    # Legend
    pos_patch = mpatches.Patch(color="#2ecc71", label="Positive influence")
    neg_patch = mpatches.Patch(color="#e74c3c", label="Negative influence")
    ax.legend(handles=[pos_patch, neg_patch], loc="upper center", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── POSIWID Alignment ────────────────────────────────────────────────────────

def plot_posiwid_alignment(posiwid_report: dict,
                            save_path: str | None = None) -> None:
    """Heatmap: intended vs actual outcomes per scenario."""
    _ensure_dir()
    if save_path is None:
        save_path = str(OUTPUT_DIR / "posiwid_alignment.png")

    results = posiwid_report.get("individual_results", [])
    if not results:
        print("  No POSIWID results to plot")
        return

    # Extract scenario labels and per-key alignment scores
    labels = []
    keys_set = set()
    for r in results:
        label = r.get("label", f"Scenario {len(labels)}")
        labels.append(label)
        alignment = r.get("alignment", {}).get("per_key", {})
        keys_set.update(alignment.keys())

    keys = sorted(keys_set)
    if not keys:
        print("  No alignment keys to plot")
        return

    # Build matrix
    matrix = np.zeros((len(labels), len(keys)))
    for i, r in enumerate(results):
        alignment = r.get("alignment", {}).get("per_key", {})
        for j, k in enumerate(keys):
            matrix[i, j] = alignment.get(k, {}).get("combined", 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(keys) * 1.2),
                                            max(4, len(labels) * 0.6)))
    fig.patch.set_facecolor("white")

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("final_", "") for k in keys],
                        rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("POSIWID Alignment — Intended vs Actual", fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Alignment Score", fontsize=10)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(keys)):
            val = matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Dashboard Summary Radar ──────────────────────────────────────────────────

def plot_dashboard_summary(dashboard_report: dict,
                            save_path: str | None = None) -> None:
    """6-axis radar chart summarizing dashboard sections."""
    _ensure_dir()
    if save_path is None:
        save_path = str(OUTPUT_DIR / "dashboard_summary.png")

    sections = dashboard_report.get("sections", {})
    if not sections:
        print("  No dashboard sections to plot")
        return

    # Extract section scores (normalize to 0-1)
    labels = []
    scores = []
    for name, section in sections.items():
        labels.append(name.capitalize())
        # Try to get a representative score
        if isinstance(section, dict):
            # Use the first numeric value as a score proxy
            score = 0.5  # default
            for k, v in section.items():
                if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                    score = min(abs(v), 1.0)
                    break
            scores.append(score)
        else:
            scores.append(0.5)

    if len(labels) < 3:
        print("  Not enough sections for radar chart")
        return

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]  # close the polygon
    angles += [angles[0]]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    ax.fill(angles, scores_plot, alpha=0.25, color=COLOR_INTERVENTION)
    ax.plot(angles, scores_plot, "o-", color=COLOR_INTERVENTION, linewidth=2,
            markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Zimmerman Dashboard — Section Scores", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def generate_all_visualizations(reports: dict | None = None,
                                 reports_dir: str | None = None) -> None:
    """Generate all available visualizations from reports.

    Args:
        reports: Dict of tool_name → report dict (from zimmerman_analysis.py).
        reports_dir: Path to directory with saved report JSON files.
            Used if reports is None.
    """
    _ensure_dir()

    if reports is None:
        reports = _load_reports(reports_dir or "artifacts/zimmerman")

    print("Generating Zimmerman visualizations...")

    if "sobol" in reports:
        # Plot for multiple key outputs
        for key in ["final_heteroplasmy", "final_atp"]:
            safe = key.replace("/", "_")
            plot_sobol_bars(reports["sobol"], output_key=key,
                            save_path=str(OUTPUT_DIR / f"sobol_{safe}.png"))

    if "contrastive" in reports:
        plot_contrastive_sensitivity(reports["contrastive"])

    if "locality" in reports:
        plot_locality_curves(reports["locality"])

    if "relation_graph" in reports:
        plot_relation_graph(reports["relation_graph"])

    if "posiwid" in reports:
        plot_posiwid_alignment(reports["posiwid"])

    if "dashboard" in reports:
        plot_dashboard_summary(reports["dashboard"])

    print(f"All visualizations saved to {OUTPUT_DIR}/")


def _load_reports(directory: str) -> dict:
    """Load saved report JSON files from a directory."""
    reports = {}
    d = Path(directory)
    if not d.exists():
        print(f"  Reports directory not found: {d}")
        return reports

    mapping = {
        "sobol_report.json": "sobol",
        "falsifier_report.json": "falsifier",
        "contrastive_report.json": "contrastive",
        "contrast_sets_report.json": "contrast_sets",
        "pds_report.json": "pds",
        "posiwid_report.json": "posiwid",
        "locality_report.json": "locality",
        "relation_graph_report.json": "relation_graph",
        "diegeticizer_report.json": "diegeticizer",
        "token_extispicy_report.json": "token_extispicy",
        "receptive_field_report.json": "receptive_field",
        "supradiegetic_benchmark_report.json": "supradiegetic_benchmark",
        "dashboard.json": "dashboard",
    }

    for filename, key in mapping.items():
        filepath = d / filename
        if filepath.exists():
            try:
                reports[key] = json.loads(filepath.read_text())
            except json.JSONDecodeError as e:
                print(f"  Warning: could not parse {filepath}: {e}")

    print(f"  Loaded {len(reports)} report(s) from {d}")
    return reports


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Zimmerman analysis visualizations")
    parser.add_argument("--reports-dir", type=str,
                        default="artifacts/zimmerman",
                        help="Directory with saved report JSONs")
    args = parser.parse_args()

    generate_all_visualizations(reports_dir=args.reports_dir)


if __name__ == "__main__":
    main()
