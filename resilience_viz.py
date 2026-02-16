"""Matplotlib visualizations for mitochondrial resilience analysis.

Generates plots showing disturbance response, recovery dynamics,
and resilience metric comparisons. All plots use Agg backend
(headless) and save to output/resilience/.

Usage:
    python resilience_viz.py                    # Full visualization suite
    python resilience_viz.py --disturbance radiation --magnitude 0.8

    # Or from Python:
    from resilience_viz import generate_all_plots
    generate_all_plots()
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib
# Agg backend: headless rendering required because this runs in batch mode
# (server, CI, or CLI without display). All output goes to PNG files.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import (
    DEFAULT_INTERVENTION, DEFAULT_PATIENT, STATE_NAMES,
    HETEROPLASMY_CLIFF,
)

# Dedicated subdirectory keeps resilience plots separate from the main
# visualize.py output and the zimmerman_viz.py output, since all three
# visualization modules share the top-level output/ directory.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "resilience")


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def plot_shock_response(
    shocked_result: dict,
    baseline_result: dict,
    title: str = "Shock Response",
    filename: str = "shock_response.png",
) -> str:
    """Plot state trajectories with shock window highlighted.

    Shows ATP, heteroplasmy, ROS, and membrane potential for both
    baseline and shocked trajectories, with the shock window shaded.

    Args:
        shocked_result: Dict from simulate_with_disturbances().
        baseline_result: Dict from simulate().
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    outdir = _ensure_output_dir()
    time = shocked_result["time"]
    shock_times = shocked_result.get("shock_times", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel selection rationale: these four state variables form a minimal
    # diagnostic dashboard for mitochondrial health.
    #   - ATP (idx 2): the primary output variable -- cellular energy supply.
    #     A shock that doesn't move ATP is clinically irrelevant.
    #   - Heteroplasmy (derived, not a raw state index): the key damage
    #     accumulation metric and the variable that determines cliff proximity.
    #     Uses the pre-computed ratio array rather than raw N_h/N_d because
    #     heteroplasmy = N_d/(N_h+N_d) handles edge cases (zero-copy).
    #   - ROS (idx 3): reactive oxygen species are both a cause and a
    #     consequence of damage -- tracking ROS reveals the vicious-cycle
    #     feedback that amplifies small shocks into sustained damage.
    #   - Membrane potential ΔΨ (idx 6): a slave variable that integrates
    #     cliff factor, NAD, and senescence. It serves as a leading indicator
    #     of impending ATP collapse because ΔΨ drops before ATP does.
    # Together these four capture energy, damage, oxidative stress, and
    # membrane integrity -- the four faces of mitochondrial resilience.
    panels = [
        (2, "ATP (MU/day)", "tab:blue"),
        (None, "Heteroplasmy", "tab:red"),  # special: use het array
        (3, "ROS", "tab:orange"),
        (6, "Membrane Potential (ΔΨ)", "tab:green"),
    ]

    for ax, (idx, label, color) in zip(axes.flat, panels):
        # Heteroplasmy uses the pre-computed ratio array (idx=None sentinel)
        # rather than a raw state column because the simulator already
        # handles the N_d/(N_h+N_d) division with zero-denominator guarding.
        if idx is not None:
            baseline_signal = baseline_result["states"][:, idx]
            shocked_signal = shocked_result["states"][:, idx]
        else:
            baseline_signal = baseline_result["heteroplasmy"]
            shocked_signal = shocked_result["heteroplasmy"]

        # Baseline drawn thin and transparent so the shocked trajectory
        # dominates visually -- the reader's eye should track the deviation,
        # not the reference.  Shocked trajectory uses full opacity and
        # heavier lineweight to convey "this is the story being told."
        ax.plot(time, baseline_signal, color=color, alpha=0.4,
                linewidth=1, label="Baseline")
        ax.plot(time, shocked_signal, color=color, linewidth=2,
                label="Shocked")

        # Shock window shading: light red fill marks the active disturbance
        # period, dashed onset line marks the exact start.  The fill uses
        # very low alpha (0.1) so it tints the background without obscuring
        # the trajectory data underneath.
        for s_start, s_end in shock_times:
            ax.axvspan(s_start, s_end, color="red", alpha=0.1)
            ax.axvline(s_start, color="red", linestyle="--",
                       alpha=0.5, linewidth=0.8)

        # The heteroplasmy cliff (0.70) is the most important threshold in
        # the entire model: above it, ATP production collapses nonlinearly
        # (sigmoid steepness=15).  Drawing it only on the het panel avoids
        # visual clutter on panels where it has no direct meaning.
        if idx is None:
            ax.axhline(HETEROPLASMY_CLIFF, color="darkred", linestyle=":",
                       alpha=0.5, label=f"Cliff ({HETEROPLASMY_CLIFF})")

        ax.set_xlabel("Time (years)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_resilience_comparison(
    results: list[dict],
    baseline_result: dict,
    labels: list[str],
    state_idx: int = 2,
    ylabel: str = "ATP (MU/day)",
    title: str = "Resilience Comparison",
    filename: str = "resilience_comparison.png",
) -> str:
    """Compare multiple shocked trajectories against baseline.

    Args:
        results: List of dicts from simulate_with_disturbances().
        baseline_result: Dict from simulate().
        labels: Label for each result.
        state_idx: State variable to plot.
        ylabel: Y-axis label.
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    outdir = _ensure_output_dir()
    time = baseline_result["time"]

    fig, ax = plt.subplots(figsize=(12, 6))
    # Baseline in black to serve as a neutral reference that doesn't compete
    # with the color-coded disturbance trajectories for visual attention.
    ax.plot(time, baseline_result["states"][:, state_idx],
            color="black", linewidth=2, label="Baseline (no shock)")

    # Set1 colormap: qualitative palette chosen so each disturbance type
    # is perceptually distinct.  Avoids sequential colormaps (viridis etc.)
    # which would falsely imply an ordering among disturbance categories.
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for result, label, color in zip(results, labels, colors):
        ax.plot(time, result["states"][:, state_idx],
                color=color, linewidth=1.5, alpha=0.8, label=label)

        # Shock windows tinted per-disturbance color (very low alpha=0.05)
        # so overlapping windows from different disturbances remain readable.
        for s_start, s_end in result.get("shock_times", []):
            ax.axvspan(s_start, s_end, color=color, alpha=0.05)

    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_resilience_summary(
    sweep_results: list[dict],
    title: str = "Resilience vs Shock Magnitude",
    filename: str = "resilience_summary.png",
) -> str:
    """Bar/line chart of resilience metrics across magnitudes.

    Args:
        sweep_results: List of dicts from compute_resilience_sweep().
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    outdir = _ensure_output_dir()

    magnitudes = [r["magnitude"] for r in sweep_results]
    scores = [r["summary_score"] for r in sweep_results]
    resistances = [r["resistance"]["relative_peak_deviation"] for r in sweep_results]
    recoveries = [r["recovery_time_years"] for r in sweep_results]
    regimes = [1.0 if r["regime"]["regime_retained"] else 0.0 for r in sweep_results]

    # Four-panel layout mirrors the four components of the composite
    # resilience score (resistance, recovery, regime, elasticity) from
    # resilience_metrics.py.  Each panel isolates one aspect so the reader
    # can see which component drives the composite score collapse.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Top-left: composite score (0-1).  This is the weighted combination
    # of resistance(25%), recovery(30%), regime(25%), elasticity(20%).
    # Fixed y-range [0,1] makes scores comparable across different plots.
    ax = axes[0, 0]
    ax.bar(range(len(magnitudes)), scores, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(magnitudes)))
    ax.set_xticklabels([f"{m:.1f}" for m in magnitudes])
    ax.set_xlabel("Shock Magnitude")
    ax.set_ylabel("Resilience Score")
    ax.set_title("Composite Resilience Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")

    # Top-right: resistance = how much the state deviates at peak shock.
    # Coral color chosen as a warm hue signaling "damage magnitude."
    # No fixed y-limit because peak deviation can exceed 1.0 for severe shocks.
    ax = axes[0, 1]
    ax.bar(range(len(magnitudes)), resistances, color="coral", alpha=0.8)
    ax.set_xticks(range(len(magnitudes)))
    ax.set_xticklabels([f"{m:.1f}" for m in magnitudes])
    ax.set_xlabel("Shock Magnitude")
    ax.set_ylabel("Relative Peak Deviation")
    ax.set_title("Resistance (lower = better)")
    ax.grid(True, alpha=0.2, axis="y")

    # Bottom-left: recovery time with traffic-light color coding.
    # The 30-year cap prevents infinite/very-large recovery times (which
    # occur when the system never recovers, e.g. post-cliff collapse) from
    # blowing out the y-axis and making recoverable cases unreadable.
    # 30 years is the simulation horizon, so "30" effectively means "never."
    ax = axes[1, 0]
    rec_display = [min(r, 30.0) for r in recoveries]
    # Traffic-light thresholds for clinical interpretability:
    #   green  (< 5 yr):  rapid recovery -- transient perturbation
    #   orange (5-15 yr): slow recovery -- sustained but reversible damage
    #   red    (>= 15 yr): functionally non-recoverable within a human
    #                       treatment horizon (or truly irreversible post-cliff)
    # These thresholds were chosen to match clinical timescales: 5 years is
    # a typical follow-up window, 15 years approaches half the remaining
    # lifespan for a 70-year-old patient.
    colors = ["green" if r < 5 else "orange" if r < 15 else "red"
              for r in recoveries]
    ax.bar(range(len(magnitudes)), rec_display, color=colors, alpha=0.8)
    ax.set_xticks(range(len(magnitudes)))
    ax.set_xticklabels([f"{m:.1f}" for m in magnitudes])
    ax.set_xlabel("Shock Magnitude")
    ax.set_ylabel("Recovery Time (years)")
    ax.set_title("Recovery Time (lower = better)")
    ax.grid(True, alpha=0.2, axis="y")

    # Bottom-right: regime retention -- binary outcome.
    # "Regime" = which side of the heteroplasmy cliff the system ends on.
    # This is the single most important clinical question: did the shock
    # push the patient past the point of no return?  Green/red binary
    # coloring reinforces the all-or-nothing nature of cliff crossing.
    # y-range [-0.1, 1.1] adds visual breathing room around the 0/1 bars.
    ax = axes[1, 1]
    regime_colors = ["green" if r > 0.5 else "red" for r in regimes]
    ax.bar(range(len(magnitudes)), regimes, color=regime_colors, alpha=0.8)
    ax.set_xticks(range(len(magnitudes)))
    ax.set_xticklabels([f"{m:.1f}" for m in magnitudes])
    ax.set_xlabel("Shock Magnitude")
    ax.set_ylabel("Regime Retained")
    ax.set_title("Regime Retention (1 = retained, 0 = shifted)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recovery_landscape(
    disturbance_class,
    magnitudes: list[float] | None = None,
    start_years: list[float] | None = None,
    intervention: dict[str, float] | None = None,
    patient: dict[str, float] | None = None,
    filename: str = "recovery_landscape.png",
) -> str:
    """Heatmap of recovery time vs shock magnitude and timing.

    Args:
        disturbance_class: Disturbance subclass to test.
        magnitudes: Magnitude values (default: 0.1 to 1.0).
        start_years: Shock start times (default: 2 to 24 years).
        intervention: Intervention dict.
        patient: Patient dict.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    # Deferred imports: these modules pull in the full ODE integrator, so
    # importing at function scope keeps the module importable without
    # triggering heavy initialization when only other functions are needed.
    from disturbances import simulate_with_disturbances
    from simulator import simulate
    from resilience_metrics import compute_resilience

    outdir = _ensure_output_dir()

    # Default magnitude grid: 10 uniform steps spanning the full [0,1]
    # disturbance range.  Default timing grid: 8 points covering the 30-year
    # simulation window, denser at early years where the system is still
    # in a transient from initial conditions.
    if magnitudes is None:
        magnitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if start_years is None:
        start_years = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 24.0]

    baseline = simulate(intervention=intervention, patient=patient)

    # Build the 2D recovery time matrix: rows = magnitude, cols = timing.
    # This is the core computation -- each cell requires a full 30-year ODE
    # integration plus resilience metric computation, so total cost is
    # len(magnitudes) * len(start_years) simulations.
    recovery_matrix = np.zeros((len(magnitudes), len(start_years)))

    for i, mag in enumerate(magnitudes):
        for j, sy in enumerate(start_years):
            shock = disturbance_class(start_year=sy, magnitude=mag)
            result = simulate_with_disturbances(
                intervention=intervention, patient=patient,
                disturbances=[shock])
            metrics = compute_resilience(result, baseline)
            rt = metrics["recovery_time_years"]
            # Cap at 30 years (the simulation horizon). Infinite or very
            # large recovery times arise when the system crosses the
            # heteroplasmy cliff and never returns -- these represent
            # permanent regime shifts, not just slow recovery.  Capping
            # keeps the colorbar range useful for distinguishing among
            # recoverable cases while still showing non-recovery as "max."
            recovery_matrix[i, j] = min(rt, 30.0)

    fig, ax = plt.subplots(figsize=(10, 7))

    # imshow renders the matrix as a continuous heatmap.
    #   origin="lower": puts low magnitudes at the bottom (natural for a
    #       y-axis where "more severe" = higher), matching the convention
    #       that magnitude increases upward.
    #   extent: maps pixel coordinates to physical units (years on x-axis,
    #       magnitude on y-axis) so tick labels reflect actual parameter
    #       values rather than array indices.
    #   aspect="auto": allows the image to stretch to fill the figure
    #       rather than enforcing square pixels, since the two axes have
    #       very different physical scales (years vs dimensionless magnitude).
    #   cmap="RdYlGn_r": reversed Red-Yellow-Green so that GREEN = fast
    #       recovery (good) and RED = slow/no recovery (bad).  The _r suffix
    #       reverses the default RdYlGn which maps red→low, green→high;
    #       here low recovery time is desirable, so we want low→green.
    im = ax.imshow(recovery_matrix, aspect="auto", origin="lower",
                   cmap="RdYlGn_r",
                   extent=[start_years[0], start_years[-1],
                           magnitudes[0], magnitudes[-1]])

    ax.set_xlabel("Shock Start Time (years)", fontsize=11)
    ax.set_ylabel("Shock Magnitude", fontsize=11)
    ax.set_title(f"Recovery Landscape: {disturbance_class.__name__}",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, label="Recovery Time (years)")

    plt.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(
    intervention: dict[str, float] | None = None,
    patient: dict[str, float] | None = None,
) -> list[str]:
    """Generate the complete resilience visualization suite.

    Args:
        intervention: Intervention dict (default: no treatment).
        patient: Patient dict (default: typical 70yo).

    Returns:
        List of paths to generated plots.
    """
    from disturbances import (
        simulate_with_disturbances,
        IonizingRadiation, ToxinExposure, ChemotherapyBurst,
        InflammationBurst,
    )
    from simulator import simulate
    from resilience_metrics import compute_resilience, compute_resilience_sweep

    paths = []
    # Single baseline computation shared across all downstream plots.
    # This is the no-shock trajectory against which all resilience metrics
    # are measured -- computing it once avoids redundant 30-year ODE solves.
    baseline = simulate(intervention=intervention, patient=patient)

    # 1. Individual shock response plots.
    # Each disturbance class models a distinct biological insult mechanism.
    # Magnitudes are chosen to produce clearly visible but non-trivial
    # responses: high enough to see deviations, not so high that every
    # case collapses (which would make comparison uninformative).
    # Toxin uses 0.6 (lower) because its effect profile is more sustained;
    # inflammation uses 0.7 as a middle ground.
    disturbance_configs = [
        (IonizingRadiation, "radiation", 0.8),
        (ToxinExposure, "toxin", 0.6),
        (ChemotherapyBurst, "chemo", 0.8),
        (InflammationBurst, "inflammation", 0.7),
    ]

    shocked_results = []
    shock_labels = []

    for DistClass, name, mag in disturbance_configs:
        # All shocks applied at year 10: early enough to observe both the
        # acute response and the 20-year recovery trajectory, but late
        # enough that the system has settled from initial transients.
        shock = DistClass(start_year=10.0, magnitude=mag)
        result = simulate_with_disturbances(
            intervention=intervention, patient=patient,
            disturbances=[shock])
        shocked_results.append(result)
        shock_labels.append(f"{shock.name} (mag={mag})")

        p = plot_shock_response(
            result, baseline,
            title=f"Shock Response: {shock.name} (magnitude={mag})",
            filename=f"shock_{name}.png")
        paths.append(p)
        print(f"  Saved {p}")

    # 2. Comparison plot: all disturbances
    p = plot_resilience_comparison(
        shocked_results, baseline, shock_labels,
        title="Disturbance Comparison: ATP Trajectories",
        filename="disturbance_comparison_atp.png")
    paths.append(p)
    print(f"  Saved {p}")

    # 3. Heteroplasmy comparison -- manual implementation.
    # plot_resilience_comparison() indexes into the states array by column,
    # but heteroplasmy is a derived ratio (N_d/(N_h+N_d)) stored in its
    # own key, not as a raw state column.  Rather than adding special-case
    # logic to the generic comparison function, we build this panel by hand.
    # The dead call below is kept for documentation of the API limitation.
    p = plot_resilience_comparison(
        shocked_results, baseline, shock_labels,
        state_idx=None,  # heteroplasmy handled specially below
        ylabel="Heteroplasmy",
        title="Disturbance Comparison: Heteroplasmy",
        filename="disturbance_comparison_het.png")

    # Manual heteroplasmy comparison with cliff reference line
    outdir = _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(12, 6))
    time = baseline["time"]
    ax.plot(time, baseline["heteroplasmy"],
            color="black", linewidth=2, label="Baseline")
    # The cliff line is the central visual anchor: it divides the plot
    # into "safe" (below) and "collapse" (above) regions.  Dark red
    # and dotted style differentiate it from the trajectory lines.
    ax.axhline(HETEROPLASMY_CLIFF, color="darkred", linestyle=":",
               alpha=0.5, label=f"Cliff ({HETEROPLASMY_CLIFF})")
    colors = plt.cm.Set1(np.linspace(0, 1, len(shocked_results)))
    for result, label, color in zip(shocked_results, shock_labels, colors):
        ax.plot(time, result["heteroplasmy"],
                color=color, linewidth=1.5, alpha=0.8, label=label)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Heteroplasmy")
    ax.set_title("Disturbance Comparison: Heteroplasmy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    het_path = os.path.join(outdir, "disturbance_comparison_het.png")
    fig.savefig(het_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(het_path)
    print(f"  Saved {het_path}")

    # 4. Magnitude sweep for radiation.
    # Non-uniform magnitude grid (denser at low end) captures the expected
    # nonlinear transition where resilience degrades rapidly past a
    # critical shock magnitude.
    sweep = compute_resilience_sweep(
        IonizingRadiation,
        [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        intervention=intervention, patient=patient)
    p = plot_resilience_summary(
        sweep,
        title="Resilience vs Radiation Magnitude",
        filename="resilience_sweep_radiation.png")
    paths.append(p)
    print(f"  Saved {p}")

    # 5. Magnitude sweep for chemo -- same grid as radiation for direct
    # comparison.  Chemotherapy and radiation are the two iatrogenic
    # disturbances (treatment-caused), so comparing their resilience
    # profiles is clinically relevant.
    sweep_chemo = compute_resilience_sweep(
        ChemotherapyBurst,
        [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        intervention=intervention, patient=patient)
    p = plot_resilience_summary(
        sweep_chemo,
        title="Resilience vs Chemotherapy Magnitude",
        filename="resilience_sweep_chemo.png")
    paths.append(p)
    print(f"  Saved {p}")

    # 6-7. Recovery landscapes: 2D heatmaps (magnitude x timing).
    # These reveal timing-dependent vulnerability: a shock at year 5 may
    # be recoverable while the same shock at year 20 (higher accumulated
    # heteroplasmy) may cross the cliff.  Coarser 6x6 grid (36 sims each)
    # balances resolution against compute cost (~36 ODE solves per landscape).
    p = plot_recovery_landscape(
        IonizingRadiation,
        magnitudes=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        start_years=[2.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        intervention=intervention, patient=patient,
        filename="recovery_landscape_radiation.png")
    paths.append(p)
    print(f"  Saved {p}")

    p = plot_recovery_landscape(
        ChemotherapyBurst,
        magnitudes=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        start_years=[2.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        intervention=intervention, patient=patient,
        filename="recovery_landscape_chemo.png")
    paths.append(p)
    print(f"  Saved {p}")

    return paths


# ── CLI ──────────────────────────────────────────────────────────────────────
# Two modes: single-disturbance (fast, for iteration) or full suite.
# Single-disturbance mode produces one 4-panel shock response plot;
# full suite generates all 9+ plots (individual shocks, comparisons,
# magnitude sweeps, recovery landscapes).

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate resilience visualizations")
    parser.add_argument("--disturbance", type=str, default=None,
                        choices=["radiation", "toxin", "chemo", "inflammation"],
                        help="Generate single disturbance plot")
    parser.add_argument("--magnitude", type=float, default=0.8,
                        help="Disturbance magnitude (default: 0.8)")
    args = parser.parse_args()

    print("=" * 70)
    print("Resilience Visualizations")
    print("=" * 70)

    if args.disturbance:
        from disturbances import (
            simulate_with_disturbances,
            IonizingRadiation, ToxinExposure,
            ChemotherapyBurst, InflammationBurst,
        )
        from simulator import simulate

        dist_map = {
            "radiation": IonizingRadiation,
            "toxin": ToxinExposure,
            "chemo": ChemotherapyBurst,
            "inflammation": InflammationBurst,
        }
        DistClass = dist_map[args.disturbance]
        # Default patient, shock at year 10 -- same timing as the full
        # suite so single-disturbance plots are visually comparable.
        shock = DistClass(start_year=10.0, magnitude=args.magnitude)
        baseline = simulate()
        result = simulate_with_disturbances(disturbances=[shock])
        p = plot_shock_response(
            result, baseline,
            title=f"{shock.name} (mag={args.magnitude})",
            filename=f"shock_{args.disturbance}.png")
        print(f"Saved: {p}")
    else:
        paths = generate_all_plots()
        print(f"\nGenerated {len(paths)} plots in {OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print("Done.")
