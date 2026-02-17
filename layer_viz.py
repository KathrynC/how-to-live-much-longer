"""Population-layer visualizations for core vs agroecology/grief effects.

This module adds cohort-level plots for comparing the baseline model against
additional layers:
    - Agroecology-inspired disturbance stack
    - Grief disturbance bridge
    - Combined agro + grief

Two primary visualizations are provided:
    1. Layer delta forest plot (distributional effect size per metric)
    2. Outcome shift matrix (baseline category -> layered category transitions)

Usage:
    python layer_viz.py --n-patients 1000 --seed 2026
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    ATP_CRISIS_FRACTION,
    DEFAULT_INTERVENTION,
    HETEROPLASMY_CLIFF,
    PATIENT_NAMES,
)
from disturbances import (
    InflammationBurst,
    IonizingRadiation,
    ToxinExposure,
    simulate_with_disturbances,
)
from generate_patients import generate_patients
from simulator import simulate


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "layers")

LAYER_ORDER = ["agro", "grief", "agro_grief"]
METRIC_ORDER = ["final_atp", "final_het", "final_del_het"]
METRIC_LABELS = {
    "final_atp": "Final ATP",
    "final_het": "Final Total Heteroplasmy",
    "final_del_het": "Final Deletion Heteroplasmy",
}
OUTCOME_LABELS = ["collapsed", "declining", "stable", "healthy"]


@dataclass(frozen=True)
class PatientOutcome:
    final_atp: float
    final_het: float
    final_del_het: float
    category: str
    ever_crossed_cliff: bool
    ever_crisis: bool


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _agro_disturbances() -> list:
    """Moderate multi-shock stack used for cohort layer comparisons."""
    return [
        IonizingRadiation(start_year=8.0, duration=1.0, magnitude=0.6),
        ToxinExposure(start_year=12.0, duration=2.0, magnitude=0.5),
        InflammationBurst(start_year=18.0, duration=1.0, magnitude=0.7),
    ]


def _classify_outcome_from_atp(final_atp: float) -> str:
    """Match generate_patients.py category bins for consistency."""
    if final_atp < 0.2:
        return "collapsed"
    if final_atp < 0.5:
        return "declining"
    if final_atp < 0.8:
        return "stable"
    return "healthy"


def _extract_patient_outcome(result: dict) -> PatientOutcome:
    atp = result["states"][:, 2]
    het = result["heteroplasmy"]
    del_het = result.get("deletion_heteroplasmy", het)

    atp0 = float(atp[0])
    crisis_threshold = ATP_CRISIS_FRACTION * atp0

    final_atp = float(atp[-1])
    final_het = float(het[-1])
    final_del_het = float(del_het[-1])

    return PatientOutcome(
        final_atp=final_atp,
        final_het=final_het,
        final_del_het=final_del_het,
        category=_classify_outcome_from_atp(final_atp),
        ever_crossed_cliff=bool(np.any(del_het >= HETEROPLASMY_CLIFF)),
        ever_crisis=bool(np.any(atp < crisis_threshold)),
    )


def simulate_layer_cohort(
    n_patients: int = 1000,
    seed: int = 2026,
) -> dict[str, list[PatientOutcome]]:
    """Simulate a cohort under core/agro/grief/combined layer conditions."""
    from grief_bridge import GriefDisturbance

    grief = GriefDisturbance(
        start_year=5.0,
        duration=10.0,
        magnitude=1.0,
        label="grief_default",
    )

    cohort = {
        "core": [],
        "agro": [],
        "grief": [],
        "agro_grief": [],
    }

    patients = generate_patients(n=n_patients, seed=seed)
    for i, p in enumerate(patients, start=1):
        patient = {k: p[k] for k in PATIENT_NAMES}

        core_r = simulate(intervention=DEFAULT_INTERVENTION, patient=patient)
        agro_r = simulate_with_disturbances(
            intervention=DEFAULT_INTERVENTION,
            patient=patient,
            disturbances=_agro_disturbances(),
        )
        grief_r = simulate_with_disturbances(
            intervention=DEFAULT_INTERVENTION,
            patient=patient,
            disturbances=[grief],
        )
        combo_r = simulate_with_disturbances(
            intervention=DEFAULT_INTERVENTION,
            patient=patient,
            disturbances=_agro_disturbances() + [grief],
        )

        cohort["core"].append(_extract_patient_outcome(core_r))
        cohort["agro"].append(_extract_patient_outcome(agro_r))
        cohort["grief"].append(_extract_patient_outcome(grief_r))
        cohort["agro_grief"].append(_extract_patient_outcome(combo_r))

        if i % 100 == 0:
            print(f"processed {i}/{n_patients}")

    return cohort


def _metric_array(outcomes: list[PatientOutcome], metric: str) -> np.ndarray:
    return np.asarray([getattr(o, metric) for o in outcomes], dtype=float)


def plot_layer_delta_forest(
    cohort: dict[str, list[PatientOutcome]],
    filename: str = "layer_delta_forest.png",
) -> str:
    """Forest-style summary of per-patient deltas vs core for key metrics."""
    outdir = _ensure_output_dir()
    baseline = cohort["core"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Layer Effect Sizes vs Core (Per-Patient Deltas)", fontweight="bold")

    for ax, metric in zip(axes, METRIC_ORDER):
        y_positions = np.arange(len(LAYER_ORDER))
        for i, layer in enumerate(LAYER_ORDER):
            delta = _metric_array(cohort[layer], metric) - _metric_array(baseline, metric)
            p10 = float(np.percentile(delta, 10))
            med = float(np.median(delta))
            p90 = float(np.percentile(delta, 90))

            ax.hlines(y=i, xmin=p10, xmax=p90, color="tab:blue", linewidth=3, alpha=0.7)
            ax.plot(med, i, "o", color="tab:red", markersize=6)

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(LAYER_ORDER)
        ax.set_xlabel(f"Delta {METRIC_LABELS[metric]} (Layer - Core)")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(alpha=0.2, axis="x")
        ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _transition_matrix(
    baseline_outcomes: list[PatientOutcome],
    layered_outcomes: list[PatientOutcome],
) -> np.ndarray:
    idx = {name: i for i, name in enumerate(OUTCOME_LABELS)}
    counts = np.zeros((4, 4), dtype=float)
    for b, l in zip(baseline_outcomes, layered_outcomes):
        counts[idx[b.category], idx[l.category]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(counts, row_sums, where=row_sums > 0.0)
    probs[row_sums[:, 0] == 0.0] = 0.0
    return probs


def plot_outcome_shift_matrix(
    cohort: dict[str, list[PatientOutcome]],
    filename: str = "outcome_shift_matrix.png",
) -> str:
    """Plot baseline->layer transition matrices for ATP outcome categories."""
    outdir = _ensure_output_dir()
    base = cohort["core"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)
    fig.suptitle("Outcome Shift Matrix: Core -> Layered Conditions", fontweight="bold")

    for ax, layer in zip(axes, LAYER_ORDER):
        mat = _transition_matrix(base, cohort[layer])
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlOrRd")
        ax.set_title(layer)
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(OUTCOME_LABELS, rotation=30, ha="right")
        ax.set_yticklabels(OUTCOME_LABELS)
        ax.set_xlabel("Layered Category")
        if layer == LAYER_ORDER[0]:
            ax.set_ylabel("Core Category")

        for r in range(4):
            for c in range(4):
                ax.text(c, r, f"{mat[r, c] * 100:.1f}%", ha="center", va="center",
                        fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Row-Normalized Transition Probability")

    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_layer_effects_dashboard(
    n_patients: int = 1000,
    seed: int = 2026,
) -> list[str]:
    """Run cohort comparison and generate both layer-effect plots."""
    cohort = simulate_layer_cohort(n_patients=n_patients, seed=seed)
    paths = [
        plot_layer_delta_forest(cohort),
        plot_outcome_shift_matrix(cohort),
    ]
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate layer-effect cohort visualizations.")
    parser.add_argument("--n-patients", type=int, default=1000,
                        help="Cohort size (default: 1000)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed for patient generation")
    args = parser.parse_args()

    print("=" * 70)
    print("Layer Visualizations")
    print("=" * 70)
    paths = generate_layer_effects_dashboard(
        n_patients=args.n_patients,
        seed=args.seed,
    )
    for p in paths:
        print(f"Saved: {p}")
    print("=" * 70)
