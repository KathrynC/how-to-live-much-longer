"""Zimmerman protocol adapter for the combined LEMURS->mito system.

Exposes the combined LEMURS college student + mitochondrial aging system as
a single Zimmerman-compatible Simulator with a 24-dimensional parameter space
(12 LEMURS + 12 mito). This makes the coupled system compatible with
all 14 zimmerman-toolkit interrogation tools and the kcramer
resilience analysis framework.

The scientific question: "Does chronic college stress leave a lasting
mitochondrial signature, and can well-being interventions protect
against it?"

Usage:
    from lemurs_mito_simulator import LEMURSMitoSimulator

    sim = LEMURSMitoSimulator()
    spec = sim.param_spec()    # 24 params with bounds
    result = sim.run({"lemurs_nature_rx": 0.8, "baseline_age": 18.0})

Requires:
    lemurs-simulator (at ~/lemurs-simulator)
    zimmerman-toolkit (at ~/zimmerman-toolkit)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# -- Path setup ---------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"
if str(ZIMMERMAN_PATH) not in sys.path:
    sys.path.insert(0, str(ZIMMERMAN_PATH))

# Project imports (mito project -- no name collision issues)
from lemurs_bridge import (
    LEMURSDisturbance,
    lemurs_trajectory,
    LEMURS_INTERVENTION_BOUNDS,
    LEMURS_PATIENT_BOUNDS,
    LEMURS_DEFAULT_INTERVENTION,
    LEMURS_DEFAULT_PATIENT,
    lemurs_simulate as _lemurs_sim,
)
from disturbances import simulate_with_disturbances
from constants import (
    INTERVENTION_PARAMS as MITO_INTERVENTION_PARAMS,
    PATIENT_PARAMS as MITO_PATIENT_PARAMS,
    INTERVENTION_NAMES as MITO_INTERVENTION_NAMES,
    PATIENT_NAMES as MITO_PATIENT_NAMES,
    DEFAULT_INTERVENTION as MITO_DEFAULT_INTERVENTION,
    DEFAULT_PATIENT as MITO_DEFAULT_PATIENT,
)
from simulator import simulate as mito_simulate
from analytics import compute_all as mito_compute_all

# LEMURS analytics -- loaded via importlib trick to avoid module collisions
_lemurs_analytics = None


def _get_lemurs_analytics():
    """Lazily load LEMURS analytics module to avoid import collisions."""
    global _lemurs_analytics
    if _lemurs_analytics is not None:
        return _lemurs_analytics

    import importlib.util
    import types

    lemurs_path = PROJECT.parent / "lemurs-simulator"

    # Save conflicting modules
    saved = {}
    for name in ("constants", "simulator", "analytics"):
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    sys.path.insert(0, str(lemurs_path))
    try:
        import constants  # lemurs constants
        import simulator  # lemurs simulator (needs lemurs constants)
        import analytics  # lemurs analytics (needs lemurs constants + simulator)
        _lemurs_analytics = analytics
    finally:
        if str(lemurs_path) in sys.path:
            sys.path.remove(str(lemurs_path))
        for name in ("constants", "simulator", "analytics"):
            sys.modules.pop(name, None)
        for name, mod in saved.items():
            sys.modules[name] = mod

    return _lemurs_analytics


# -- Prefix convention ---------------------------------------------------------
# LEMURS params are prefixed with "lemurs_" to avoid name collisions.
# Mito params keep their original names.

LEMURS_PREFIX = "lemurs_"

INF_CAP = 999.0


class LEMURSMitoSimulator:
    """Zimmerman Simulator adapter for the combined LEMURS->mito system.

    24D input: 12 LEMURS params (prefixed "lemurs_") + 12 mito params.
    Output: LEMURS-side metrics (prefixed "lemurs_") + mito-side metrics.
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return all 24 parameter bounds."""
        spec = {}
        # LEMURS params (prefixed)
        for name, bounds in LEMURS_INTERVENTION_BOUNDS.items():
            spec[LEMURS_PREFIX + name] = bounds
        for name, bounds in LEMURS_PATIENT_BOUNDS.items():
            spec[LEMURS_PREFIX + name] = bounds
        # Mito params (unprefixed)
        for name, info in MITO_INTERVENTION_PARAMS.items():
            spec[name] = info["range"]
        for name, info in MITO_PATIENT_PARAMS.items():
            spec[name] = info["range"]
        return spec

    def run(self, params: dict[str, float]) -> dict[str, float]:
        """Run combined LEMURS->mito simulation."""
        # Split params into LEMURS and mito groups
        lemurs_patient = dict(LEMURS_DEFAULT_PATIENT)
        lemurs_intervention = dict(LEMURS_DEFAULT_INTERVENTION)
        mito_intervention = {}
        mito_patient = {}

        # Track whether baseline_age was explicitly provided
        baseline_age_provided = "baseline_age" in params

        for k, v in params.items():
            if k.startswith(LEMURS_PREFIX):
                lemurs_key = k[len(LEMURS_PREFIX):]
                if lemurs_key in LEMURS_INTERVENTION_BOUNDS:
                    lemurs_intervention[lemurs_key] = v
                elif lemurs_key in LEMURS_PATIENT_BOUNDS:
                    lemurs_patient[lemurs_key] = v
            else:
                if k in MITO_INTERVENTION_NAMES:
                    mito_intervention[k] = v
                elif k in MITO_PATIENT_NAMES:
                    mito_patient[k] = v

        # Fill mito defaults
        for name in MITO_INTERVENTION_NAMES:
            mito_intervention.setdefault(name, MITO_DEFAULT_INTERVENTION[name])
        for name in MITO_PATIENT_NAMES:
            mito_patient.setdefault(name, MITO_DEFAULT_PATIENT[name])

        # Override baseline_age to 18.0 for college student if not explicitly provided
        if not baseline_age_provided:
            mito_patient["baseline_age"] = 18.0

        # Run LEMURS sim -> disturbance -> mito sim
        lemurs_disturbance = LEMURSDisturbance(
            lemurs_patient=lemurs_patient,
            lemurs_intervention=lemurs_intervention,
            start_year=0.0,
            duration=30.0,
        )
        mito_result = simulate_with_disturbances(
            intervention=mito_intervention,
            patient=mito_patient,
            disturbances=[lemurs_disturbance],
        )

        # Compute mito analytics (vs no-treatment baseline)
        mito_baseline = mito_simulate(
            intervention=MITO_DEFAULT_INTERVENTION,
            patient=mito_patient,
        )
        mito_analytics = mito_compute_all(mito_result, mito_baseline)

        # Compute LEMURS analytics
        lemurs_analytics_mod = _get_lemurs_analytics()
        # Re-run LEMURS sim for clean analytics result dict
        lemurs_full = _lemurs_sim(
            intervention=lemurs_intervention,
            patient=lemurs_patient,
        )
        # LEMURS analytics needs both treated and baseline results
        lemurs_baseline = _lemurs_sim(
            intervention=dict(LEMURS_DEFAULT_INTERVENTION),
            patient=lemurs_patient,
        )
        lemurs_analytics = lemurs_analytics_mod.compute_all(
            lemurs_full, baseline=lemurs_baseline
        )

        # Flatten mito metrics
        flat: dict[str, float] = {}
        for pillar_name, pillar_metrics in mito_analytics.items():
            for metric_name, value in pillar_metrics.items():
                flat[f"{pillar_name}_{metric_name}"] = float(value)

        # Add mito trajectory endpoints
        flat["final_heteroplasmy"] = float(mito_result["heteroplasmy"][-1])
        flat["final_atp"] = float(mito_result["states"][-1, 2])
        flat["final_ros"] = float(mito_result["states"][-1, 3])
        flat["final_nad"] = float(mito_result["states"][-1, 4])
        flat["final_senescent"] = float(mito_result["states"][-1, 5])
        flat["final_membrane_potential"] = float(mito_result["states"][-1, 6])

        # Flatten LEMURS metrics (prefixed)
        for pillar_name, pillar_metrics in lemurs_analytics.items():
            for metric_name, value in pillar_metrics.items():
                flat[f"lemurs_{metric_name}"] = float(value)

        # Cap infs, replace NaN
        out: dict[str, float] = {}
        for k, v in flat.items():
            if math.isinf(v):
                v = INF_CAP if v > 0 else -INF_CAP
            if math.isnan(v):
                v = 0.0
            out[k] = v

        return out
