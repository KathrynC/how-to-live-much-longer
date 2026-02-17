"""Zimmerman protocol adapter for the combined grief->mito system.

Exposes the combined grief + mitochondrial aging system as a single
Zimmerman-compatible Simulator with a 26-dimensional parameter space
(14 grief + 12 mito). This makes the coupled system compatible with
all 14 zimmerman-toolkit interrogation tools and the cramer-toolkit
resilience analysis framework.

Usage:
    from grief_mito_simulator import GriefMitoSimulator

    sim = GriefMitoSimulator()
    spec = sim.param_spec()    # 26 params with bounds
    result = sim.run({"grief_B": 0.9, "baseline_age": 70.0})

Requires:
    grief-simulator (at ~/grief-simulator)
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

# Project imports (mito project — no name collision issues)
from grief_bridge import (
    GriefDisturbance,
    grief_trajectory,
    GRIEF_INTERVENTION_BOUNDS,
    GRIEF_PATIENT_BOUNDS,
    GRIEF_DEFAULT_INTERVENTION,
    GRIEF_DEFAULT_PATIENT,
    _grief_constants,
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

# Grief analytics — loaded via the same importlib trick in grief_bridge
_grief_analytics = None


def _get_grief_analytics():
    """Lazily load grief analytics module to avoid import collisions."""
    global _grief_analytics
    if _grief_analytics is not None:
        return _grief_analytics

    import importlib.util
    import types

    grief_path = PROJECT.parent / "grief-simulator"

    # Save conflicting modules
    saved = {}
    for name in ("constants", "simulator", "analytics"):
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    sys.path.insert(0, str(grief_path))
    try:
        import constants  # grief constants
        import simulator  # grief simulator (needs grief constants)
        import analytics  # grief analytics (needs grief constants + simulator)
        _grief_analytics = analytics
    finally:
        if str(grief_path) in sys.path:
            sys.path.remove(str(grief_path))
        for name in ("constants", "simulator", "analytics"):
            sys.modules.pop(name, None)
        for name, mod in saved.items():
            sys.modules[name] = mod

    return _grief_analytics


# -- Prefix convention ---------------------------------------------------------
# Grief params are prefixed with "grief_" to avoid name collisions.
# Mito params keep their original names.

GRIEF_PREFIX = "grief_"

INF_CAP = 999.0


class GriefMitoSimulator:
    """Zimmerman Simulator adapter for the combined grief->mito system.

    26D input: 14 grief params (prefixed "grief_") + 12 mito params.
    Output: grief-side metrics (prefixed "grief_") + mito-side metrics.
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return all 26 parameter bounds."""
        spec = {}
        # Grief params (prefixed)
        for name, bounds in GRIEF_INTERVENTION_BOUNDS.items():
            spec[GRIEF_PREFIX + name] = bounds
        for name, bounds in GRIEF_PATIENT_BOUNDS.items():
            spec[GRIEF_PREFIX + name] = bounds
        # Mito params (unprefixed)
        for name, info in MITO_INTERVENTION_PARAMS.items():
            spec[name] = info["range"]
        for name, info in MITO_PATIENT_PARAMS.items():
            spec[name] = info["range"]
        return spec

    def run(self, params: dict[str, float]) -> dict[str, float]:
        """Run combined grief->mito simulation."""
        # Split params into grief and mito groups
        grief_patient = dict(GRIEF_DEFAULT_PATIENT)
        grief_intervention = dict(GRIEF_DEFAULT_INTERVENTION)
        mito_intervention = {}
        mito_patient = {}

        for k, v in params.items():
            if k.startswith(GRIEF_PREFIX):
                grief_key = k[len(GRIEF_PREFIX):]
                if grief_key in GRIEF_INTERVENTION_BOUNDS:
                    grief_intervention[grief_key] = v
                elif grief_key in GRIEF_PATIENT_BOUNDS:
                    grief_patient[grief_key] = v
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

        # Run grief sim -> disturbance -> mito sim
        grief_disturbance = GriefDisturbance(
            grief_patient=grief_patient,
            grief_intervention=grief_intervention,
        )
        mito_result = simulate_with_disturbances(
            intervention=mito_intervention,
            patient=mito_patient,
            disturbances=[grief_disturbance],
        )

        # Compute mito analytics (vs no-treatment baseline)
        mito_baseline = mito_simulate(
            intervention=MITO_DEFAULT_INTERVENTION,
            patient=mito_patient,
        )
        mito_analytics = mito_compute_all(mito_result, mito_baseline)

        # Compute grief analytics
        grief_analytics_mod = _get_grief_analytics()
        grief_sim_fn = grief_disturbance._grief_times  # already computed
        # Re-run grief sim for clean analytics result dict
        from grief_bridge import grief_simulate as _grief_sim
        grief_full = _grief_sim(
            intervention=grief_intervention,
            patient=grief_patient,
        )
        grief_analytics = grief_analytics_mod.compute_all(grief_full)

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

        # Flatten grief metrics (prefixed)
        for pillar_name, pillar_metrics in grief_analytics.items():
            for metric_name, value in pillar_metrics.items():
                flat[f"grief_{metric_name}"] = float(value)

        # Cap infs, replace NaN
        out: dict[str, float] = {}
        for k, v in flat.items():
            if math.isinf(v):
                v = INF_CAP if v > 0 else -INF_CAP
            if math.isnan(v):
                v = 0.0
            out[k] = v

        return out
