"""Zimmerman Simulator protocol adapter for the mitochondrial aging model.

Wraps the JGC mitochondrial aging simulator (simulator.py + analytics.py)
as a zimmerman-toolkit Simulator, enabling all 14 interrogation tools:
Sobol, Falsifier, ContrastiveGenerator, ContrastSetGenerator, PDSMapper,
POSIWIDAuditor, LocalityProfiler, RelationGraphExtractor, Diegeticizer,
TokenExtispicyWorkbench, PromptReceptiveField, SuperdiegeticBenchmark,
MeaningConstructionDashboard, PromptBuilder.

The Simulator protocol requires:
    run(params: dict) -> dict   — flat param dict in, flat metric dict out
    param_spec() -> dict[str, tuple[float, float]]   — parameter bounds

Usage:
    from zimmerman_bridge import MitoSimulator

    sim = MitoSimulator()                      # full 12D mode
    sim_iv = MitoSimulator(intervention_only=True)  # 6D intervention-only

    result = sim.run({"rapamycin_dose": 0.5, "baseline_age": 70, ...})
    spec = sim.param_spec()

Requires:
    zimmerman-toolkit (at ~/zimmerman-toolkit)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"
if str(ZIMMERMAN_PATH) not in sys.path:
    sys.path.insert(0, str(ZIMMERMAN_PATH))

# Project imports
from constants import (
    INTERVENTION_PARAMS, PATIENT_PARAMS,
    INTERVENTION_NAMES, PATIENT_NAMES,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    STATE_NAMES,
)
from simulator import simulate
from analytics import compute_all

# Zimmerman protocol (for isinstance checks)
from zimmerman.base import Simulator


# ── Infinity cap ──────────────────────────────────────────────────────────────

INF_CAP = 999.0


def _cap_infs(d: dict) -> dict:
    """Replace inf/-inf values with ±INF_CAP, drop NaN."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            v = float(v)
            if np.isnan(v):
                continue
            if np.isinf(v):
                v = INF_CAP if v > 0 else -INF_CAP
            out[k] = v
    return out


# ── MitoSimulator ─────────────────────────────────────────────────────────────


class MitoSimulator:
    """Zimmerman Simulator adapter for the mitochondrial aging ODE model.

    Accepts a flat parameter dict, splits into intervention + patient dicts,
    runs a 30-year simulation, computes 4-pillar analytics, and returns a
    flat dict of scalar metrics.

    Args:
        intervention_only: If True, expose only the 6 intervention params.
            Patient params are fixed to ``patient_override`` (or defaults).
        patient_override: Fixed patient dict for intervention-only mode.
            Ignored if ``intervention_only`` is False.
    """

    def __init__(
        self,
        intervention_only: bool = False,
        patient_override: dict[str, float] | None = None,
    ) -> None:
        self.intervention_only = intervention_only
        self.patient_override = patient_override or dict(DEFAULT_PATIENT)

        # Cache baseline for the intervention pillar
        self._baseline_cache: dict | None = None
        self._baseline_patient_key: tuple | None = None

    def _get_baseline(self, patient: dict[str, float]) -> dict:
        """Return cached baseline simulation for the given patient."""
        key = tuple(sorted(patient.items()))
        if self._baseline_cache is None or self._baseline_patient_key != key:
            self._baseline_cache = simulate(
                intervention=DEFAULT_INTERVENTION, patient=patient,
            )
            self._baseline_patient_key = key
        return self._baseline_cache

    def run(self, params: dict) -> dict:
        """Execute simulation with the given parameter values.

        Args:
            params: Flat dict mapping parameter names to float values.
                In full mode: all 12 params (6 intervention + 6 patient).
                In intervention-only mode: 6 intervention params.

        Returns:
            Flat dict mapping ``pillar_metric`` names to float values.
            Keys have the form ``"energy_atp_final"``, ``"damage_het_slope"``, etc.
        """
        # Split params into intervention + patient
        intervention = {}
        for name in INTERVENTION_NAMES:
            intervention[name] = float(params.get(name, 0.0))

        if self.intervention_only:
            patient = dict(self.patient_override)
        else:
            patient = {}
            for name in PATIENT_NAMES:
                patient[name] = float(params.get(name, DEFAULT_PATIENT[name]))

        # Run simulation
        result = simulate(intervention=intervention, patient=patient)

        # Get baseline for intervention pillar
        baseline = self._get_baseline(patient)

        # Compute 4-pillar analytics
        analytics = compute_all(result, baseline)

        # Flatten to pillar_metric: float
        flat = {}
        for pillar_name, pillar_metrics in analytics.items():
            for metric_name, value in pillar_metrics.items():
                flat[f"{pillar_name}_{metric_name}"] = value

        # Add top-level trajectory endpoints for convenience
        flat["final_heteroplasmy"] = float(result["heteroplasmy"][-1])
        flat["final_atp"] = float(result["states"][-1, 2])
        flat["final_ros"] = float(result["states"][-1, 3])
        flat["final_nad"] = float(result["states"][-1, 4])
        flat["final_senescent"] = float(result["states"][-1, 5])
        flat["final_membrane_potential"] = float(result["states"][-1, 6])

        return _cap_infs(flat)

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds.

        Returns:
            Dict mapping parameter names to ``(low, high)`` tuples.
            12 keys in full mode, 6 in intervention-only mode.
        """
        spec = {}
        for name, info in INTERVENTION_PARAMS.items():
            spec[name] = info["range"]

        if not self.intervention_only:
            for name, info in PATIENT_PARAMS.items():
                spec[name] = info["range"]

        return spec

    def to_standard_output(self, params: dict) -> dict:
        """Run simulation and return shared output schema dict.

        Produces the zimmerman-toolkit SimulatorOutput format
        for cross-simulator analysis and comparison.
        """
        from zimmerman.output_schema import SimulatorOutput

        # Split params into intervention + patient
        intervention = {}
        for name in INTERVENTION_NAMES:
            intervention[name] = float(params.get(name, 0.0))

        if self.intervention_only:
            patient = dict(self.patient_override)
        else:
            patient = {}
            for name in PATIENT_NAMES:
                patient[name] = float(params.get(name, DEFAULT_PATIENT[name]))

        # Run simulation
        result = simulate(intervention=intervention, patient=patient)

        # Get baseline for intervention pillar
        baseline = self._get_baseline(patient)

        # Compute 4-pillar analytics
        analytics = compute_all(result, baseline)

        # Build extra arrays
        extra = {}
        if "heteroplasmy" in result:
            extra["heteroplasmy"] = result["heteroplasmy"]
        if "deletion_heteroplasmy" in result:
            extra["deletion_heteroplasmy"] = result["deletion_heteroplasmy"]

        output = SimulatorOutput(
            simulator_name="mito",
            simulator_description="8-state mitochondrial aging ODE (30-year horizon)",
            state_dim=8,
            param_dim=len(self.param_spec()),
            state_names=list(STATE_NAMES),
            time_unit="years",
            time_horizon=30.0,
            times=result["time"],
            states=result["states"],
            extra_arrays=extra,
            pillars=analytics,
            input_params=params,
            param_bounds=dict(self.param_spec()),
        )
        return output.to_dict()
