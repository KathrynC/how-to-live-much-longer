"""Grief->Mitochondrial aging bridge.

Connects the grief biological stress simulator (O'Connor 2022, 2025) to the
mitochondrial aging simulator (Cramer, forthcoming 2026) via the disturbance framework.

The grief simulator produces time-varying biological stress signals --
inflammation, cortisol, sympathetic arousal, sleep disruption, and
cardiovascular damage rate -- that feed into the mito ODE as a
GriefDisturbance, perturbing mitochondrial dynamics day by day.

This answers the Phase 2 question: "Can interventions during grief
protect mitochondria?"

Usage:
    from grief_bridge import GriefDisturbance, grief_scenarios
    from disturbances import simulate_with_disturbances

    # Single scenario
    d = GriefDisturbance(
        grief_patient={"B": 0.8, "M": 0.8, "age": 65.0, "E_ctx": 0.2},
        grief_intervention={"act_int": 0.7, "slp_int": 0.8},
    )
    result = simulate_with_disturbances(disturbances=[d])

    # All 8 clinical seeds x 2 intervention levels
    for s in grief_scenarios():
        result = simulate_with_disturbances(disturbances=[s])

Requires:
    grief-simulator (at ~/grief-simulator)
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import numpy.typing as npt

# -- Import mito modules FIRST (before any grief path manipulation) -----------
# This ensures mito's constants, simulator, analytics get cached in sys.modules
# before we touch the grief-simulator, which has identically-named modules.

from disturbances import Disturbance

# -- Load grief-simulator modules via importlib --------------------------------
# Both projects have constants.py, simulator.py, analytics.py. Direct import
# would collide. We use importlib.util to load grief modules by explicit file
# path, temporarily swapping sys.modules so the grief simulator's internal
# imports (e.g., grief's simulator.py importing grief's constants.py) resolve
# to the correct grief versions.

PROJECT = Path(__file__).resolve().parent
GRIEF_PATH = PROJECT.parent / "grief-simulator"


def _load_grief_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Load grief-simulator constants and simulator without namespace pollution.

    Temporarily swaps sys.modules so grief's internal imports resolve correctly,
    then restores the mito modules.
    """
    # Save mito modules currently in sys.modules
    saved: dict[str, types.ModuleType] = {}
    conflict_names = ("constants", "simulator", "analytics")
    for name in conflict_names:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    # Temporarily add grief path at the front
    sys.path.insert(0, str(GRIEF_PATH))

    try:
        # Import grief modules fresh (no cached mito versions in the way)
        import constants as _grief_constants  # noqa: F811
        import simulator as _grief_simulator  # noqa: F811
    finally:
        # Always clean up, even if imports fail
        if str(GRIEF_PATH) in sys.path:
            sys.path.remove(str(GRIEF_PATH))

        # Remove grief modules from sys.modules cache
        for name in conflict_names:
            sys.modules.pop(name, None)

        # Restore mito modules
        for name, mod in saved.items():
            sys.modules[name] = mod

    return _grief_constants, _grief_simulator


_grief_constants, _grief_simulator = _load_grief_modules()

# Re-export grief constants under prefixed names
GRIEF_CLINICAL_SEEDS = _grief_constants.CLINICAL_SEEDS
GRIEF_DEFAULT_INTERVENTION = _grief_constants.DEFAULT_INTERVENTION
GRIEF_DEFAULT_PATIENT = _grief_constants.DEFAULT_PATIENT
GRIEF_INTERVENTION_BOUNDS = _grief_constants.INTERVENTION_BOUNDS
GRIEF_PATIENT_BOUNDS = _grief_constants.PATIENT_BOUNDS

grief_simulate = _grief_simulator.simulate

# State variable indices in the grief simulator (from grief constants.py)
_GRIEF_SNS = 2
_GRIEF_CORT = 3
_GRIEF_INFL = 4
_GRIEF_SLP = 6
_GRIEF_CVD = 10


# -- Mapping coefficients -----------------------------------------------------
# These control how strongly each grief signal perturbs the mito simulator.
# Calibrated to produce effects comparable to existing disturbance types.

INFL_COEFF = 0.4       # grief Infl -> mito inflammation_level (additive)
CORT_COEFF = 0.2       # grief Cort -> mito metabolic_demand (additive)
SNS_ROS_COEFF = 0.05   # grief SNS -> mito ROS state (additive per step)
SLP_COEFF = 0.15       # grief (1-Slp) -> mito inflammation_level (additive)
CVD_VULN_COEFF = 0.3   # grief d(CVD)/dt -> mito genetic_vulnerability (multiplicative)


def grief_trajectory(
    grief_patient: dict[str, float] | None = None,
    grief_intervention: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Run the grief simulator and extract the 5 bridge-relevant curves.

    Returns dict with keys: times, infl, cort, sns, slp, cvd_risk.
    All arrays have the same length (3,651 daily points over 10 years).
    """
    result = grief_simulate(
        intervention=grief_intervention,
        patient=grief_patient,
    )
    states = result["states"]
    times = result["times"]

    return {
        "times": times,
        "infl": states[:, _GRIEF_INFL],
        "cort": states[:, _GRIEF_CORT],
        "sns": states[:, _GRIEF_SNS],
        "slp": states[:, _GRIEF_SLP],
        "cvd_risk": states[:, _GRIEF_CVD],
    }


class GriefDisturbance(Disturbance):
    """Time-varying grief stress disturbance for the mito simulator.

    Pre-computes the grief trajectory at construction time, then
    interpolates the 5 grief signals into the mito ODE at each timestep.

    Args:
        grief_patient: Grief simulator patient params (7D). Defaults to
            grief-simulator's DEFAULT_PATIENT.
        grief_intervention: Grief simulator intervention params (7D).
            Defaults to grief-simulator's DEFAULT_INTERVENTION (no support).
        start_year: When grief onset occurs in the mito simulation timeline.
        duration: How long the grief disturbance is active (default: 10 years,
            matching the grief simulator's full horizon).
        magnitude: Overall scaling factor on all grief->mito effects (0-1).
        label: Human-readable label for this disturbance.
    """

    def __init__(
        self,
        grief_patient: dict[str, float] | None = None,
        grief_intervention: dict[str, float] | None = None,
        start_year: float = 0.0,
        duration: float = 10.0,
        magnitude: float = 1.0,
        label: str | None = None,
    ) -> None:
        name = label or "Grief"
        super().__init__(name, start_year, duration, magnitude)

        # Store grief params for metadata/reproduction
        self.grief_patient = grief_patient or dict(GRIEF_DEFAULT_PATIENT)
        self.grief_intervention = grief_intervention or dict(GRIEF_DEFAULT_INTERVENTION)

        # Pre-compute the grief trajectory
        curves = grief_trajectory(grief_patient, grief_intervention)
        self._grief_times = curves["times"]  # in years, 0..10
        self._grief_infl = curves["infl"]
        self._grief_cort = curves["cort"]
        self._grief_sns = curves["sns"]
        self._grief_slp = curves["slp"]
        self._grief_cvd = curves["cvd_risk"]

        # Pre-compute CVD rate of change (daily diff / dt)
        dt_grief = self._grief_times[1] - self._grief_times[0] if len(self._grief_times) > 1 else 1.0 / 365.0
        cvd_diff = np.diff(self._grief_cvd)
        self._grief_cvd_rate = np.append(cvd_diff, cvd_diff[-1]) / dt_grief

    def _interp(self, arr: np.ndarray, t: float) -> float:
        """Interpolate a grief curve at mito time t."""
        # t is relative to mito sim start; convert to grief-relative time
        grief_t = t - self.start_year
        grief_t = np.clip(grief_t, 0.0, self._grief_times[-1])
        return float(np.interp(grief_t, self._grief_times, arr))

    def modify_state(
        self,
        state: npt.NDArray[np.float64],
        t: float,
    ) -> npt.NDArray[np.float64]:
        """Add SNS-driven ROS to the mito state vector.

        Sympathetic arousal generates reactive oxygen species through
        catecholamine metabolism and mitochondrial ETC stress.
        """
        state = state.copy()
        sns_val = self._interp(self._grief_sns, t)
        # state[3] = ROS
        state[3] += SNS_ROS_COEFF * sns_val * self.magnitude
        return state

    def modify_params(
        self,
        intervention: dict[str, float],
        patient: dict[str, float],
        t: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Perturb mito patient params based on current grief state.

        Four channels:
          1. Grief Infl -> inflammation_level (additive)
          2. Grief Cort -> metabolic_demand (additive)
          3. Grief (1-Slp) -> inflammation_level (additive, stacks with #1)
          4. Grief d(CVD)/dt -> genetic_vulnerability (multiplicative)
        """
        if not self.is_active(t):
            return intervention, patient

        patient = dict(patient)
        mag = self.magnitude

        infl_val = self._interp(self._grief_infl, t)
        cort_val = self._interp(self._grief_cort, t)
        slp_val = self._interp(self._grief_slp, t)
        cvd_rate = self._interp(self._grief_cvd_rate, t)

        # Channel 1 + 3: Inflammation from grief Infl and sleep disruption
        infl_boost = (INFL_COEFF * infl_val + SLP_COEFF * (1.0 - slp_val)) * mag
        patient["inflammation_level"] = min(
            patient.get("inflammation_level", 0.25) + infl_boost, 1.0)

        # Channel 2: Metabolic demand from cortisol
        cort_boost = CORT_COEFF * cort_val * mag
        patient["metabolic_demand"] = min(
            patient.get("metabolic_demand", 1.0) + cort_boost, 2.0)

        # Channel 4: Genetic vulnerability from CVD damage rate
        cvd_rate_clamped = max(cvd_rate, 0.0)
        patient["genetic_vulnerability"] = (
            patient.get("genetic_vulnerability", 1.0) * (1.0 + CVD_VULN_COEFF * cvd_rate_clamped * mag))

        return intervention, patient


# -- Full intervention profile for "with help" scenarios -----------------------

_GRIEF_FULL_SUPPORT: dict[str, float] = {
    "slp_int": 0.8,
    "act_int": 0.7,
    "nut_int": 0.6,
    "alc_int": 0.8,
    "br_int":  0.5,
    "med_int": 0.5,
    "soc_int": 0.7,
}


def grief_scenarios() -> list[GriefDisturbance]:
    """Build GriefDisturbance objects for all 8 clinical seeds x 2 intervention levels.

    Returns 16 disturbances: each clinical seed without intervention (default
    coping) and with full intervention support. These can be passed individually
    to simulate_with_disturbances() for comparison.
    """
    scenarios = []
    for seed in GRIEF_CLINICAL_SEEDS:
        # Without intervention (default coping)
        scenarios.append(GriefDisturbance(
            grief_patient=seed["patient"],
            grief_intervention=None,
            label=f"grief_{seed['name']}_no_support",
        ))
        # With full intervention support
        scenarios.append(GriefDisturbance(
            grief_patient=seed["patient"],
            grief_intervention=_GRIEF_FULL_SUPPORT,
            label=f"grief_{seed['name']}_full_support",
        ))
    return scenarios
