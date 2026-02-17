# Grief→Mito Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the grief biological stress simulator to the mitochondrial aging simulator so grief signals (inflammation, cortisol, SNS, sleep disruption, CVD damage rate) feed into the mito ODE as a time-varying disturbance, answering: "Can interventions during grief protect mitochondria?"

**Architecture:** The grief sim runs first (10-year daily ODE, ~5ms), producing time-varying stress curves. A new `GriefDisturbance(Disturbance)` class interpolates these curves into the mito sim's existing `simulate_with_disturbances()` loop. A Zimmerman adapter (`GriefMitoSimulator`) exposes the combined 28D system to zimmerman-toolkit and cramer-toolkit. Visualization and scenario modules complete the pipeline.

**Tech Stack:** Python 3.11+, numpy (no scipy), matplotlib Agg backend. Both simulators are numpy-only ODE integrators.

**Design doc:** `docs/plans/2026-02-16-grief-mito-integration-design.md`

**Working directory:** `/Users/gardenofcomputation/how-to-live-much-longer/`

**Sibling dependency:** `/Users/gardenofcomputation/grief-simulator/` (imported via sys.path)

---

### Task 1: `grief_bridge.py` — Core Bridge (GriefDisturbance + grief_trajectory)

**Files:**
- Create: `grief_bridge.py`
- Create: `tests/test_grief_bridge.py`
- Reference: `disturbances.py` (Disturbance ABC, simulate_with_disturbances)
- Reference: `~/grief-simulator/simulator.py` (simulate function)
- Reference: `~/grief-simulator/constants.py` (CLINICAL_SEEDS, DEFAULT_INTERVENTION, DEFAULT_PATIENT, INTERVENTION_BOUNDS, PATIENT_BOUNDS)

**Step 1: Write the failing tests**

Create `tests/test_grief_bridge.py`:

```python
"""Tests for grief→mito bridge."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure grief-simulator is importable
GRIEF_PATH = Path(__file__).resolve().parent.parent.parent / "grief-simulator"
if str(GRIEF_PATH) not in sys.path:
    sys.path.insert(0, str(GRIEF_PATH))

from grief_bridge import GriefDisturbance, grief_trajectory, grief_scenarios
from disturbances import Disturbance, simulate_with_disturbances
from simulator import simulate


class TestGriefTrajectory:
    """Test grief_trajectory() helper."""

    def test_returns_five_curves(self):
        curves = grief_trajectory()
        assert set(curves.keys()) == {"infl", "cort", "sns", "slp", "cvd_risk"}

    def test_curves_are_numpy_arrays(self):
        curves = grief_trajectory()
        for name, arr in curves.items():
            assert isinstance(arr, np.ndarray), f"{name} is not ndarray"

    def test_curves_have_matching_lengths(self):
        curves = grief_trajectory()
        lengths = [len(v) for v in curves.values()]
        assert len(set(lengths)) == 1, f"Mismatched lengths: {lengths}"

    def test_times_included(self):
        curves = grief_trajectory()
        assert "times" in curves


class TestGriefDisturbance:
    """Test GriefDisturbance class."""

    def test_is_disturbance_subclass(self):
        d = GriefDisturbance()
        assert isinstance(d, Disturbance)

    def test_has_modify_state(self):
        d = GriefDisturbance()
        assert callable(d.modify_state)

    def test_has_modify_params(self):
        d = GriefDisturbance()
        assert callable(d.modify_params)

    def test_is_active_during_grief_window(self):
        d = GriefDisturbance(start_year=5.0)
        assert d.is_active(6.0)
        assert not d.is_active(4.0)
        assert not d.is_active(16.0)  # default 10-year duration

    def test_default_starts_at_year_zero(self):
        d = GriefDisturbance()
        assert d.start_year == 0.0
        assert d.is_active(0.0)

    def test_custom_grief_patient(self):
        d = GriefDisturbance(grief_patient={"B": 0.9, "M": 0.9, "age": 70.0})
        assert d.is_active(0.0)

    def test_modify_state_adds_ros(self):
        d = GriefDisturbance()
        state = np.array([0.5, 0.3, 0.8, 0.1, 0.6, 0.05, 0.9])
        new_state = d.modify_state(state, 0.5)
        # SNS-driven ROS should increase state[3]
        assert new_state[3] >= state[3]

    def test_modify_params_increases_inflammation(self):
        d = GriefDisturbance()
        intervention = {"rapamycin_dose": 0.0}
        patient = {"inflammation_level": 0.1, "metabolic_demand": 1.0,
                   "genetic_vulnerability": 1.0}
        new_int, new_pat = d.modify_params(intervention, patient, 0.5)
        # Grief inflammation should add to patient inflammation
        assert new_pat["inflammation_level"] >= patient["inflammation_level"]

    def test_modify_params_no_change_outside_window(self):
        d = GriefDisturbance(start_year=5.0)
        patient = {"inflammation_level": 0.1, "metabolic_demand": 1.0,
                   "genetic_vulnerability": 1.0}
        _, new_pat = d.modify_params({}, patient, 1.0)
        assert new_pat["inflammation_level"] == patient["inflammation_level"]


class TestGriefDisturbanceIntegration:
    """Integration tests: grief disturbance in the mito simulator."""

    @pytest.fixture(scope="class")
    def baseline(self):
        return simulate()

    @pytest.fixture(scope="class")
    def bereaved(self):
        d = GriefDisturbance(
            grief_patient={"B": 0.8, "M": 0.8, "age": 65.0, "E_ctx": 0.6},
        )
        return simulate_with_disturbances(disturbances=[d])

    @pytest.fixture(scope="class")
    def bereaved_with_help(self):
        d = GriefDisturbance(
            grief_patient={"B": 0.8, "M": 0.8, "age": 65.0, "E_ctx": 0.6},
            grief_intervention={"act_int": 0.7, "slp_int": 0.8, "soc_int": 0.7},
        )
        return simulate_with_disturbances(disturbances=[d])

    def test_no_nan_in_bereaved(self, bereaved):
        assert not np.any(np.isnan(bereaved["states"]))

    def test_no_negative_states(self, bereaved):
        assert np.all(bereaved["states"] >= -1e-10)

    def test_bereaved_higher_het_than_baseline(self, baseline, bereaved):
        """Grief should cause more mitochondrial damage."""
        assert bereaved["heteroplasmy"][-1] > baseline["heteroplasmy"][-1]

    def test_interventions_reduce_het(self, bereaved, bereaved_with_help):
        """Grief interventions should reduce mitochondrial damage."""
        assert bereaved_with_help["heteroplasmy"][-1] < bereaved["heteroplasmy"][-1]

    def test_bereaved_lower_atp_than_baseline(self, baseline, bereaved):
        """Grief should reduce energy production."""
        assert bereaved["states"][-1, 2] < baseline["states"][-1, 2]

    def test_composes_with_radiation(self, bereaved):
        """Grief + radiation should stack."""
        from disturbances import IonizingRadiation
        shocks = [
            GriefDisturbance(grief_patient={"B": 0.8, "M": 0.8, "age": 65.0}),
            IonizingRadiation(start_year=10.0, magnitude=0.5),
        ]
        result = simulate_with_disturbances(disturbances=shocks)
        assert not np.any(np.isnan(result["states"]))
        # Combined should be worse than grief alone
        assert result["heteroplasmy"][-1] > bereaved["heteroplasmy"][-1]


class TestGriefScenarios:
    """Test grief_scenarios() convenience function."""

    def test_returns_list_of_disturbances(self):
        scenarios = grief_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        for s in scenarios:
            assert isinstance(s, GriefDisturbance)

    def test_includes_with_and_without_intervention(self):
        scenarios = grief_scenarios()
        # Should have pairs: each clinical seed with and without intervention
        assert len(scenarios) == 16  # 8 seeds x 2

    def test_all_scenarios_run_without_crash(self):
        scenarios = grief_scenarios()
        for s in scenarios[:4]:  # test first 4 for speed
            result = simulate_with_disturbances(disturbances=[s])
            assert not np.any(np.isnan(result["states"]))
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_grief_bridge.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'grief_bridge'"

**Step 3: Write implementation**

Create `grief_bridge.py`:

```python
"""Grief→Mitochondrial aging bridge.

Connects the grief biological stress simulator (O'Connor 2022, 2025) to the
mitochondrial aging simulator (Cramer 2025) via the disturbance framework.

The grief simulator produces time-varying biological stress signals —
inflammation, cortisol, sympathetic arousal, sleep disruption, and
cardiovascular damage rate — that feed into the mito ODE as a
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

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from disturbances import Disturbance

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
GRIEF_PATH = PROJECT.parent / "grief-simulator"
if str(GRIEF_PATH) not in sys.path:
    sys.path.insert(0, str(GRIEF_PATH))

# Grief simulator imports (prefixed to avoid collision with mito constants)
from constants import (  # noqa: E402
    CLINICAL_SEEDS as GRIEF_CLINICAL_SEEDS,
    DEFAULT_INTERVENTION as GRIEF_DEFAULT_INTERVENTION,
    DEFAULT_PATIENT as GRIEF_DEFAULT_PATIENT,
    INTERVENTION_BOUNDS as GRIEF_INTERVENTION_BOUNDS,
    PATIENT_BOUNDS as GRIEF_PATIENT_BOUNDS,
)
from simulator import simulate as grief_simulate  # noqa: E402

# Remove grief-simulator from sys.path to avoid polluting the namespace
# for subsequent imports from the mito project
sys.path.remove(str(GRIEF_PATH))

# State variable indices in the grief simulator (from grief constants.py)
_GRIEF_SNS = 2
_GRIEF_CORT = 3
_GRIEF_INFL = 4
_GRIEF_SLP = 6
_GRIEF_CVD = 10


# ── Mapping coefficients ────────────────────────────────────────────────────
# These control how strongly each grief signal perturbs the mito simulator.
# Calibrated to produce effects comparable to existing disturbance types.

INFL_COEFF = 0.4       # grief Infl → mito inflammation_level (additive)
CORT_COEFF = 0.2       # grief Cort → mito metabolic_demand (additive)
SNS_ROS_COEFF = 0.05   # grief SNS → mito ROS state (additive per step)
SLP_COEFF = 0.15       # grief (1-Slp) → mito inflammation_level (additive)
CVD_VULN_COEFF = 0.3   # grief d(CVD)/dt → mito genetic_vulnerability (multiplicative)


def grief_trajectory(
    grief_patient: dict[str, float] | None = None,
    grief_intervention: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Run the grief simulator and extract the 5 bridge-relevant curves.

    Returns dict with keys: times, infl, cort, sns, slp, cvd_risk.
    All arrays have the same length (3,650 daily points over 10 years).
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
        magnitude: Overall scaling factor on all grief→mito effects (0-1).
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
          1. Grief Infl → inflammation_level (additive)
          2. Grief Cort → metabolic_demand (additive)
          3. Grief (1-Slp) → inflammation_level (additive, stacks with #1)
          4. Grief d(CVD)/dt → genetic_vulnerability (multiplicative)
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


# ── Full intervention profile for "with help" scenarios ──────────────────────

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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_grief_bridge.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add grief_bridge.py tests/test_grief_bridge.py
git commit -m "feat: grief→mito bridge with GriefDisturbance and 5-channel mapping"
```

---

### Task 2: `grief_mito_simulator.py` — Zimmerman Protocol Adapter (28D)

**Files:**
- Create: `grief_mito_simulator.py`
- Add tests to: `tests/test_grief_bridge.py`
- Reference: `zimmerman_bridge.py` (MitoSimulator pattern)
- Reference: `grief_bridge.py` (GriefDisturbance)

**Step 1: Write the failing tests**

Append to `tests/test_grief_bridge.py`:

```python
from grief_mito_simulator import GriefMitoSimulator


class TestGriefMitoSimulator:
    """Test the Zimmerman adapter for the combined system."""

    def test_has_run_method(self):
        sim = GriefMitoSimulator()
        assert callable(sim.run)

    def test_has_param_spec(self):
        sim = GriefMitoSimulator()
        spec = sim.param_spec()
        assert isinstance(spec, dict)

    def test_param_spec_has_grief_and_mito_params(self):
        sim = GriefMitoSimulator()
        spec = sim.param_spec()
        # Should have grief params (prefixed)
        assert "grief_B" in spec
        assert "grief_age" in spec
        assert "grief_slp_int" in spec
        # Should have mito params
        assert "baseline_age" in spec
        assert "rapamycin_dose" in spec

    def test_run_with_empty_params(self):
        sim = GriefMitoSimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_run_returns_flat_dict_of_floats(self):
        sim = GriefMitoSimulator()
        result = sim.run({})
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is {type(v)}"

    def test_run_includes_grief_metrics(self):
        sim = GriefMitoSimulator()
        result = sim.run({})
        # Should include grief-side metrics
        assert "grief_pgd_risk_score" in result

    def test_run_includes_mito_metrics(self):
        sim = GriefMitoSimulator()
        result = sim.run({})
        assert "final_heteroplasmy" in result
        assert "final_atp" in result

    def test_no_nan_in_output(self):
        sim = GriefMitoSimulator()
        result = sim.run({})
        for k, v in result.items():
            assert not np.isnan(v), f"{k} is NaN"

    def test_no_inf_in_output(self):
        sim = GriefMitoSimulator()
        result = sim.run({})
        for k, v in result.items():
            assert not np.isinf(v), f"{k} is inf"

    def test_grief_intervention_affects_mito_outcome(self):
        sim = GriefMitoSimulator()
        no_help = sim.run({"grief_B": 0.9, "grief_M": 0.9})
        with_help = sim.run({"grief_B": 0.9, "grief_M": 0.9,
                             "grief_act_int": 0.8, "grief_slp_int": 0.8})
        # Interventions should reduce mitochondrial damage
        assert with_help["final_heteroplasmy"] < no_help["final_heteroplasmy"]
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_grief_bridge.py::TestGriefMitoSimulator -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'grief_mito_simulator'"

**Step 3: Write implementation**

Create `grief_mito_simulator.py`:

```python
"""Zimmerman protocol adapter for the combined grief→mito system.

Exposes a 28-dimensional parameter space (14 grief + 14 mito) through the
standard Simulator protocol (run + param_spec). This makes the combined
system compatible with all 14 zimmerman-toolkit interrogation tools and
the cramer-toolkit resilience analysis framework.

Usage:
    from grief_mito_simulator import GriefMitoSimulator

    sim = GriefMitoSimulator()
    spec = sim.param_spec()    # 28 params with bounds
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

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
GRIEF_PATH = PROJECT.parent / "grief-simulator"
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"

for p in (GRIEF_PATH, ZIMMERMAN_PATH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Project imports
from grief_bridge import GriefDisturbance  # noqa: E402
from disturbances import simulate_with_disturbances  # noqa: E402
from constants import (  # noqa: E402
    INTERVENTION_PARAMS as MITO_INTERVENTION_PARAMS,
    PATIENT_PARAMS as MITO_PATIENT_PARAMS,
    INTERVENTION_NAMES as MITO_INTERVENTION_NAMES,
    PATIENT_NAMES as MITO_PATIENT_NAMES,
    DEFAULT_INTERVENTION as MITO_DEFAULT_INTERVENTION,
    DEFAULT_PATIENT as MITO_DEFAULT_PATIENT,
)
from simulator import simulate as mito_simulate  # noqa: E402
from analytics import compute_all as mito_compute_all  # noqa: E402

# Grief simulator param specs (via grief_bridge's cached imports)
from grief_bridge import (  # noqa: E402
    GRIEF_INTERVENTION_BOUNDS,
    GRIEF_PATIENT_BOUNDS,
    GRIEF_DEFAULT_INTERVENTION,
    GRIEF_DEFAULT_PATIENT,
    grief_trajectory,
)

# Grief analytics (for grief-side metrics)
# Temporarily add grief path for analytics import
if str(GRIEF_PATH) not in sys.path:
    sys.path.insert(0, str(GRIEF_PATH))
from analytics import compute_all as grief_compute_all  # noqa: E402
if str(GRIEF_PATH) in sys.path:
    sys.path.remove(str(GRIEF_PATH))

# ── Prefix convention ───────────────────────────────────────────────────────
# Grief params are prefixed with "grief_" to avoid name collisions.
# Mito params keep their original names.

GRIEF_PREFIX = "grief_"

INF_CAP = 999.0


class GriefMitoSimulator:
    """Zimmerman Simulator adapter for the combined grief→mito system.

    28D input: 14 grief params (prefixed "grief_") + 14 mito params.
    Output: grief-side metrics (prefixed "grief_") + mito-side metrics.
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return all 28 parameter bounds."""
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
        """Run combined grief→mito simulation."""
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
                if k in [name for name in MITO_INTERVENTION_NAMES]:
                    mito_intervention[k] = v
                elif k in [name for name in MITO_PATIENT_NAMES]:
                    mito_patient[k] = v

        # Fill mito defaults
        for name in MITO_INTERVENTION_NAMES:
            mito_intervention.setdefault(name, MITO_DEFAULT_INTERVENTION[name])
        for name in MITO_PATIENT_NAMES:
            mito_patient.setdefault(name, MITO_DEFAULT_PATIENT[name])

        # Run grief sim → disturbance → mito sim
        grief_disturbance = GriefDisturbance(
            grief_patient=grief_patient,
            grief_intervention=grief_intervention,
        )
        mito_result = simulate_with_disturbances(
            intervention=mito_intervention,
            patient=mito_patient,
            disturbances=[grief_disturbance],
        )

        # Compute mito analytics
        mito_baseline = mito_simulate(
            intervention=MITO_DEFAULT_INTERVENTION,
            patient=mito_patient,
        )
        mito_analytics = mito_compute_all(mito_result, mito_baseline)

        # Compute grief analytics
        grief_result = {
            "states": grief_disturbance._grief_times,  # placeholder
            "times": grief_disturbance._grief_times,
        }
        # Re-run grief sim for full analytics
        from grief_bridge import grief_trajectory as _gt
        _grief_sim_result = {
            "states": np.column_stack([
                grief_disturbance._grief_infl,  # need full state
            ]),
            "times": grief_disturbance._grief_times,
        }
        # Actually just re-run the grief sim for clean analytics
        if str(GRIEF_PATH) not in sys.path:
            sys.path.insert(0, str(GRIEF_PATH))
        from simulator import simulate as _grief_sim
        grief_full = _grief_sim(
            intervention=grief_intervention,
            patient=grief_patient,
        )
        grief_analytics = grief_compute_all(grief_full)
        if str(GRIEF_PATH) in sys.path:
            sys.path.remove(str(GRIEF_PATH))

        # Flatten mito metrics
        flat = {}
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
        out = {}
        for k, v in flat.items():
            if math.isinf(v):
                v = INF_CAP if v > 0 else -INF_CAP
            if math.isnan(v):
                v = 0.0
            out[k] = v

        return out
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_grief_bridge.py::TestGriefMitoSimulator -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add grief_mito_simulator.py tests/test_grief_bridge.py
git commit -m "feat: GriefMitoSimulator Zimmerman adapter (28D combined system)"
```

---

### Task 3: `grief_mito_scenarios.py` — Cramer-Toolkit Scenario Bank

**Files:**
- Create: `grief_mito_scenarios.py`
- Add tests to: `tests/test_grief_bridge.py`
- Reference: `cramer_bridge.py` (scenario bank pattern, ScenarioSet, PROTOCOLS)

**Step 1: Write the failing tests**

Append to `tests/test_grief_bridge.py`:

```python
from grief_mito_scenarios import (
    GRIEF_STRESS_SCENARIOS,
    GRIEF_PROTOCOLS,
    grief_scenario_disturbances,
)


class TestGriefMitoScenarios:
    """Test cramer-toolkit grief scenario bank."""

    def test_scenarios_is_list(self):
        assert isinstance(GRIEF_STRESS_SCENARIOS, list)

    def test_scenarios_not_empty(self):
        assert len(GRIEF_STRESS_SCENARIOS) > 0

    def test_each_scenario_has_name_and_disturbance(self):
        for s in GRIEF_STRESS_SCENARIOS:
            assert "name" in s
            assert "disturbance" in s
            assert isinstance(s["disturbance"], GriefDisturbance)

    def test_protocols_is_dict(self):
        assert isinstance(GRIEF_PROTOCOLS, dict)
        assert "no_grief_support" in GRIEF_PROTOCOLS
        assert "full_grief_support" in GRIEF_PROTOCOLS

    def test_grief_scenario_disturbances_returns_list(self):
        disturbances = grief_scenario_disturbances("spouse_sudden_65")
        assert isinstance(disturbances, list)
        assert len(disturbances) == 2  # with and without support

    def test_all_scenarios_simulate_without_nan(self):
        for s in GRIEF_STRESS_SCENARIOS[:4]:  # first 4 for speed
            result = simulate_with_disturbances(disturbances=[s["disturbance"]])
            assert not np.any(np.isnan(result["states"])), f"NaN in {s['name']}"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_grief_bridge.py::TestGriefMitoScenarios -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'grief_mito_scenarios'"

**Step 3: Write implementation**

Create `grief_mito_scenarios.py`:

```python
"""Grief-derived stress scenarios for cramer-toolkit resilience analysis.

Defines 16 grief stress scenarios (8 clinical seeds x 2 intervention levels)
and named grief intervention protocols, compatible with the cramer-toolkit's
scenario-based analysis framework.

Usage:
    from grief_mito_scenarios import (
        GRIEF_STRESS_SCENARIOS, GRIEF_PROTOCOLS,
        grief_scenario_disturbances,
    )
    from disturbances import simulate_with_disturbances

    # Run a specific grief scenario through the mito simulator
    for scenario in GRIEF_STRESS_SCENARIOS:
        result = simulate_with_disturbances(disturbances=[scenario["disturbance"]])
        print(f"{scenario['name']}: het={result['heteroplasmy'][-1]:.4f}")

Requires:
    grief-simulator (at ~/grief-simulator)
"""
from __future__ import annotations

from grief_bridge import (
    GriefDisturbance,
    GRIEF_CLINICAL_SEEDS,
    GRIEF_DEFAULT_INTERVENTION,
)

# ── Grief intervention protocols ─────────────────────────────────────────────
# Named profiles matching O'Connor's behavioral recommendations.

GRIEF_PROTOCOLS: dict[str, dict[str, float]] = {
    "no_grief_support": dict(GRIEF_DEFAULT_INTERVENTION),
    "minimal_support": {
        "slp_int": 0.3, "act_int": 0.2, "nut_int": 0.2,
        "alc_int": 0.5, "br_int": 0.0, "med_int": 0.0, "soc_int": 0.3,
    },
    "moderate_support": {
        "slp_int": 0.5, "act_int": 0.5, "nut_int": 0.5,
        "alc_int": 0.7, "br_int": 0.3, "med_int": 0.3, "soc_int": 0.5,
    },
    "full_grief_support": {
        "slp_int": 0.8, "act_int": 0.7, "nut_int": 0.6,
        "alc_int": 0.8, "br_int": 0.5, "med_int": 0.5, "soc_int": 0.7,
    },
}


# ── Build scenario bank ─────────────────────────────────────────────────────

def _build_scenarios() -> list[dict]:
    """Build all grief stress scenarios from clinical seeds."""
    scenarios = []
    for seed in GRIEF_CLINICAL_SEEDS:
        # Without intervention
        scenarios.append({
            "name": f"{seed['name']}_no_support",
            "description": f"{seed['description']} — no grief support",
            "seed": seed["name"],
            "intervention": "no_grief_support",
            "disturbance": GriefDisturbance(
                grief_patient=seed["patient"],
                grief_intervention=None,
                label=f"grief_{seed['name']}_no_support",
            ),
        })
        # With full support
        scenarios.append({
            "name": f"{seed['name']}_full_support",
            "description": f"{seed['description']} — full grief support",
            "seed": seed["name"],
            "intervention": "full_grief_support",
            "disturbance": GriefDisturbance(
                grief_patient=seed["patient"],
                grief_intervention=GRIEF_PROTOCOLS["full_grief_support"],
                label=f"grief_{seed['name']}_full_support",
            ),
        })
    return scenarios


GRIEF_STRESS_SCENARIOS: list[dict] = _build_scenarios()


def grief_scenario_disturbances(seed_name: str) -> list[GriefDisturbance]:
    """Get the with/without-support disturbance pair for a clinical seed."""
    return [
        s["disturbance"] for s in GRIEF_STRESS_SCENARIOS
        if s["seed"] == seed_name
    ]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_grief_bridge.py::TestGriefMitoScenarios -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add grief_mito_scenarios.py tests/test_grief_bridge.py
git commit -m "feat: grief stress scenario bank for cramer-toolkit integration"
```

---

### Task 4: `grief_mito_viz.py` — Visualization

**Files:**
- Create: `grief_mito_viz.py`
- Reference: `resilience_viz.py` (plotting patterns)
- Reference: `~/grief-simulator/visualize.py` (grief panel layout)

**Step 1: Write a smoke test**

Append to `tests/test_grief_bridge.py`:

```python
import os
from grief_mito_viz import (
    plot_grief_mito_trajectory,
    plot_intervention_comparison,
    plot_all_grief_scenarios,
)


class TestGriefMitoViz:
    """Smoke tests for grief-mito visualization."""

    def test_plot_trajectory_creates_file(self, tmp_path):
        d = GriefDisturbance()
        mito_result = simulate_with_disturbances(disturbances=[d])
        grief_curves = grief_trajectory()
        out = str(tmp_path / "test_traj.png")
        plot_grief_mito_trajectory(grief_curves, mito_result, "Test", out)
        assert os.path.exists(out)

    def test_plot_comparison_creates_file(self, tmp_path):
        out = str(tmp_path / "test_compare.png")
        plot_intervention_comparison(
            grief_patient={"B": 0.8, "M": 0.8, "age": 65.0},
            output_path=out,
        )
        assert os.path.exists(out)

    def test_plot_all_scenarios_creates_files(self, tmp_path):
        out_dir = str(tmp_path / "viz_output")
        plot_all_grief_scenarios(output_dir=out_dir, max_scenarios=2)
        assert os.path.isdir(out_dir)
        files = os.listdir(out_dir)
        assert len(files) >= 2
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_grief_bridge.py::TestGriefMitoViz -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'grief_mito_viz'"

**Step 3: Write implementation**

Create `grief_mito_viz.py`:

```python
"""Visualization for the grief→mitochondrial aging integration.

Side-by-side panels showing grief trajectories (left) and mitochondrial
trajectories (right), with the 5 mapping channels highlighted.

Usage:
    from grief_mito_viz import plot_all_grief_scenarios
    plot_all_grief_scenarios()           # generates all plots to output/grief_mito/

    from grief_mito_viz import plot_intervention_comparison
    plot_intervention_comparison(
        grief_patient={"B": 0.8, "M": 0.8, "age": 65.0, "E_ctx": 0.2},
        output_path="output/grief_mito/comparison.png",
    )

Uses Agg backend (headless).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grief_bridge import (
    GriefDisturbance,
    grief_trajectory,
    GRIEF_CLINICAL_SEEDS,
    GRIEF_DEFAULT_INTERVENTION,
)
from grief_mito_scenarios import GRIEF_PROTOCOLS
from disturbances import simulate_with_disturbances
from simulator import simulate


def plot_grief_mito_trajectory(
    grief_curves: dict,
    mito_result: dict,
    title: str,
    output_path: str,
) -> None:
    """2x4 panel plot: grief (left column) + mito (right column)."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    grief_t = grief_curves["times"]
    mito_t = mito_result["time"]
    mito_s = mito_result["states"]

    # Left column: Grief signals
    axes[0, 0].plot(grief_t, grief_curves["infl"], color="C5", label="Inflammation")
    axes[0, 0].set_title("Grief: Inflammation")
    axes[0, 0].set_ylabel("Normalized")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(grief_t, grief_curves["cort"], color="C3", label="Cortisol")
    axes[1, 0].set_title("Grief: Cortisol")
    axes[1, 0].set_ylabel("Normalized")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(grief_t, grief_curves["sns"], color="C2", label="SNS")
    axes[2, 0].plot(grief_t, grief_curves["slp"], color="C9", label="Sleep", linestyle="--")
    axes[2, 0].set_title("Grief: SNS & Sleep")
    axes[2, 0].set_ylabel("Normalized")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    axes[3, 0].plot(grief_t, grief_curves["cvd_risk"], color="red", label="CVD risk")
    axes[3, 0].set_title("Grief: CVD Risk (cumulative)")
    axes[3, 0].set_xlabel("Time (years)")
    axes[3, 0].set_ylabel("Cumulative")
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].grid(True, alpha=0.3)

    # Right column: Mito response
    het = mito_result["heteroplasmy"]
    axes[0, 1].plot(mito_t, het, color="C1", label="Heteroplasmy")
    axes[0, 1].axhline(y=0.7, color="red", linestyle=":", alpha=0.5, label="Cliff (0.70)")
    axes[0, 1].set_title("Mito: Heteroplasmy")
    axes[0, 1].set_ylabel("Fraction damaged")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(mito_t, mito_s[:, 2], color="C0", label="ATP")
    axes[1, 1].set_title("Mito: ATP Production")
    axes[1, 1].set_ylabel("MU/day")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 1].plot(mito_t, mito_s[:, 3], color="C4", label="ROS")
    axes[2, 1].plot(mito_t, mito_s[:, 5], color="C8", label="Senescence", linestyle="--")
    axes[2, 1].set_title("Mito: ROS & Senescence")
    axes[2, 1].set_ylabel("Normalized")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    axes[3, 1].plot(mito_t, mito_s[:, 4], color="C6", label="NAD+")
    axes[3, 1].plot(mito_t, mito_s[:, 6], color="C7", label="Membrane Potential", linestyle="--")
    axes[3, 1].set_title("Mito: NAD+ & Membrane Potential")
    axes[3, 1].set_xlabel("Time (years)")
    axes[3, 1].set_ylabel("Normalized")
    axes[3, 1].legend(fontsize=8)
    axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_intervention_comparison(
    grief_patient: dict[str, float] | None = None,
    mito_patient: dict[str, float] | None = None,
    output_path: str = "output/grief_mito/intervention_comparison.png",
    title: str = "Grief Interventions → Mitochondrial Impact",
) -> None:
    """Compare mito outcomes with and without grief interventions."""
    baseline_mito = simulate(patient=mito_patient)

    protocols = {
        "No grief": None,
        "No support": GRIEF_PROTOCOLS["no_grief_support"],
        "Moderate": GRIEF_PROTOCOLS["moderate_support"],
        "Full support": GRIEF_PROTOCOLS["full_grief_support"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = ["gray", "C3", "C1", "C2"]

    for i, (label, grief_int) in enumerate(protocols.items()):
        if label == "No grief":
            result = baseline_mito
            het = result["heteroplasmy"]
            t = result["time"]
            atp = result["states"][:, 2]
        else:
            d = GriefDisturbance(
                grief_patient=grief_patient,
                grief_intervention=grief_int,
            )
            result = simulate_with_disturbances(
                patient=mito_patient,
                disturbances=[d],
            )
            het = result["heteroplasmy"]
            t = result["time"]
            atp = result["states"][:, 2]

        axes[0].plot(t, het, label=label, color=colors[i],
                     linewidth=2 if label == "No grief" else 1.5)
        axes[1].plot(t, atp, label=label, color=colors[i],
                     linewidth=2 if label == "No grief" else 1.5)

    # Bar chart of final heteroplasmy
    final_hets = []
    for label, grief_int in protocols.items():
        if label == "No grief":
            final_hets.append(baseline_mito["heteroplasmy"][-1])
        else:
            d = GriefDisturbance(grief_patient=grief_patient, grief_intervention=grief_int)
            r = simulate_with_disturbances(patient=mito_patient, disturbances=[d])
            final_hets.append(r["heteroplasmy"][-1])
    axes[2].bar(list(protocols.keys()), final_hets, color=colors)
    axes[2].set_ylabel("Final Heteroplasmy")
    axes[2].set_title("30-Year Mitochondrial Damage")
    axes[2].axhline(y=0.7, color="red", linestyle=":", alpha=0.5)

    axes[0].set_title("Heteroplasmy Over Time")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Heteroplasmy")
    axes[0].axhline(y=0.7, color="red", linestyle=":", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("ATP Production Over Time")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("ATP (MU/day)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_grief_scenarios(
    output_dir: str = "output/grief_mito",
    max_scenarios: int | None = None,
) -> None:
    """Generate trajectory plots for all grief clinical scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    seeds = GRIEF_CLINICAL_SEEDS
    if max_scenarios is not None:
        seeds = seeds[:max_scenarios]

    results_no = []
    results_yes = []
    labels = []

    for seed in seeds:
        labels.append(seed["name"])

        # Without support
        d_no = GriefDisturbance(grief_patient=seed["patient"], grief_intervention=None)
        r_no = simulate_with_disturbances(disturbances=[d_no])
        results_no.append(r_no)

        # With support
        d_yes = GriefDisturbance(
            grief_patient=seed["patient"],
            grief_intervention=GRIEF_PROTOCOLS["full_grief_support"],
        )
        r_yes = simulate_with_disturbances(disturbances=[d_yes])
        results_yes.append(r_yes)

        # Individual trajectory
        grief_curves = grief_trajectory(seed["patient"])
        plot_grief_mito_trajectory(
            grief_curves, r_no,
            f"{seed['name']}: {seed['description']}",
            os.path.join(output_dir, f"grief_mito_{seed['name']}.png"),
        )

    # Comparison overlay: final heteroplasmy with/without support
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grief→Mito Impact: All Scenarios", fontsize=14, fontweight="bold")

    x = np.arange(len(labels))
    hets_no = [r["heteroplasmy"][-1] for r in results_no]
    hets_yes = [r["heteroplasmy"][-1] for r in results_yes]

    axes[0].bar(x - 0.2, hets_no, 0.4, label="No support", color="C3")
    axes[0].bar(x + 0.2, hets_yes, 0.4, label="Full support", color="C2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Final Heteroplasmy")
    axes[0].set_title("30-Year Mitochondrial Damage")
    axes[0].axhline(y=0.7, color="red", linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Protection percentage
    protection = [(no - yes) / no * 100 if no > 0 else 0
                  for no, yes in zip(hets_no, hets_yes)]
    axes[1].bar(x, protection, color="C0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Heteroplasmy Reduction (%)")
    axes[1].set_title("Mitochondrial Protection from Grief Interventions")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "grief_mito_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/grief_mito_comparison.png")


if __name__ == "__main__":
    print("Generating grief→mito visualizations...")
    plot_all_grief_scenarios()
    print("Done.")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_grief_bridge.py::TestGriefMitoViz -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add grief_mito_viz.py tests/test_grief_bridge.py
git commit -m "feat: grief→mito visualization (trajectory + comparison + scenario plots)"
```

---

### Task 5: Full Test Suite + Calibration Check

**Files:**
- Modify: `tests/test_grief_bridge.py` (if any adjustments needed)
- Reference: All new files

**Step 1: Run the full mito test suite to verify nothing existing is broken**

Run: `python -m pytest tests/ -v`
Expected: All existing tests PASS (163+), plus all new grief bridge tests PASS

**Step 2: Run the grief simulator tests to verify they still pass**

Run: `cd ~/grief-simulator && python -m pytest tests/ -v`
Expected: 37/37 PASS

**Step 3: Calibration sanity check**

Run a quick Python script to verify the integration produces biologically reasonable results:

```python
python -c "
from grief_bridge import GriefDisturbance
from disturbances import simulate_with_disturbances
from simulator import simulate

baseline = simulate()
d = GriefDisturbance(grief_patient={'B': 0.8, 'M': 0.8, 'age': 65.0, 'E_ctx': 0.6})
bereaved = simulate_with_disturbances(disturbances=[d])

print('Baseline:')
print(f'  Final het: {baseline[\"heteroplasmy\"][-1]:.4f}')
print(f'  Final ATP: {baseline[\"states\"][-1, 2]:.4f}')
print()
print('Bereaved (no support):')
print(f'  Final het: {bereaved[\"heteroplasmy\"][-1]:.4f}')
print(f'  Final ATP: {bereaved[\"states\"][-1, 2]:.4f}')
print()
print(f'Het increase: {bereaved[\"heteroplasmy\"][-1] - baseline[\"heteroplasmy\"][-1]:.4f}')
print(f'ATP decrease: {baseline[\"states\"][-1, 2] - bereaved[\"states\"][-1, 2]:.4f}')
"
```

Expected: Het increase is positive (grief damages mitochondria). ATP decrease is positive (grief reduces energy). Values should be modest (grief is not chemo — het increase should be ~0.01-0.05, not 0.3).

If coefficients need tuning, adjust `INFL_COEFF`, `CORT_COEFF`, `SNS_ROS_COEFF`, `SLP_COEFF`, `CVD_VULN_COEFF` in `grief_bridge.py` and re-run.

**Step 4: Generate visualizations**

Run: `python grief_mito_viz.py`
Expected: Plots generated to `output/grief_mito/`. Visually inspect: grief curves on left should show the expected patterns (inflammation spike, cortisol spike, sleep dip), mito curves on right should show slightly accelerated aging compared to baseline.

**Step 5: Commit**

```bash
git add -A
git commit -m "test: full integration verification, all tests pass"
```

---

### Task 6: Documentation + CLAUDE.md Updates

**Files:**
- Modify: `CLAUDE.md` (add grief bridge section)
- Modify: `~/CLAUDE.md` (update workspace overview)

**Step 1: Update how-to-live-much-longer/CLAUDE.md**

Add after the cramer-toolkit section in the dependency graph:

```
# Grief→Mito Integration (Phase 2, requires ~/grief-simulator)
grief_bridge.py            ← GriefDisturbance + grief_trajectory() + grief_scenarios()
grief_mito_simulator.py    ← GriefMitoSimulator: Zimmerman adapter for 28D combined system
grief_mito_viz.py          ← Side-by-side grief/mito visualization
grief_mito_scenarios.py    ← Grief stress scenario bank for cramer-toolkit
```

Add a commands section:

```bash
# Grief→Mito integration (requires ~/grief-simulator)
python grief_mito_viz.py                # Generate all grief→mito plots
python -m pytest tests/test_grief_bridge.py -v  # Grief bridge tests
python -c "from grief_mito_simulator import GriefMitoSimulator; s = GriefMitoSimulator(); print(s.run({}))"
```

**Step 2: Update workspace CLAUDE.md**

Update the how-to-live-much-longer description to mention Phase 2 is complete.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add grief→mito integration to CLAUDE.md"
```

---

### Task 7: Push to GitHub

**Step 1: Run full test suite one final time**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Push**

```bash
git push
```

---

### Progress checkpoint

After each task, update `docs/plans/progress.json`:

```json
{
  "plan": "2026-02-16-grief-mito-integration.md",
  "total_tasks": 7,
  "completed": [],
  "in_progress": null,
  "next": 1,
  "log": []
}
```
