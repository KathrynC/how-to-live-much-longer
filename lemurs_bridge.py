"""LEMURS->Mitochondrial aging bridge.

Connects the LEMURS college student sleep/stress/anxiety ODE simulator
(Bloomfield et al. 2024-2025, UVM LEMURS study) to the mitochondrial
aging simulator (Cramer, forthcoming 2026) via the disturbance framework.

The LEMURS simulator produces time-varying biopsychosocial signals --
sleep duration, perceived stress, anxiety, heart rate variability, and
directed attention capacity -- that feed into the mito ODE as a
LEMURSDisturbance, perturbing mitochondrial dynamics semester by semester
over the 4-year college window.

This answers the question: "Does chronic college stress leave a lasting
mitochondrial signature, and can well-being interventions protect
against it?"

Usage:
    from lemurs_bridge import LEMURSDisturbance, lemurs_scenarios
    from disturbances import simulate_with_disturbances

    # Single scenario
    d = LEMURSDisturbance(
        lemurs_patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0},
        lemurs_intervention={"nature_rx": 0.8, "exercise_rx": 0.6},
    )
    result = simulate_with_disturbances(disturbances=[d])

    # All 8 archetypes x 2 intervention levels
    for s in lemurs_scenarios():
        result = simulate_with_disturbances(disturbances=[s])

Requires:
    lemurs-simulator (at ~/lemurs-simulator)
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import numpy.typing as npt

# -- Import mito modules FIRST (before any LEMURS path manipulation) ----------
# This ensures mito's constants, simulator, analytics get cached in sys.modules
# before we touch the lemurs-simulator, which has identically-named modules.

from disturbances import Disturbance

# -- Load lemurs-simulator modules via importlib ------------------------------
# Both projects have constants.py, simulator.py, analytics.py. Direct import
# would collide. We use the same importlib isolation pattern as grief_bridge.py:
# temporarily swap sys.modules so the LEMURS simulator's internal imports
# resolve to the correct LEMURS versions.

PROJECT = Path(__file__).resolve().parent
LEMURS_PATH = PROJECT.parent / "lemurs-simulator"


def _load_lemurs_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Load lemurs-simulator constants and simulator without namespace pollution.

    Temporarily swaps sys.modules so LEMURS's internal imports resolve correctly,
    then restores the mito modules.
    """
    # Save mito modules currently in sys.modules
    saved: dict[str, types.ModuleType] = {}
    conflict_names = ("constants", "simulator", "analytics")
    for name in conflict_names:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    # Temporarily add LEMURS path at the front
    sys.path.insert(0, str(LEMURS_PATH))

    try:
        # Import LEMURS modules fresh (no cached mito versions in the way)
        import constants as _lemurs_constants  # noqa: F811
        import simulator as _lemurs_simulator  # noqa: F811
    finally:
        # Always clean up, even if imports fail
        if str(LEMURS_PATH) in sys.path:
            sys.path.remove(str(LEMURS_PATH))

        # Remove LEMURS modules from sys.modules cache
        for name in conflict_names:
            sys.modules.pop(name, None)

        # Restore mito modules
        for name, mod in saved.items():
            sys.modules[name] = mod

    return _lemurs_constants, _lemurs_simulator


_lemurs_constants, _lemurs_simulator = _load_lemurs_modules()

# Re-export LEMURS constants under prefixed names
LEMURS_STUDENT_ARCHETYPES = _lemurs_constants.STUDENT_ARCHETYPES
LEMURS_DEFAULT_INTERVENTION = _lemurs_constants.DEFAULT_INTERVENTION
LEMURS_DEFAULT_PATIENT = _lemurs_constants.DEFAULT_PATIENT
LEMURS_INTERVENTION_BOUNDS = _lemurs_constants.INTERVENTION_BOUNDS
LEMURS_PATIENT_BOUNDS = _lemurs_constants.PATIENT_BOUNDS

lemurs_simulate = _lemurs_simulator.simulate

# State variable indices in the LEMURS simulator (from lemurs constants.py)
_LEMURS_TST = 0    # Total Sleep Time (hours)
_LEMURS_PSS = 2    # Perceived Stress Scale (0-40)
_LEMURS_GAD = 3    # GAD-7 anxiety (0-21)
_LEMURS_HRV = 8    # Heart Rate Variability (ms RMSSD)
_LEMURS_DAC = 13   # Directed Attention Capacity (0-1)


# -- Mapping coefficients from mito constants.py -----------------------------
# These control how strongly each LEMURS signal perturbs the mito simulator.

from constants import (
    LEMURS_TST_INFL_COEFF, LEMURS_PSS_DEMAND_COEFF, LEMURS_GAD_ROS_COEFF,
    LEMURS_HRV_VULN_COEFF, LEMURS_DAC_REPAIR_COEFF,
    LEMURS_COLLEGE_START_AGE, LEMURS_COLLEGE_END_AGE,
    LEMURS_SEMESTER_WEEKS, LEMURS_SEMESTERS_PER_YEAR,
    LEMURS_ALLOSTATIC_RATE, LEMURS_POST_COLLEGE_DECAY,
    LEMURS_TST_MIN, LEMURS_TST_MAX,
    LEMURS_PSS_MAX, LEMURS_GAD_MAX,
    LEMURS_HRV_MIN, LEMURS_HRV_MAX,
)


def lemurs_trajectory(
    lemurs_patient: dict | None = None,
    lemurs_intervention: dict | None = None,
) -> dict[str, np.ndarray]:
    """Run the LEMURS simulator and extract the 5 bridge-relevant curves.

    Returns dict with keys: times, tst, pss, gad7, hrv, dac.
    All arrays have the same length (106 points over 15 weeks).
    LEMURS simulate returns {"states": (106,14), "times": (106,)}.
    """
    result = lemurs_simulate(
        intervention=lemurs_intervention,
        patient=lemurs_patient,
    )
    states = result["states"]
    times = result["times"]

    return {
        "times": times,
        "tst": states[:, _LEMURS_TST],
        "pss": states[:, _LEMURS_PSS],
        "gad7": states[:, _LEMURS_GAD],
        "hrv": states[:, _LEMURS_HRV],
        "dac": states[:, _LEMURS_DAC],
    }


class LEMURSDisturbance(Disturbance):
    """Time-varying college stress disturbance for the mito simulator.

    Pre-computes the LEMURS semester trajectory at construction time, then
    interpolates the 5 bridge signals into the mito ODE at each timestep.

    The disturbance is active for the full sim duration (default 30 years).
    During the college window (first college_years of the active period),
    semester phase logic determines whether perturbations use in-semester
    values (interpolated from LEMURS curves) or break values (reduced).
    After the college window, perturbation decays exponentially.

    Five coupling channels:
      1. Sleep deficit (low TST) -> inflammation_level (additive)
      2. Perceived stress (PSS) -> metabolic_demand (additive)
      3. GAD-7 anxiety -> ROS (one-time impulse at onset)
      4. HRV decline -> genetic_vulnerability (multiplicative)
      5. DAC depletion -> repair capacity reduction (rapamycin scaling)
    Plus: GAD-7 above clinical threshold -> additional inflammation.

    Args:
        lemurs_patient: LEMURS simulator patient params (6D). Defaults to
            LEMURS DEFAULT_PATIENT.
        lemurs_intervention: LEMURS simulator intervention params (6D).
            Defaults to LEMURS DEFAULT_INTERVENTION (no support).
        college_start_age: Age when college begins (for semester counting).
        college_end_age: Age when college ends.
        start_year: When the disturbance activates in the mito simulation
            timeline (years from sim start). Default 0.0.
        duration: How long the disturbance is active. Default 30.0 (full sim).
        magnitude: Overall scaling factor on all LEMURS->mito effects (0-1).
        label: Human-readable label for this disturbance.
    """

    def __init__(
        self,
        lemurs_patient: dict | None = None,
        lemurs_intervention: dict | None = None,
        college_start_age: float = LEMURS_COLLEGE_START_AGE,
        college_end_age: float = LEMURS_COLLEGE_END_AGE,
        start_year: float = 0.0,
        duration: float = 30.0,
        magnitude: float = 1.0,
        label: str | None = None,
    ) -> None:
        name = label or "LEMURS"
        super().__init__(name, start_year, duration, magnitude)

        # Store LEMURS params for metadata/reproduction
        self.lemurs_patient = lemurs_patient or dict(LEMURS_DEFAULT_PATIENT)
        self.lemurs_intervention = lemurs_intervention or dict(LEMURS_DEFAULT_INTERVENTION)

        # College window duration in years
        self._college_start_age = college_start_age
        self._college_end_age = college_end_age
        self._college_years = college_end_age - college_start_age  # typically 4.0

        # Pre-compute the LEMURS semester trajectory
        curves = lemurs_trajectory(lemurs_patient, lemurs_intervention)

        # LEMURS times are in weeks (0 to 15); convert to year-fractions
        # for semester mapping within each semester cycle
        self._semester_times = curves["times"] / 52.0  # weeks -> year-fractions

        # Normalize bridge variables to [0, 1] for uniform coupling
        self._tst_norm = np.clip(
            (curves["tst"] - LEMURS_TST_MIN) / (LEMURS_TST_MAX - LEMURS_TST_MIN),
            0.0, 1.0,
        )
        self._pss_norm = np.clip(curves["pss"] / LEMURS_PSS_MAX, 0.0, 1.0)
        self._gad_norm = np.clip(curves["gad7"] / LEMURS_GAD_MAX, 0.0, 1.0)
        self._hrv_norm = np.clip(
            (curves["hrv"] - LEMURS_HRV_MIN) / (LEMURS_HRV_MAX - LEMURS_HRV_MIN),
            0.0, 1.0,
        )
        # DAC is already 0-1; just clip for safety
        self._dac = np.clip(curves["dac"], 0.0, 1.0)

    def _semester_phase(self, t: float) -> tuple[bool, float, int]:
        """Determine semester phase at mito sim time t.

        Args:
            t: Time in years from mito sim start.

        Returns:
            (in_semester, semester_week, semesters_completed)
            - in_semester: True if currently in a fall or spring semester
            - semester_week: Week position within the current semester (0-14)
            - semesters_completed: Total number of semesters completed so far
        """
        t_rel = t - self.start_year  # time since disturbance started
        if t_rel < 0:
            return (False, 0.0, 0)

        # Which year of college (0-indexed)?
        year_in_college = t_rel  # years since college started
        if year_in_college >= self._college_years:
            # Past college: all semesters completed
            total_semesters = int(self._college_years * LEMURS_SEMESTERS_PER_YEAR)
            return (False, 0.0, total_semesters)

        # Position within the current academic year
        year_frac = year_in_college % 1.0
        week_of_year = year_frac * 52.0

        # Academic calendar (approximate):
        # Fall semester:   weeks 0-14  (15 weeks)
        # Winter break:    weeks 15-21 (7 weeks)
        # Spring semester: weeks 22-36 (15 weeks)
        # Summer break:    weeks 37-51 (15 weeks)
        if week_of_year < 15.0:
            # Fall semester
            in_semester = True
            semester_week = week_of_year
        elif week_of_year < 22.0:
            # Winter break
            in_semester = False
            semester_week = 0.0
        elif week_of_year < 37.0:
            # Spring semester
            in_semester = True
            semester_week = week_of_year - 22.0
        else:
            # Summer break
            in_semester = False
            semester_week = 0.0

        # Count completed semesters
        completed_years = int(year_in_college)
        semesters_from_full_years = completed_years * 2
        # Add semesters from current partial year
        if week_of_year >= 37.0:
            semesters_this_year = 2  # both fall and spring done
        elif week_of_year >= 15.0:
            semesters_this_year = 1  # fall done
        else:
            semesters_this_year = 0  # still in fall
        semesters_completed = semesters_from_full_years + semesters_this_year

        return (in_semester, semester_week, semesters_completed)

    def _interp_semester(self, arr: np.ndarray, semester_week: float) -> float:
        """Interpolate a LEMURS curve at a given week within the semester.

        Args:
            arr: Normalized LEMURS signal array (106 points).
            semester_week: Position within semester (0-14).

        Returns:
            Interpolated value.
        """
        # Convert semester_week to year-fraction to match self._semester_times
        t_frac = semester_week / 52.0
        return float(np.interp(t_frac, self._semester_times, arr))

    def modify_state(
        self,
        state: npt.NDArray[np.float64],
        t: float,
    ) -> npt.NDArray[np.float64]:
        """Add initial GAD-7-driven ROS to the mito state vector.

        Anxiety-driven oxidative stress: GAD-7 elevation at college onset
        produces reactive oxygen species through HPA axis activation and
        catecholamine-mediated mitochondrial ETC stress.
        """
        state = state.copy()
        gad_val = self._gad_norm[0]  # initial GAD-7 value (normalized)
        # state[3] = ROS
        state[3] += LEMURS_GAD_ROS_COEFF * gad_val * self.magnitude
        return state

    def modify_params(
        self,
        intervention: dict[str, float],
        patient: dict[str, float],
        t: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Perturb mito patient/intervention params based on current LEMURS state.

        Five channels:
          1. Sleep deficit (low TST) -> inflammation_level (additive)
          2. Perceived stress (PSS) -> metabolic_demand (additive)
          3. HRV decline -> genetic_vulnerability (multiplicative)
          4. DAC depletion -> repair capacity reduction (rapamycin scaling)
          5. GAD-7 above clinical threshold -> additional inflammation
        """
        if not self.is_active(t):
            return intervention, patient

        patient = dict(patient)
        intervention = dict(intervention)

        in_semester, week, n_semesters = self._semester_phase(t)

        # Allostatic load accumulation: each completed semester ratchets up
        # the cumulative stress effect (3% per semester)
        allostatic = 1.0 + LEMURS_ALLOSTATIC_RATE * n_semesters

        # Post-college decay: after the college window, perturbation
        # decays exponentially (half-life ~1.4 years)
        t_rel = t - self.start_year
        if t_rel > self._college_years:
            decay = np.exp(-LEMURS_POST_COLLEGE_DECAY * (t_rel - self._college_years))
        else:
            decay = 1.0

        mag = self.magnitude * allostatic * decay

        if in_semester:
            tst_val = self._interp_semester(self._tst_norm, week)
            pss_val = self._interp_semester(self._pss_norm, week)
            gad_val = self._interp_semester(self._gad_norm, week)
            hrv_val = self._interp_semester(self._hrv_norm, week)
            dac_val = self._interp_semester(self._dac, week)
        else:
            # During breaks, use baseline (beginning of semester) values
            # with reduced effect (partial recovery during breaks)
            tst_val = self._tst_norm[0]
            pss_val = self._pss_norm[0]
            gad_val = self._gad_norm[0]
            hrv_val = self._hrv_norm[0]
            dac_val = self._dac[0]
            mag *= 0.3  # reduced effect during breaks

        # Channel 1: Sleep deficit -> inflammation
        # Low TST increases systemic inflammation via NF-kB activation
        # and elevated inflammatory cytokines (IL-6, TNF-alpha)
        sleep_deficit = 1.0 - tst_val
        patient["inflammation_level"] = min(
            patient.get("inflammation_level", 0.25) + LEMURS_TST_INFL_COEFF * sleep_deficit * mag,
            1.0,
        )

        # Channel 2: Perceived stress -> metabolic demand
        # Chronic stress elevates cortisol, increases basal metabolic rate,
        # and diverts energy to allostatic load maintenance
        patient["metabolic_demand"] = min(
            patient.get("metabolic_demand", 1.0) + LEMURS_PSS_DEMAND_COEFF * pss_val * mag,
            2.0,
        )

        # Channel 3: HRV decline -> vulnerability (multiplicative)
        # Low HRV reflects poor autonomic regulation, reduced vagal tone,
        # and impaired stress recovery — increasing cellular vulnerability
        hrv_deficit = 1.0 - hrv_val
        patient["genetic_vulnerability"] = patient.get("genetic_vulnerability", 1.0) * (
            1.0 + LEMURS_HRV_VULN_COEFF * hrv_deficit * mag
        )

        # Channel 4: DAC depletion -> repair capacity reduction
        # Directed attention depletion reflects cognitive exhaustion that
        # compromises adherence to health protocols (scaled as rapamycin
        # efficacy reduction)
        dac_deficit = 1.0 - dac_val
        repair_scale = 1.0 - LEMURS_DAC_REPAIR_COEFF * dac_deficit * mag
        intervention["rapamycin_dose"] = intervention.get("rapamycin_dose", 0.0) * max(repair_scale, 0.0)

        # Channel 5: GAD-7 above clinical threshold -> additional inflammation
        # Clinical anxiety (GAD-7 >= 10, normalized threshold = 10/21 ~ 0.476)
        # produces additional inflammatory burden via chronic HPA axis activation
        if gad_val > 10.0 / LEMURS_GAD_MAX:
            patient["inflammation_level"] = min(
                patient["inflammation_level"] + LEMURS_GAD_ROS_COEFF * gad_val * mag * 0.5,
                1.0,
            )

        return intervention, patient


# -- Full intervention profile for "with support" scenarios -------------------

_LEMURS_FULL_SUPPORT: dict[str, float] = {
    "nature_rx": 0.8,
    "exercise_rx": 0.8,
    "therapy_rx": 0.5,
    "sleep_hygiene": 0.8,
    "caffeine_reduction": 0.5,
    "academic_load": 0.3,
}


def lemurs_scenarios() -> list[LEMURSDisturbance]:
    """Build LEMURSDisturbance objects for all 8 archetypes x 2 intervention levels.

    Returns 16 disturbances: each student archetype without intervention
    (default coping) and with full intervention support. These can be passed
    individually to simulate_with_disturbances() for comparison.

    LEMURS STUDENT_ARCHETYPES is a list of dicts, each with "name",
    "description", "patient", and optionally "intervention" keys.
    """
    scenarios = []
    for archetype in LEMURS_STUDENT_ARCHETYPES:
        name = archetype["name"]
        # Without intervention (default coping)
        scenarios.append(LEMURSDisturbance(
            lemurs_patient=archetype["patient"],
            lemurs_intervention=None,
            label=f"lemurs_{name}_no_support",
        ))
        # With full intervention support
        scenarios.append(LEMURSDisturbance(
            lemurs_patient=archetype["patient"],
            lemurs_intervention=_LEMURS_FULL_SUPPORT,
            label=f"lemurs_{name}_full_support",
        ))
    return scenarios
