"""Age-dependent sleep trajectory model for mitochondrial aging.

Computes time-varying sleep quality based on epidemiological age curves,
patient sleep intervention level, alcohol intake, and optional LEMURS/grief
overrides for specific age windows. Returns 5 coupling channel modifiers
that feed into the ParameterResolver's Step 5.

Usage::

    from sleep_trajectory import SleepTrajectory

    st = SleepTrajectory(
        sleep_intervention=0.7,
        alcohol_trajectory=np.array([0.2, 0.15, 0.1]),
        time_points=np.array([0, 15, 30]),
        baseline_age=50.0,
        genetic_mods={'mitophagy_efficiency': 0.65},
    )
    effects = st.compute(t=10.0)
    # effects = {
    #   'sleep_quality': 0.78,
    #   'inflammation_delta': 0.018,
    #   'sleep_repair_factor': 0.89,
    #   'ros_boost': 0.009,
    #   'nad_drain': 0.004,
    #   'membrane_penalty': 0.007,
    # }
"""
from __future__ import annotations

import numpy as np
from constants import (
    SLEEP_AGE_ANCHORS, SLEEP_QUALITY_ANCHORS,
    SLEEP_INTERVENTION_RECOVERY,
    SLEEP_INFLAMMATION_COEFF, SLEEP_INFLAMMATION_AGE_GAIN,
    SLEEP_REPAIR_COEFF, SLEEP_ROS_COEFF,
    SLEEP_NAD_DRAIN_COEFF, SLEEP_MEMBRANE_COEFF,
    SLEEP_ATP_BOOST_COEFF,
    SLEEP_AGE_SENSITIVITY_RATE, SLEEP_AGE_SENSITIVITY_MAX,
    APOE4_SLEEP_AMPLIFICATION_ENABLED,
    ALCOHOL_SLEEP_DISRUPTION,
)


class SleepTrajectory:
    """Age-dependent sleep model with 5 mito coupling channels."""

    def __init__(
        self,
        sleep_intervention: float = 0.5,
        alcohol_trajectory: np.ndarray | None = None,
        time_points: np.ndarray | None = None,
        baseline_age: float = 70.0,
        genetic_mods: dict | None = None,
        sim_years: float = 30.0,
        lemurs_override: callable | None = None,
    ):
        self._sleep_int = np.clip(sleep_intervention, 0.0, 1.0)
        self._baseline_age = baseline_age
        self._sim_years = sim_years
        self._genetic_mods = genetic_mods or {}
        self._lemurs_override = lemurs_override

        # Default flat alcohol trajectory if not provided
        if alcohol_trajectory is None or time_points is None:
            self._alcohol_t = np.zeros(2)
            self._time_pts = np.array([0.0, sim_years])
        else:
            self._alcohol_t = alcohol_trajectory
            self._time_pts = time_points

    def _age_baseline_quality(self, age: float) -> float:
        """Piecewise-linear sleep quality from epidemiological anchors."""
        return float(np.interp(
            np.clip(age, SLEEP_AGE_ANCHORS[0], SLEEP_AGE_ANCHORS[-1]),
            SLEEP_AGE_ANCHORS,
            SLEEP_QUALITY_ANCHORS,
        ))

    def _age_sensitivity(self, age: float) -> float:
        """Age-dependent vulnerability multiplier (1.0 at 30, up to 1.5)."""
        extra = max(age - 30.0, 0.0) * SLEEP_AGE_SENSITIVITY_RATE
        return min(1.0 + extra, SLEEP_AGE_SENSITIVITY_MAX)

    def compute(self, t: float) -> dict:
        """Compute sleep quality and 5 coupling effects at simulation time t.

        Args:
            t: Simulation time (years from start, 0 to sim_years).

        Returns:
            Dict with keys: sleep_quality, inflammation_delta,
            sleep_repair_factor, ros_boost, nad_drain, membrane_penalty.
        """
        age = self._baseline_age + t

        # 1. Epidemiological baseline for this age
        baseline_q = self._age_baseline_quality(age)

        # 2. Sleep intervention recovers some of the age-related decline
        #    from optimal (0.95). Recovery fraction is SLEEP_INTERVENTION_RECOVERY.
        #    Intervention is centered: 0.5 = neutral (no net effect).
        age_decline = max(SLEEP_QUALITY_ANCHORS[0] - baseline_q, 0.0)
        intervention_centered = self._sleep_int - 0.5  # -0.5 to +0.5
        recovery = age_decline * SLEEP_INTERVENTION_RECOVERY * intervention_centered * 2.0
        quality = baseline_q + recovery

        # 3. Alcohol degrades sleep quality
        alcohol_t = float(np.interp(t, self._time_pts, self._alcohol_t))
        quality = max(0.0, quality - alcohol_t * ALCOHOL_SLEEP_DISRUPTION)

        # 4. Sleep deficit relative to age baseline (0 = age-typical, positive = worse,
        #    negative = better than age-typical). This ensures the resolver is neutral
        #    when sleep_intervention=0.5 at any age — age-typical sleep is already
        #    modeled by the raw ODE's natural aging trajectory.
        #    Finding 4 correction (2026-02-22): deficit from age-baseline, not from 1.0.
        deficit = max(baseline_q - quality, 0.0)  # penalty only for worse-than-baseline
        benefit = max(quality - baseline_q, 0.0)   # bonus for better-than-baseline

        # 4b. LEMURS override for inflammation channel only
        # When LEMURS bridge provides empirical TST-derived sleep quality
        # for ages 18-22, use it for the inflammation channel. Other channels
        # still use the epidemiological baseline quality.
        quality_for_inflammation = quality
        if self._lemurs_override is not None:
            lemurs_q = self._lemurs_override(t)
            if lemurs_q is not None:
                quality_for_inflammation = lemurs_q
        deficit_infl = max(baseline_q - quality_for_inflammation, 0.0)
        benefit_infl = max(quality_for_inflammation - baseline_q, 0.0)

        # 5. Age sensitivity multiplier
        sensitivity = self._age_sensitivity(age)

        # 6. Compute 5 coupling channels
        mitophagy_eff = self._genetic_mods.get('mitophagy_efficiency', 1.0)
        # APOE4 amplification factor (Finding 3 correction)
        # WT (mitophagy_eff=1.0) → 1.0x, APOE4-het (0.65) → 1.35x, hom (0.45) → 1.55x
        apoe4_amp = 2.0 - mitophagy_eff if APOE4_SLEEP_AMPLIFICATION_ENABLED else 1.0

        # Channel 1: Inflammation (age-modulated, uses LEMURS quality if available)
        age_infl_coeff = SLEEP_INFLAMMATION_COEFF + max(age - 40, 0) * SLEEP_INFLAMMATION_AGE_GAIN
        inflammation_delta = (deficit_infl - 1.0 * benefit_infl) * age_infl_coeff * sensitivity * apoe4_amp

        # Channel 2: Repair factor (genotype-gated)
        sleep_repair_factor = 1.0 - SLEEP_REPAIR_COEFF * deficit * apoe4_amp + 0.5 * benefit
        sleep_repair_factor = np.clip(sleep_repair_factor, 0.0, 2.0)

        # Channel 3: ROS boost
        ros_boost = deficit * SLEEP_ROS_COEFF * sensitivity * apoe4_amp

        # Channel 4: NAD drain
        nad_drain = deficit * SLEEP_NAD_DRAIN_COEFF * sensitivity

        # Channel 5: Membrane penalty
        membrane_penalty = deficit * SLEEP_MEMBRANE_COEFF * sensitivity

        # Channel 6: ATP boost (NEW)
        atp_boost = benefit * SLEEP_ATP_BOOST_COEFF * sensitivity

        return {
            'sleep_quality': float(quality),
            'inflammation_delta': float(inflammation_delta),
            'sleep_repair_factor': float(sleep_repair_factor),
            'ros_boost': float(ros_boost),
            'nad_drain': float(nad_drain),
            'membrane_penalty': float(membrane_penalty),
            'atp_boost': float(atp_boost),
        }
