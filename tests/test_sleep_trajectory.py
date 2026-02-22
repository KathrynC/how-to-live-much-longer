"""Tests for the age-dependent sleep trajectory model."""
from __future__ import annotations

import numpy as np
import pytest

from sleep_trajectory import SleepTrajectory
from constants import (
    SLEEP_AGE_ANCHORS, SLEEP_QUALITY_ANCHORS,
    SLEEP_INTERVENTION_RECOVERY,
    SLEEP_AGE_SENSITIVITY_MAX,
)


class TestAgeBaselineQuality:
    """Verify epidemiological anchor points and interpolation."""

    def test_age_baseline_quality(self):
        """Anchors match at 20, 40, 60, 80."""
        st = SleepTrajectory(sleep_intervention=0.0, baseline_age=0.0)
        for age, expected_q in zip(SLEEP_AGE_ANCHORS, SLEEP_QUALITY_ANCHORS):
            result = st._age_baseline_quality(age)
            assert result == pytest.approx(expected_q, abs=1e-9), (
                f"At age {age}: expected {expected_q}, got {result}"
            )

    def test_age_baseline_interpolation(self):
        """Smooth interpolation at intermediate ages (e.g., age 30)."""
        st = SleepTrajectory(sleep_intervention=0.0, baseline_age=0.0)
        # Age 30 is midpoint between anchors at 20 (0.95) and 40 (0.88)
        q30 = st._age_baseline_quality(30.0)
        expected = 0.5 * (0.95 + 0.88)  # 0.915
        assert q30 == pytest.approx(expected, abs=1e-9)
        # Must be between the two anchor values
        assert 0.88 < q30 < 0.95

    def test_age_clamped_below(self):
        """Ages below 20 clamp to the age-20 anchor."""
        st = SleepTrajectory(sleep_intervention=0.0, baseline_age=0.0)
        q10 = st._age_baseline_quality(10.0)
        assert q10 == pytest.approx(SLEEP_QUALITY_ANCHORS[0], abs=1e-9)

    def test_age_clamped_above(self):
        """Ages above 80 clamp to the age-80 anchor."""
        st = SleepTrajectory(sleep_intervention=0.0, baseline_age=0.0)
        q100 = st._age_baseline_quality(100.0)
        assert q100 == pytest.approx(SLEEP_QUALITY_ANCHORS[-1], abs=1e-9)


class TestSleepIntervention:
    """Verify intervention recovery mechanics."""

    def test_sleep_intervention_recovery(self):
        """Higher intervention -> better quality at age 70."""
        low = SleepTrajectory(sleep_intervention=0.0, baseline_age=70.0)
        mid = SleepTrajectory(sleep_intervention=0.5, baseline_age=70.0)
        high = SleepTrajectory(sleep_intervention=1.0, baseline_age=70.0)

        q_low = low.compute(t=0.0)['sleep_quality']
        q_mid = mid.compute(t=0.0)['sleep_quality']
        q_high = high.compute(t=0.0)['sleep_quality']

        assert q_high > q_mid > q_low
        # At intervention=0.0, quality equals baseline for age 70
        baseline_q_70 = low._age_baseline_quality(70.0)
        assert q_low == pytest.approx(baseline_q_70, abs=1e-9)

    def test_intervention_clipped(self):
        """Intervention values outside [0, 1] are clipped."""
        st_neg = SleepTrajectory(sleep_intervention=-0.5, baseline_age=70.0)
        st_zero = SleepTrajectory(sleep_intervention=0.0, baseline_age=70.0)
        st_over = SleepTrajectory(sleep_intervention=2.0, baseline_age=70.0)
        st_one = SleepTrajectory(sleep_intervention=1.0, baseline_age=70.0)

        assert st_neg.compute(0.0)['sleep_quality'] == pytest.approx(
            st_zero.compute(0.0)['sleep_quality'], abs=1e-9)
        assert st_over.compute(0.0)['sleep_quality'] == pytest.approx(
            st_one.compute(0.0)['sleep_quality'], abs=1e-9)


class TestAlcoholEffect:
    """Verify alcohol trajectory degrades sleep quality."""

    def test_alcohol_degrades_quality(self):
        """Alcohol trajectory reduces effective quality."""
        # No alcohol
        st_sober = SleepTrajectory(
            sleep_intervention=0.5,
            baseline_age=50.0,
        )
        # Constant alcohol = 0.5
        st_drinker = SleepTrajectory(
            sleep_intervention=0.5,
            alcohol_trajectory=np.array([0.5, 0.5]),
            time_points=np.array([0.0, 30.0]),
            baseline_age=50.0,
        )
        q_sober = st_sober.compute(t=10.0)['sleep_quality']
        q_drinker = st_drinker.compute(t=10.0)['sleep_quality']
        assert q_drinker < q_sober

    def test_alcohol_time_varying(self):
        """Alcohol effect varies with interpolated trajectory."""
        st = SleepTrajectory(
            sleep_intervention=0.5,
            alcohol_trajectory=np.array([0.0, 1.0]),
            time_points=np.array([0.0, 30.0]),
            baseline_age=50.0,
        )
        q_early = st.compute(t=0.0)['sleep_quality']
        q_late = st.compute(t=30.0)['sleep_quality']
        # More alcohol later -> worse sleep later
        assert q_late < q_early


class TestDeficitChannels:
    """Verify coupling channel monotonicity with sleep deficit."""

    def test_deficit_channels_monotonic(self):
        """Worse sleep -> higher inflammation, ROS, NAD drain, membrane
        penalty; lower repair factor."""
        good_sleep = SleepTrajectory(sleep_intervention=1.0, baseline_age=50.0)
        bad_sleep = SleepTrajectory(sleep_intervention=0.0, baseline_age=50.0)

        good = good_sleep.compute(t=0.0)
        bad = bad_sleep.compute(t=0.0)

        assert bad['inflammation_delta'] > good['inflammation_delta']
        assert bad['ros_boost'] > good['ros_boost']
        assert bad['nad_drain'] > good['nad_drain']
        assert bad['membrane_penalty'] > good['membrane_penalty']
        assert bad['sleep_repair_factor'] < good['sleep_repair_factor']


class TestAgeSensitivity:
    """Verify age-dependent sensitivity multiplier."""

    def test_age_sensitivity_young(self):
        """Age 25 (t=0, baseline_age=25) -> sensitivity ~1.0."""
        st = SleepTrajectory(baseline_age=25.0)
        sensitivity = st._age_sensitivity(25.0)
        assert sensitivity == pytest.approx(1.0, abs=1e-9)

    def test_age_sensitivity_at_30(self):
        """Age 30 is the threshold: sensitivity exactly 1.0."""
        st = SleepTrajectory(baseline_age=30.0)
        sensitivity = st._age_sensitivity(30.0)
        assert sensitivity == pytest.approx(1.0, abs=1e-9)

    def test_age_sensitivity_old(self):
        """Age 80 -> sensitivity = 1.5 (capped)."""
        st = SleepTrajectory(baseline_age=80.0)
        sensitivity = st._age_sensitivity(80.0)
        assert sensitivity == pytest.approx(SLEEP_AGE_SENSITIVITY_MAX, abs=1e-9)

    def test_age_sensitivity_very_old_capped(self):
        """Age 120 -> sensitivity still capped at 1.5."""
        st = SleepTrajectory(baseline_age=120.0)
        sensitivity = st._age_sensitivity(120.0)
        assert sensitivity == pytest.approx(SLEEP_AGE_SENSITIVITY_MAX, abs=1e-9)


class TestGenotype:
    """Verify genotype-gated repair factor."""

    def test_genotype_gating(self):
        """APOE4 (mitophagy_eff=0.65) -> worse repair factor than WT (1.0)."""
        wt = SleepTrajectory(
            sleep_intervention=0.3,
            baseline_age=70.0,
            genetic_mods={'mitophagy_efficiency': 1.0},
        )
        apoe4 = SleepTrajectory(
            sleep_intervention=0.3,
            baseline_age=70.0,
            genetic_mods={'mitophagy_efficiency': 0.65},
        )
        wt_repair = wt.compute(t=0.0)['sleep_repair_factor']
        apoe4_repair = apoe4.compute(t=0.0)['sleep_repair_factor']
        assert apoe4_repair < wt_repair

    def test_genotype_default_is_neutral(self):
        """No genetic_mods -> mitophagy_efficiency defaults to 1.0."""
        st_default = SleepTrajectory(sleep_intervention=0.3, baseline_age=70.0)
        st_explicit = SleepTrajectory(
            sleep_intervention=0.3,
            baseline_age=70.0,
            genetic_mods={'mitophagy_efficiency': 1.0},
        )
        r_default = st_default.compute(t=0.0)['sleep_repair_factor']
        r_explicit = st_explicit.compute(t=0.0)['sleep_repair_factor']
        assert r_default == pytest.approx(r_explicit, abs=1e-9)


class TestBoundaryConditions:
    """Verify boundary and extreme conditions."""

    def test_perfect_sleep_zero_effects(self):
        """With intervention=1.0 and age=20, deficit~0.05, effects near-zero."""
        st = SleepTrajectory(
            sleep_intervention=1.0,
            baseline_age=20.0,
        )
        effects = st.compute(t=0.0)
        # At age 20, baseline quality = 0.95.
        # Recovery = (0.95 - 0.95) * 0.6 * 1.0 = 0.0 -> quality = 0.95
        # Deficit = 0.05, sensitivity = 1.0 (age 20 < 30)
        assert effects['sleep_quality'] == pytest.approx(0.95, abs=1e-9)
        assert effects['inflammation_delta'] < 0.01
        assert effects['ros_boost'] < 0.01
        assert effects['nad_drain'] < 0.01
        assert effects['membrane_penalty'] < 0.01
        assert effects['sleep_repair_factor'] > 0.97

    def test_no_sleep_max_effects(self):
        """With intervention=0.0 and age=80, maximum penalties."""
        st = SleepTrajectory(
            sleep_intervention=0.0,
            baseline_age=80.0,
        )
        effects = st.compute(t=0.0)
        # At age 80, baseline quality = 0.60 (worst anchor)
        # No recovery. deficit = 0.40. sensitivity = 1.5
        assert effects['sleep_quality'] == pytest.approx(0.60, abs=1e-9)
        assert effects['inflammation_delta'] > 0.0
        assert effects['ros_boost'] > 0.0
        assert effects['nad_drain'] > 0.0
        assert effects['membrane_penalty'] > 0.0
        assert effects['sleep_repair_factor'] < 1.0

    def test_output_keys(self):
        """Verify all 6 expected keys present in compute() return."""
        st = SleepTrajectory()
        effects = st.compute(t=0.0)
        expected_keys = {
            'sleep_quality',
            'inflammation_delta',
            'sleep_repair_factor',
            'ros_boost',
            'nad_drain',
            'membrane_penalty',
        }
        assert set(effects.keys()) == expected_keys

    def test_clamp_bounds(self):
        """sleep_repair_factor stays in [0, 1] even with extreme inputs."""
        # Very low mitophagy efficiency -> repair factor could go negative
        st_extreme = SleepTrajectory(
            sleep_intervention=0.0,
            baseline_age=80.0,
            genetic_mods={'mitophagy_efficiency': 0.1},
        )
        effects = st_extreme.compute(t=0.0)
        assert 0.0 <= effects['sleep_repair_factor'] <= 1.0

        # Very high mitophagy efficiency -> should not exceed 1.0
        st_high = SleepTrajectory(
            sleep_intervention=1.0,
            baseline_age=20.0,
            genetic_mods={'mitophagy_efficiency': 10.0},
        )
        effects_high = st_high.compute(t=0.0)
        assert 0.0 <= effects_high['sleep_repair_factor'] <= 1.0

    def test_quality_non_negative_under_heavy_alcohol(self):
        """Sleep quality stays non-negative even with heavy alcohol."""
        st = SleepTrajectory(
            sleep_intervention=0.0,
            alcohol_trajectory=np.array([2.0, 2.0]),
            time_points=np.array([0.0, 30.0]),
            baseline_age=80.0,
        )
        effects = st.compute(t=0.0)
        assert effects['sleep_quality'] >= 0.0
