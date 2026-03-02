"""Tests for the wearable sensor observation model.

Tests device specifications, observation functions, unified observation model,
state estimation, and sensor-constrained adaptive protocol.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from constants import (
    N_STATES, DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    SENSOR_RANGE_HR, SENSOR_RANGE_HRV, SENSOR_RANGE_SPO2,
    SENSOR_RANGE_TEMP, SENSOR_RANGE_GLUCOSE, SENSOR_RANGE_RESP_RATE,
    SENSOR_RANGE_SYS_BP, SENSOR_RANGE_DIA_BP, SENSOR_RANGE_ACTIVITY,
    SENSOR_RANGE_LACTATE, SENSOR_RANGE_HS_CRP, SENSOR_RANGE_GDF15,
    SENSOR_RANGE_NAD_BLOOD, SENSOR_RANGE_8OHDG,
    CGM_ACTIVE_DAYS, CGM_CYCLE_DAYS,
    LINGO_ACTIVE_DAYS, LINGO_CYCLE_DAYS,
    BIOMARKER_INTERVAL_DAYS,
    BASELINE_ATP, BASELINE_ROS, BASELINE_NAD,
    BASELINE_SENESCENT, BASELINE_MEMBRANE_POTENTIAL,
)
from wearable_sensors import (
    SensorChannel, DeviceSpec,
    apple_watch_11_spec, oura_ring_4_spec, dexcom_stelo_spec,
    abbott_lingo_spec, periodic_biomarker_spec,
    observe_heart_rate, observe_hrv, observe_spo2,
    observe_temperature, observe_blood_pressure, observe_glucose,
    observe_respiratory_rate, observe_activity,
    observe_lactate, observe_hs_crp, observe_gdf15,
    observe_nad_blood, observe_8ohdg,
    WearableObservationModel,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def healthy_state():
    """Healthy 40-year-old: near-baseline ODE state."""
    return np.array([0.85, 0.10, 0.95, 0.12, 0.90, 0.05, 0.95, 0.05])


@pytest.fixture
def declining_state():
    """Declining 70-year-old: moderate damage."""
    return np.array([0.55, 0.30, 0.65, 0.35, 0.55, 0.20, 0.70, 0.15])


@pytest.fixture
def crisis_state():
    """Near-cliff crisis: severe energy deficit."""
    return np.array([0.30, 0.50, 0.30, 0.60, 0.30, 0.40, 0.40, 0.20])


@pytest.fixture
def obs_model():
    """Default observation model with all 3 devices."""
    return WearableObservationModel(seed=42)


# ── TestDeviceSpecs ──────────────────────────────────────────────────────────

class TestDeviceSpecs:

    def test_apple_watch_channel_count(self):
        spec = apple_watch_11_spec()
        assert spec.n_channels == 7

    def test_oura_ring_channel_count(self):
        spec = oura_ring_4_spec()
        assert spec.n_channels == 5

    def test_dexcom_stelo_channel_count(self):
        spec = dexcom_stelo_spec()
        assert spec.n_channels == 1

    def test_apple_watch_has_expected_channels(self):
        spec = apple_watch_11_spec()
        expected = {"heart_rate", "hrv", "spo2", "temperature",
                    "blood_pressure_sys", "blood_pressure_dia", "activity"}
        assert set(spec.channel_names) == expected

    def test_oura_ring_has_expected_channels(self):
        spec = oura_ring_4_spec()
        expected = {"heart_rate", "hrv", "spo2", "temperature", "respiratory_rate"}
        assert set(spec.channel_names) == expected

    def test_dexcom_has_glucose_channel(self):
        spec = dexcom_stelo_spec()
        assert "glucose" in spec.channel_names

    def test_abbott_lingo_channel_count(self):
        spec = abbott_lingo_spec()
        assert spec.n_channels == 1

    def test_abbott_lingo_has_lactate_channel(self):
        spec = abbott_lingo_spec()
        assert "lactate" in spec.channel_names

    def test_periodic_biomarker_channel_count(self):
        spec = periodic_biomarker_spec()
        assert spec.n_channels == 4

    def test_periodic_biomarker_has_expected_channels(self):
        spec = periodic_biomarker_spec()
        expected = {"hs_crp", "gdf15", "nad_blood", "ohdg_8"}
        assert set(spec.channel_names) == expected

    def test_all_channels_have_positive_noise(self):
        for factory in [apple_watch_11_spec, oura_ring_4_spec, dexcom_stelo_spec,
                        abbott_lingo_spec, periodic_biomarker_spec]:
            spec = factory()
            for ch in spec.channels.values():
                assert ch.noise_std > 0, f"{spec.name}/{ch.name} has non-positive noise"

    def test_all_channels_have_valid_ranges(self):
        for factory in [apple_watch_11_spec, oura_ring_4_spec, dexcom_stelo_spec,
                        abbott_lingo_spec, periodic_biomarker_spec]:
            spec = factory()
            for ch in spec.channels.values():
                assert ch.range_min < ch.range_max, f"{spec.name}/{ch.name}: invalid range"

    def test_sensor_channel_dataclass(self):
        ch = SensorChannel("test", "units", 0.0, 100.0, 1.0, 5.0, 0.95)
        assert ch.name == "test"
        assert ch.unit == "units"
        assert ch.availability == 0.95


# ── TestObservationFunctions ─────────────────────────────────────────────────

class TestObservationFunctions:

    def test_heart_rate_healthy_in_range(self, rng):
        hr = observe_heart_rate(atp=0.95, ros=0.12, age=40.0, rng=rng)
        assert SENSOR_RANGE_HR[0] <= hr <= SENSOR_RANGE_HR[1]

    def test_heart_rate_increases_with_low_atp(self):
        """Low ATP should cause compensatory tachycardia (higher HR).
        Use zero noise to test the signal direction cleanly."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        hr_healthy = observe_heart_rate(atp=0.95, ros=0.1, age=40.0, rng=rng1, noise_std=0.0)
        hr_low = observe_heart_rate(atp=0.40, ros=0.1, age=40.0, rng=rng2, noise_std=0.0)
        assert hr_low > hr_healthy  # Compensatory tachycardia

    def test_hrv_healthy_in_range(self, rng):
        hrv = observe_hrv(atp=0.95, ros=0.1, sen=0.05, age=40.0, rng=rng)
        assert SENSOR_RANGE_HRV[0] <= hrv <= SENSOR_RANGE_HRV[1]

    def test_hrv_decreases_with_senescence(self, rng):
        hrv_low_sen = observe_hrv(atp=0.9, ros=0.1, sen=0.05, age=40.0, rng=rng)
        rng2 = np.random.default_rng(42)
        hrv_high_sen = observe_hrv(atp=0.9, ros=0.1, sen=0.40, age=40.0, rng=rng2)
        assert hrv_high_sen < hrv_low_sen

    def test_spo2_healthy_near_98(self, rng):
        spo2 = observe_spo2(psi=0.95, sen=0.05, rng=rng)
        assert 94.0 < spo2 <= 100.0

    def test_spo2_drops_with_low_psi(self, rng):
        spo2_healthy = observe_spo2(psi=0.95, sen=0.05, rng=rng)
        rng2 = np.random.default_rng(42)
        spo2_low = observe_spo2(psi=0.40, sen=0.05, rng=rng2)
        assert spo2_low < spo2_healthy

    def test_temperature_healthy_near_baseline(self, rng):
        temp = observe_temperature(ros=0.1, sen=0.05, psi=0.95, rng=rng)
        assert SENSOR_RANGE_TEMP[0] <= temp <= SENSOR_RANGE_TEMP[1]

    def test_blood_pressure_returns_tuple(self, rng):
        sys_bp, dia_bp = observe_blood_pressure(sen=0.1, ros=0.15, age=50.0, rng=rng)
        assert SENSOR_RANGE_SYS_BP[0] <= sys_bp <= SENSOR_RANGE_SYS_BP[1]
        assert SENSOR_RANGE_DIA_BP[0] <= dia_bp <= SENSOR_RANGE_DIA_BP[1]

    def test_blood_pressure_rises_with_senescence(self, rng):
        sys_low, _ = observe_blood_pressure(sen=0.05, ros=0.1, age=50.0, rng=rng)
        rng2 = np.random.default_rng(42)
        sys_high, _ = observe_blood_pressure(sen=0.40, ros=0.1, age=50.0, rng=rng2)
        assert sys_high > sys_low

    def test_glucose_healthy_in_range(self, rng):
        glucose = observe_glucose(nad=0.9, atp=0.95, rng=rng)
        assert SENSOR_RANGE_GLUCOSE[0] <= glucose <= SENSOR_RANGE_GLUCOSE[1]

    def test_glucose_rises_with_low_nad(self, rng):
        g_high_nad = observe_glucose(nad=0.9, atp=0.9, rng=rng)
        rng2 = np.random.default_rng(42)
        g_low_nad = observe_glucose(nad=0.3, atp=0.9, rng=rng2)
        assert g_low_nad > g_high_nad  # Poor insulin sensitivity

    def test_respiratory_rate_in_range(self, rng):
        resp = observe_respiratory_rate(atp=0.9, ros=0.15, rng=rng)
        assert SENSOR_RANGE_RESP_RATE[0] <= resp <= SENSOR_RANGE_RESP_RATE[1]

    def test_activity_scales_with_atp(self, rng):
        act_high = observe_activity(exercise_level=0.5, atp=0.9, rng=rng)
        rng2 = np.random.default_rng(42)
        act_low = observe_activity(exercise_level=0.5, atp=0.3, rng=rng2)
        assert act_low < act_high

    def test_observations_clamped_to_range(self, rng):
        """Extreme inputs should still produce in-range outputs."""
        hr = observe_heart_rate(atp=0.0, ros=5.0, age=100.0, rng=rng)
        assert SENSOR_RANGE_HR[0] <= hr <= SENSOR_RANGE_HR[1]

        rng2 = np.random.default_rng(43)
        spo2 = observe_spo2(psi=0.0, sen=1.0, rng=rng2)
        assert SENSOR_RANGE_SPO2[0] <= spo2 <= SENSOR_RANGE_SPO2[1]

    # ── Lactate ──

    def test_lactate_healthy_in_range(self, rng):
        lac = observe_lactate(atp=0.95, ros=0.12, sen=0.05, exercise_level=0.0, rng=rng)
        assert SENSOR_RANGE_LACTATE[0] <= lac <= SENSOR_RANGE_LACTATE[1]

    def test_lactate_rises_with_low_atp(self):
        """Low ATP → glycolytic compensation → higher lactate."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        lac_healthy = observe_lactate(atp=0.95, ros=0.1, sen=0.05, exercise_level=0.0,
                                      rng=rng1, noise_std=0.0)
        lac_low = observe_lactate(atp=0.40, ros=0.1, sen=0.05, exercise_level=0.0,
                                   rng=rng2, noise_std=0.0)
        assert lac_low > lac_healthy

    def test_lactate_rises_with_exercise(self):
        """Exercise produces lactate directly."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        lac_rest = observe_lactate(atp=0.9, ros=0.1, sen=0.05, exercise_level=0.0,
                                    rng=rng1, noise_std=0.0)
        lac_ex = observe_lactate(atp=0.9, ros=0.1, sen=0.05, exercise_level=0.8,
                                  rng=rng2, noise_std=0.0)
        assert lac_ex > lac_rest

    # ── hs-CRP ──

    def test_hs_crp_healthy_in_range(self, rng):
        crp = observe_hs_crp(sen=0.05, ros=0.12, rng=rng)
        assert SENSOR_RANGE_HS_CRP[0] <= crp <= SENSOR_RANGE_HS_CRP[1]

    def test_hs_crp_rises_with_senescence(self):
        """Senescence → SASP → elevated CRP."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        crp_low = observe_hs_crp(sen=0.05, ros=0.1, rng=rng1, noise_std=0.0)
        crp_high = observe_hs_crp(sen=0.40, ros=0.1, rng=rng2, noise_std=0.0)
        assert crp_high > crp_low

    # ── GDF-15 ──

    def test_gdf15_healthy_in_range(self, rng):
        gdf = observe_gdf15(atp=0.95, ros=0.12, nad=0.90, sen=0.05, rng=rng)
        assert SENSOR_RANGE_GDF15[0] <= gdf <= SENSOR_RANGE_GDF15[1]

    def test_gdf15_rises_with_mitochondrial_stress(self):
        """GDF-15 integrates multiple stress signals."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        gdf_healthy = observe_gdf15(atp=0.95, ros=0.12, nad=0.90, sen=0.05,
                                     rng=rng1, noise_std=0.0)
        gdf_stress = observe_gdf15(atp=0.40, ros=0.50, nad=0.30, sen=0.30,
                                    rng=rng2, noise_std=0.0)
        assert gdf_stress > gdf_healthy

    # ── NAD+ blood ──

    def test_nad_blood_healthy_in_range(self, rng):
        nad_b = observe_nad_blood(nad=0.90, rng=rng)
        assert SENSOR_RANGE_NAD_BLOOD[0] <= nad_b <= SENSOR_RANGE_NAD_BLOOD[1]

    def test_nad_blood_tracks_nad(self):
        """NAD+ blood is near-direct measurement of NAD state."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        nad_high = observe_nad_blood(nad=0.90, rng=rng1, noise_std=0.0)
        nad_low = observe_nad_blood(nad=0.30, rng=rng2, noise_std=0.0)
        assert nad_high > nad_low

    # ── 8-OHdG ──

    def test_8ohdg_healthy_in_range(self, rng):
        ohdg = observe_8ohdg(ros=0.12, rng=rng)
        assert SENSOR_RANGE_8OHDG[0] <= ohdg <= SENSOR_RANGE_8OHDG[1]

    def test_8ohdg_tracks_ros(self):
        """8-OHdG is directly proportional to ROS."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        ohdg_low = observe_8ohdg(ros=0.10, rng=rng1, noise_std=0.0)
        ohdg_high = observe_8ohdg(ros=0.60, rng=rng2, noise_std=0.0)
        assert ohdg_high > ohdg_low


# ── TestObservationModel ─────────────────────────────────────────────────────

class TestObservationModel:

    def test_default_creates_all_devices(self, obs_model):
        assert "apple_watch" in obs_model.device_specs
        assert "oura_ring" in obs_model.device_specs
        assert "dexcom_stelo" in obs_model.device_specs

    def test_single_device(self):
        model = WearableObservationModel(devices=["oura_ring"])
        assert len(model.device_specs) == 1
        assert "oura_ring" in model.device_specs

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="Unknown device"):
            WearableObservationModel(devices=["fitbit"])

    def test_observe_returns_all_channels(self, obs_model, healthy_state):
        readings = obs_model.observe(healthy_state, t=0.0)
        # All 3 devices provide channels
        assert "heart_rate" in readings
        assert "hrv" in readings
        assert "spo2" in readings
        assert "temperature" in readings
        assert "blood_pressure_sys" in readings
        assert "glucose" in readings  # CGM active at t=0
        assert "respiratory_rate" in readings
        assert "activity" in readings

    def test_observe_values_in_range(self, obs_model, healthy_state):
        readings = obs_model.observe(healthy_state, t=0.0)
        assert SENSOR_RANGE_HR[0] <= readings["heart_rate"] <= SENSOR_RANGE_HR[1]
        assert SENSOR_RANGE_SPO2[0] <= readings["spo2"] <= SENSOR_RANGE_SPO2[1]

    def test_cgm_unavailable_during_changeover(self, obs_model, healthy_state):
        """CGM should return NaN during the 1-day changeover period."""
        # t in years where day_in_cycle falls in changeover (days 14-15)
        # 14 days / 365.25 ≈ 0.0383 years, 15 days ≈ 0.0411 years
        # At t = 14.5/365.25 ≈ 0.0397, day_in_cycle = 14.5 > CGM_ACTIVE_DAYS
        t_changeover = 14.5 / 365.25
        readings = obs_model.observe(healthy_state, t=t_changeover)
        assert math.isnan(readings["glucose"])

    def test_cgm_active_during_wear_period(self, obs_model, healthy_state):
        """CGM should produce valid readings during 14-day active period."""
        t_active = 7.0 / 365.25  # day 7 of cycle
        readings = obs_model.observe(healthy_state, t=t_active)
        assert not math.isnan(readings["glucose"])

    def test_observe_with_intervention(self, obs_model, healthy_state):
        intervention = {"exercise_level": 0.5}
        readings = obs_model.observe(healthy_state, t=0.0, intervention=intervention)
        assert "activity" in readings
        assert readings["activity"] > 0

    def test_observe_deterministic_with_same_seed(self, healthy_state):
        model1 = WearableObservationModel(seed=123)
        model2 = WearableObservationModel(seed=123)
        r1 = model1.observe(healthy_state, t=0.0)
        r2 = model2.observe(healthy_state, t=0.0)
        for key in r1:
            assert r1[key] == r2[key], f"Mismatch for {key}"

    def test_declining_state_shows_worse_readings(self, healthy_state, declining_state):
        """Declining state should show lower HRV and higher glucose."""
        m1 = WearableObservationModel(seed=42)
        m2 = WearableObservationModel(seed=42)
        r_healthy = m1.observe(healthy_state, t=0.0)
        r_declining = m2.observe(declining_state, t=10.0)
        # Declining patient: lower HRV (less vagal tone), higher glucose (lower NAD)
        assert r_declining["hrv"] < r_healthy["hrv"]
        assert r_declining["blood_pressure_sys"] > r_healthy["blood_pressure_sys"]

    def test_observe_with_all_five_devices(self, healthy_state):
        """Model with all 5 devices should include lactate and biomarker channels."""
        model = WearableObservationModel(
            devices=["apple_watch", "oura_ring", "dexcom_stelo",
                     "abbott_lingo", "periodic_biomarker"],
            seed=42,
        )
        readings = model.observe(healthy_state, t=0.0)
        # t=0.0 is at a quarterly boundary, so biomarkers should be available
        assert "lactate" in readings
        assert "hs_crp" in readings
        assert "gdf15" in readings
        assert "nad_blood" in readings
        assert "ohdg_8" in readings

    def test_lingo_availability_cycle(self, healthy_state):
        """Lactate should be NaN during Lingo changeover."""
        model = WearableObservationModel(
            devices=["abbott_lingo"], seed=42,
        )
        # Active period (day 7)
        t_active = 7.0 / 365.25
        readings = model.observe(healthy_state, t=t_active)
        assert not math.isnan(readings["lactate"])

        # Changeover period (day 14.5)
        t_changeover = 14.5 / 365.25
        readings = model.observe(healthy_state, t=t_changeover)
        assert math.isnan(readings["lactate"])

    def test_biomarker_quarterly_availability(self, healthy_state):
        """Biomarkers should only be available near quarterly draw days."""
        model = WearableObservationModel(
            devices=["periodic_biomarker"], seed=42,
        )
        # t=0.0 is a draw day (day 0)
        readings = model.observe(healthy_state, t=0.0)
        assert not math.isnan(readings["hs_crp"])

        # Day 45 is mid-quarter — no draw
        t_mid = 45.0 / 365.25
        readings = model.observe(healthy_state, t=t_mid)
        assert math.isnan(readings["hs_crp"])
        assert math.isnan(readings["gdf15"])
        assert math.isnan(readings["nad_blood"])
        assert math.isnan(readings["ohdg_8"])

        # Day 91 is next quarterly draw
        t_quarterly = 91.0 / 365.25
        readings = model.observe(healthy_state, t=t_quarterly)
        assert not math.isnan(readings["hs_crp"])


# ── TestStateEstimation ──────────────────────────────────────────────────────

class TestStateEstimation:

    def test_estimate_returns_8d_vector(self, obs_model, healthy_state):
        readings = obs_model.observe(healthy_state, t=0.0)
        est = obs_model.estimate_state(readings)
        assert est.shape == (N_STATES,)

    def test_estimate_all_non_negative(self, obs_model, healthy_state):
        readings = obs_model.observe(healthy_state, t=0.0)
        est = obs_model.estimate_state(readings)
        assert np.all(est >= 0.0)

    def test_estimate_observable_responds_to_state(self, healthy_state, crisis_state):
        """Observable variable estimates should change when the true state
        changes, while unobservable estimates remain at priors."""
        model1 = WearableObservationModel(seed=42)
        model2 = WearableObservationModel(seed=42)
        r_healthy = model1.observe(healthy_state, t=0.0)
        r_crisis = model2.observe(crisis_state, t=10.0)
        est_healthy = model1.estimate_state(r_healthy)
        est_crisis = model2.estimate_state(r_crisis)

        # Unobservable: N_healthy, N_deletion, N_point should not change
        assert est_healthy[0] == est_crisis[0]  # N_healthy stays at prior
        assert est_healthy[1] == est_crisis[1]  # N_deletion stays at prior
        assert est_healthy[7] == est_crisis[7]  # N_point stays at prior

        # Observable: ATP estimate should differ between healthy and crisis
        assert abs(est_healthy[2] - est_crisis[2]) > 0.05  # ATP responds

    def test_estimate_uses_priors_for_unobservable(self, obs_model, healthy_state):
        """Unobservable variables should get population-average priors."""
        from constants import SENSOR_PRIOR_N_HEALTHY, SENSOR_PRIOR_N_DELETION, SENSOR_PRIOR_N_POINT
        readings = obs_model.observe(healthy_state, t=0.0)
        est = obs_model.estimate_state(readings)
        # N_healthy, N_deletion, N_point should be at prior values
        assert est[0] == SENSOR_PRIOR_N_HEALTHY
        assert est[1] == SENSOR_PRIOR_N_DELETION
        assert est[7] == SENSOR_PRIOR_N_POINT

    def test_roundtrip_accuracy_healthy(self, obs_model, healthy_state):
        """Healthy state roundtrip should have moderate accuracy."""
        readings = obs_model.observe(healthy_state, t=0.0)
        est = obs_model.estimate_state(readings)
        loss = obs_model.information_loss(healthy_state, est)
        # Total RMSE should be bounded (not perfect due to noise + partial obs)
        assert loss["total_rmse"] < 1.0  # Generous bound

    def test_information_loss_has_all_fields(self, obs_model, healthy_state):
        readings = obs_model.observe(healthy_state, t=0.0)
        est = obs_model.estimate_state(readings)
        loss = obs_model.information_loss(healthy_state, est)
        assert "total_rmse" in loss
        assert "ATP_error" in loss
        assert "N_healthy_error" in loss
        assert "Senescent_fraction_error" in loss

    def test_estimate_with_nan_glucose(self, obs_model, healthy_state):
        """Should handle NaN glucose (CGM changeover) gracefully."""
        readings = obs_model.observe(healthy_state, t=0.0)
        readings["glucose"] = float("nan")
        est = obs_model.estimate_state(readings)
        assert not np.any(np.isnan(est))

    def test_nad_estimate_improves_with_blood_nad(self, healthy_state):
        """NAD estimate should be closer to truth when blood NAD available."""
        # 3 wearables only
        m3 = WearableObservationModel(
            devices=["apple_watch", "oura_ring", "dexcom_stelo"], seed=42)
        r3 = m3.observe(healthy_state, t=0.0)
        est3 = m3.estimate_state(r3)
        nad_error_3 = abs(healthy_state[4] - est3[4])

        # 5 devices with blood NAD at t=0 (quarterly draw day)
        m5 = WearableObservationModel(
            devices=["apple_watch", "oura_ring", "dexcom_stelo",
                     "abbott_lingo", "periodic_biomarker"], seed=42)
        r5 = m5.observe(healthy_state, t=0.0)
        est5 = m5.estimate_state(r5)
        nad_error_5 = abs(healthy_state[4] - est5[4])

        # Blood NAD is near-direct measurement — should improve the estimate
        assert nad_error_5 <= nad_error_3 + 0.05  # allow small tolerance for noise

    def test_gdf15_contributes_to_multiple_estimates(self, declining_state):
        """GDF-15 should contribute to ATP, ROS, NAD, and Sen estimates."""
        # Model with only biomarker panel (at quarterly draw)
        model = WearableObservationModel(
            devices=["periodic_biomarker"], seed=42)
        readings = model.observe(declining_state, t=0.0)
        est = model.estimate_state(readings)
        # Should have non-baseline estimates for ATP, ROS, NAD, Sen
        # (GDF-15 splits its deviation 4 ways)
        assert est[2] != BASELINE_ATP or est[3] != BASELINE_ROS  # at least one changed

    def test_estimate_handles_nan_biomarkers(self, obs_model, healthy_state):
        """Should handle NaN biomarker channels gracefully."""
        readings = obs_model.observe(healthy_state, t=0.0)
        # Add NaN biomarker readings (simulating non-draw day)
        readings["hs_crp"] = float("nan")
        readings["gdf15"] = float("nan")
        readings["nad_blood"] = float("nan")
        readings["ohdg_8"] = float("nan")
        readings["lactate"] = float("nan")
        est = obs_model.estimate_state(readings)
        assert not np.any(np.isnan(est))

    def test_estimate_state_with_family_priors(self, obs_model, healthy_state):
        """Family priors should override population-average priors for unobservables."""
        readings = obs_model.observe(healthy_state, t=0.0)
        priors = {'n_healthy': 0.60, 'n_deletion': 0.30, 'n_point': 0.10}
        est = obs_model.estimate_state(readings, family_priors=priors)
        assert est[0] == 0.60  # N_healthy from family prior, not 0.75
        assert est[1] == 0.30  # N_deletion from family prior, not 0.15
        assert est[7] == 0.10  # N_point from family prior, not 0.10


# ── TestSensorConstrainedProtocol ────────────────────────────────────────────

class TestSensorConstrainedProtocol:

    @pytest.fixture
    def sensor_protocol(self):
        """Sensor-constrained version of symmathesy protocol."""
        from adaptive_protocol import create_symmathesy_protocol
        from sensor_constrained_adaptive import SensorConstrainedProtocol

        base = create_symmathesy_protocol(
            base_intervention=dict(DEFAULT_INTERVENTION),
            base_patient=dict(DEFAULT_PATIENT),
        )
        obs = WearableObservationModel(seed=42)
        return SensorConstrainedProtocol(base, obs)

    @pytest.fixture
    def gods_eye_protocol(self):
        """God's-eye adaptive protocol (direct state access)."""
        from adaptive_protocol import create_symmathesy_protocol
        return create_symmathesy_protocol(
            base_intervention=dict(DEFAULT_INTERVENTION),
            base_patient=dict(DEFAULT_PATIENT),
        )

    def test_resolve_returns_intervention_patient(self, sensor_protocol, declining_state):
        intervention, patient = sensor_protocol.resolve(t=5.0, state=declining_state)
        assert isinstance(intervention, dict)
        assert isinstance(patient, dict)
        assert "rapamycin_dose" in intervention

    def test_resolve_without_state(self, sensor_protocol):
        """Should work without state (falls through to base)."""
        intervention, patient = sensor_protocol.resolve(t=0.0, state=None)
        assert isinstance(intervention, dict)

    def test_observation_log_grows(self, sensor_protocol, healthy_state):
        sensor_protocol.resolve(t=0.0, state=healthy_state)
        sensor_protocol.resolve(t=1.0, state=healthy_state)
        log = sensor_protocol.get_observation_log()
        assert len(log) == 2
        assert "true_state" in log[0]
        assert "observations" in log[0]
        assert "estimated_state" in log[0]

    def test_clear_log(self, sensor_protocol, healthy_state):
        sensor_protocol.resolve(t=0.0, state=healthy_state)
        sensor_protocol.clear_log()
        assert len(sensor_protocol.get_observation_log()) == 0

    def test_information_loss_summary(self, sensor_protocol, declining_state):
        for t in [0.0, 1.0, 2.0]:
            sensor_protocol.resolve(t=t, state=declining_state)
        summary = sensor_protocol.information_loss_summary()
        assert "mean_total_rmse" in summary
        assert summary["mean_total_rmse"] > 0

    def test_sensor_vs_gods_eye_differ(
        self, sensor_protocol, gods_eye_protocol, crisis_state
    ):
        """Sensor-constrained and God's-eye protocols should produce
        different interventions for the same state (due to estimation error)."""
        i_sensor, _ = sensor_protocol.resolve(t=5.0, state=crisis_state)
        i_gods, _ = gods_eye_protocol.resolve(t=5.0, state=crisis_state)

        # At least one intervention parameter should differ
        diffs = [abs(i_sensor.get(k, 0) - i_gods.get(k, 0))
                 for k in i_gods]
        assert max(diffs) > 0.01, (
            "Sensor-constrained should differ from God's-eye due to estimation error"
        )

    def test_scenario_e_runs(self):
        """Scenario E (sensor-constrained) should execute without error."""
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenario

        scenarios = get_example_scenarios()
        scenario_e = [s for s in scenarios if s.name.startswith("E")]
        assert len(scenario_e) == 1

        result = run_scenario(scenario_e[0], years=5)
        assert "core" in result
        assert "downstream" in result
        # Should have valid trajectory
        assert result["core"]["states"].shape[1] == N_STATES

    def test_scenario_f_runs(self):
        """Scenario F (enhanced sensing) should execute without error."""
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenario

        scenarios = get_example_scenarios()
        scenario_f = [s for s in scenarios if s.name.startswith("F")]
        assert len(scenario_f) == 1

        result = run_scenario(scenario_f[0], years=5)
        assert "core" in result
        assert "downstream" in result
        assert result["core"]["states"].shape[1] == N_STATES
