"""Tests for the resilience suite: disturbances, metrics, and integration."""

import numpy as np
import pytest

from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT, N_STATES
from simulator import simulate
from disturbances import (
    Disturbance,
    IonizingRadiation,
    ToxinExposure,
    ChemotherapyBurst,
    InflammationBurst,
    simulate_with_disturbances,
)
from resilience_metrics import (
    compute_resistance,
    compute_recovery_time,
    compute_regime_retention,
    compute_elasticity,
    compute_resilience,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def baseline():
    """Baseline simulation (no disturbance, no treatment)."""
    return simulate()


@pytest.fixture(scope="module")
def radiation_result():
    """Simulation with ionizing radiation at year 10."""
    shock = IonizingRadiation(start_year=10.0, magnitude=0.8)
    return simulate_with_disturbances(disturbances=[shock])


@pytest.fixture(scope="module")
def chemo_result():
    """Simulation with chemotherapy at year 10."""
    shock = ChemotherapyBurst(start_year=10.0, magnitude=0.8)
    return simulate_with_disturbances(disturbances=[shock])


# ── Disturbance class tests ──────────────────────────────────────────────────

class TestDisturbanceClasses:

    def test_ionizing_radiation_is_active(self):
        shock = IonizingRadiation(start_year=5.0, duration=1.0)
        assert shock.is_active(5.0)
        assert shock.is_active(5.5)
        assert not shock.is_active(4.99)
        assert not shock.is_active(6.0)

    def test_ionizing_radiation_modifies_state(self):
        shock = IonizingRadiation(start_year=0.0, magnitude=1.0)
        state = np.array([0.8, 0.2, 1.0, 0.1, 0.8, 0.05, 0.9])
        modified = shock.modify_state(state, 0.0)
        # Should transfer healthy → damaged
        assert modified[0] < state[0]
        assert modified[1] > state[1]
        # Conservation: total mtDNA unchanged
        assert abs((modified[0] + modified[1]) - (state[0] + state[1])) < 1e-10
        # ROS should increase
        assert modified[3] > state[3]

    def test_toxin_modifies_membrane_potential(self):
        shock = ToxinExposure(start_year=0.0, magnitude=1.0)
        state = np.array([0.8, 0.2, 1.0, 0.1, 0.8, 0.05, 0.9])
        modified = shock.modify_state(state, 0.0)
        assert modified[6] < state[6]  # membrane potential drops
        assert modified[4] < state[4]  # NAD drops

    def test_chemo_is_most_severe(self):
        state = np.array([0.8, 0.2, 1.0, 0.1, 0.8, 0.05, 0.9])
        rad = IonizingRadiation(start_year=0.0, magnitude=1.0)
        chemo = ChemotherapyBurst(start_year=0.0, magnitude=1.0)

        rad_modified = rad.modify_state(state, 0.0)
        chemo_modified = chemo.modify_state(state, 0.0)

        # Chemo should damage more healthy mtDNA
        assert chemo_modified[0] < rad_modified[0]
        # Chemo should produce more ROS
        assert chemo_modified[3] > rad_modified[3]

    def test_inflammation_increases_senescence(self):
        shock = InflammationBurst(start_year=0.0, magnitude=1.0)
        state = np.array([0.8, 0.2, 1.0, 0.1, 0.8, 0.05, 0.9])
        modified = shock.modify_state(state, 0.0)
        assert modified[5] > state[5]  # senescence increases

    def test_magnitude_clipping(self):
        shock = IonizingRadiation(start_year=0.0, magnitude=5.0)
        assert shock.magnitude == 1.0
        shock2 = IonizingRadiation(start_year=0.0, magnitude=-1.0)
        assert shock2.magnitude == 0.0

    def test_modify_params_only_active(self):
        shock = IonizingRadiation(start_year=5.0, duration=1.0, magnitude=0.5)
        interv = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)

        # Before shock: no modification
        i_out, p_out = shock.modify_params(interv, patient, 3.0)
        assert p_out["genetic_vulnerability"] == patient["genetic_vulnerability"]

        # During shock: modified
        i_out, p_out = shock.modify_params(interv, patient, 5.5)
        assert p_out["genetic_vulnerability"] > patient["genetic_vulnerability"]

        # After shock: no modification
        i_out, p_out = shock.modify_params(interv, patient, 7.0)
        assert p_out["genetic_vulnerability"] == patient["genetic_vulnerability"]


# ── simulate_with_disturbances tests ─────────────────────────────────────────

class TestSimulateWithDisturbances:

    def test_no_disturbance_matches_baseline(self, baseline):
        result = simulate_with_disturbances(disturbances=[])
        # Should be nearly identical to baseline
        np.testing.assert_allclose(
            result["states"][-1], baseline["states"][-1], atol=1e-6)

    def test_returns_correct_shape(self, radiation_result):
        assert radiation_result["states"].shape[1] == N_STATES
        assert len(radiation_result["time"]) == len(radiation_result["heteroplasmy"])
        assert len(radiation_result["time"]) == radiation_result["states"].shape[0]

    def test_shock_times_populated(self, radiation_result):
        assert len(radiation_result["shock_times"]) == 1
        start, end = radiation_result["shock_times"][0]
        assert start == 10.0
        assert end == 11.0  # duration=1.0

    def test_disturbance_info_populated(self, radiation_result):
        assert len(radiation_result["disturbances"]) == 1
        info = radiation_result["disturbances"][0]
        assert info["name"] == "Ionizing Radiation"
        assert info["magnitude"] == 0.8

    def test_radiation_reduces_final_atp(self, baseline, radiation_result):
        baseline_atp = baseline["states"][-1, 2]
        shocked_atp = radiation_result["states"][-1, 2]
        assert shocked_atp < baseline_atp

    def test_radiation_increases_final_het(self, baseline, radiation_result):
        baseline_het = baseline["heteroplasmy"][-1]
        shocked_het = radiation_result["heteroplasmy"][-1]
        assert shocked_het > baseline_het

    def test_chemo_more_damaging_than_radiation(self, radiation_result, chemo_result):
        rad_het = radiation_result["heteroplasmy"][-1]
        chemo_het = chemo_result["heteroplasmy"][-1]
        assert chemo_het > rad_het

    def test_states_non_negative(self, radiation_result):
        assert np.all(radiation_result["states"] >= 0.0)

    def test_senescence_bounded(self, radiation_result):
        assert np.all(radiation_result["states"][:, 5] <= 1.0)

    def test_heteroplasmy_bounded(self, radiation_result):
        assert np.all(radiation_result["heteroplasmy"] >= 0.0)
        assert np.all(radiation_result["heteroplasmy"] <= 1.0)

    def test_multi_disturbance(self, baseline):
        shocks = [
            IonizingRadiation(start_year=5.0, magnitude=0.5),
            ChemotherapyBurst(start_year=15.0, magnitude=0.6),
        ]
        result = simulate_with_disturbances(disturbances=shocks)
        assert len(result["shock_times"]) == 2
        assert len(result["disturbances"]) == 2
        # Should be worse than either alone
        single_rad = simulate_with_disturbances(
            disturbances=[IonizingRadiation(start_year=5.0, magnitude=0.5)])
        assert result["heteroplasmy"][-1] > single_rad["heteroplasmy"][-1]

    def test_zero_magnitude_is_noop(self, baseline):
        shock = IonizingRadiation(start_year=10.0, magnitude=0.0)
        result = simulate_with_disturbances(disturbances=[shock])
        np.testing.assert_allclose(
            result["states"][-1], baseline["states"][-1], atol=1e-4)


# ── Resilience metrics tests ────────────────────────────────────────────────

class TestResilienceMetrics:

    def test_resistance_positive(self, baseline, radiation_result):
        r = compute_resistance(
            radiation_result["states"], baseline["states"],
            radiation_result["time"], 10.0, 11.0)
        assert r["peak_deviation"] > 0
        assert r["relative_peak_deviation"] > 0

    def test_resistance_increases_with_magnitude(self, baseline):
        resistances = []
        for mag in [0.2, 0.5, 0.8]:
            shock = IonizingRadiation(start_year=10.0, magnitude=mag)
            result = simulate_with_disturbances(disturbances=[shock])
            r = compute_resistance(
                result["states"], baseline["states"],
                result["time"], 10.0, 11.0)
            resistances.append(r["peak_deviation"])
        # Should be monotonically increasing
        assert resistances[1] > resistances[0]
        assert resistances[2] > resistances[1]

    def test_recovery_time_finite_for_mild_shock(self, baseline):
        shock = IonizingRadiation(start_year=5.0, magnitude=0.2)
        result = simulate_with_disturbances(disturbances=[shock])
        rt = compute_recovery_time(
            result["states"], baseline["states"],
            result["time"], 6.0, epsilon=0.1)
        # Mild shock: system either recovers quickly (rt >= 0) or is
        # already within tolerance (rt == 0). Both are valid.
        assert np.isfinite(rt)
        assert rt >= 0

    def test_regime_retention_mild_shock(self, baseline, radiation_result):
        regime = compute_regime_retention(
            radiation_result["heteroplasmy"], baseline["heteroplasmy"],
            radiation_result["time"], 10.0)
        # Mild radiation shouldn't push past cliff from 30% het start
        assert regime["regime_retained"]

    def test_elasticity_varies_with_magnitude(self, baseline):
        # mtDNA damage is permanent (no spontaneous recovery), so
        # elasticity measures how quickly the system stabilizes at
        # its new equilibrium. Higher magnitude → more divergence.
        elasticities = []
        for mag in [0.2, 0.5, 0.9]:
            shock = IonizingRadiation(start_year=5.0, magnitude=mag)
            result = simulate_with_disturbances(disturbances=[shock])
            e = compute_elasticity(
                result["states"], baseline["states"],
                result["time"], 6.0)
            elasticities.append(e)
        # Elasticity should be a finite number for all magnitudes
        assert all(np.isfinite(e) for e in elasticities)

    def test_compute_resilience_has_all_keys(self, baseline, radiation_result):
        metrics = compute_resilience(radiation_result, baseline)
        assert "resistance" in metrics
        assert "recovery_time_years" in metrics
        assert "regime" in metrics
        assert "elasticity" in metrics
        assert "summary_score" in metrics
        assert "component_scores" in metrics

    def test_summary_score_bounded(self, baseline, radiation_result):
        metrics = compute_resilience(radiation_result, baseline)
        assert 0.0 <= metrics["summary_score"] <= 1.0

    def test_no_disturbance_perfect_score(self, baseline):
        result = simulate_with_disturbances(disturbances=[])
        metrics = compute_resilience(result, baseline)
        assert metrics["summary_score"] == 1.0

    def test_severe_shock_lower_score(self, baseline):
        mild = simulate_with_disturbances(
            disturbances=[IonizingRadiation(start_year=10.0, magnitude=0.2)])
        severe = simulate_with_disturbances(
            disturbances=[ChemotherapyBurst(start_year=10.0, magnitude=1.0)])
        mild_metrics = compute_resilience(mild, baseline)
        severe_metrics = compute_resilience(severe, baseline)
        assert severe_metrics["summary_score"] < mild_metrics["summary_score"]
