"""Tests for the 4-pillar health analytics."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from simulator import simulate
from analytics import (
    compute_energy, compute_damage, compute_dynamics,
    compute_intervention, compute_all, NumpyEncoder,
)
from constants import DEFAULT_INTERVENTION


class TestComputeEnergy:
    """Energy pillar tests."""

    def test_energy_keys(self, default_patient):
        result = simulate()
        energy = compute_energy(result)
        expected_keys = {
            "atp_initial", "atp_final", "atp_min", "atp_max",
            "atp_mean", "atp_cv", "reserve_ratio", "atp_slope",
            "terminal_slope", "time_to_crisis_years",
        }
        assert set(energy.keys()) == expected_keys

    def test_atp_range(self):
        result = simulate()
        energy = compute_energy(result)
        assert energy["atp_min"] <= energy["atp_mean"] <= energy["atp_max"]

    def test_declining_patient_has_crisis(self, near_cliff_patient):
        result = simulate(patient=near_cliff_patient)
        energy = compute_energy(result)
        assert energy["time_to_crisis_years"] < 30.0


class TestComputeDamage:
    """Damage pillar tests."""

    def test_damage_keys(self):
        result = simulate()
        damage = compute_damage(result)
        expected_keys = {
            "het_initial", "het_final", "het_max", "delta_het",
            "het_slope", "het_acceleration", "cliff_distance_initial",
            "cliff_distance_final", "time_to_cliff_years", "frac_above_cliff",
        }
        assert set(damage.keys()) == expected_keys

    def test_cliff_distance_consistency(self):
        result = simulate()
        damage = compute_damage(result)
        assert damage["cliff_distance_initial"] == pytest.approx(
            0.7 - damage["het_initial"], abs=1e-10)


class TestComputeDynamics:
    """Dynamics pillar tests."""

    def test_dynamics_keys(self):
        result = simulate()
        dynamics = compute_dynamics(result)
        expected_keys = {
            "ros_dominant_freq", "ros_amplitude", "membrane_potential_cv",
            "membrane_potential_slope", "nad_slope", "ros_het_correlation",
            "ros_atp_correlation", "senescent_final", "senescent_slope",
        }
        assert set(dynamics.keys()) == expected_keys


class TestComputeIntervention:
    """Intervention pillar tests."""

    def test_intervention_keys(self, cocktail_intervention):
        treated = simulate(intervention=cocktail_intervention)
        baseline = simulate()
        interv = compute_intervention(treated, baseline)
        expected_keys = {
            "atp_benefit_terminal", "atp_benefit_mean",
            "het_benefit_terminal", "energy_cost_per_year",
            "benefit_cost_ratio", "total_dose", "crisis_delay_years",
        }
        assert set(interv.keys()) == expected_keys

    def test_cocktail_has_positive_benefit(self, cocktail_intervention):
        treated = simulate(intervention=cocktail_intervention)
        baseline = simulate()
        interv = compute_intervention(treated, baseline)
        assert interv["atp_benefit_terminal"] > 0
        assert interv["het_benefit_terminal"] > 0

    def test_no_treatment_zero_benefit(self):
        result = simulate()
        interv = compute_intervention(result, result)
        assert interv["atp_benefit_terminal"] == pytest.approx(0.0, abs=1e-10)


class TestComputeAll:
    """Combined analytics tests."""

    def test_all_pillars_present(self):
        result = simulate()
        analytics = compute_all(result)
        assert set(analytics.keys()) == {"energy", "damage", "dynamics", "intervention"}

    def test_json_serialization(self, cocktail_intervention):
        treated = simulate(intervention=cocktail_intervention)
        baseline = simulate()
        analytics = compute_all(treated, baseline)
        json_str = json.dumps(analytics, cls=NumpyEncoder)
        parsed = json.loads(json_str)
        assert "energy" in parsed


class TestNumpyEncoder:
    """JSON encoder tests."""

    def test_numpy_float(self):
        assert json.loads(json.dumps(np.float64(3.14159), cls=NumpyEncoder)) == pytest.approx(3.14159, abs=1e-5)

    def test_numpy_int(self):
        assert json.loads(json.dumps(np.int64(42), cls=NumpyEncoder)) == 42

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = json.loads(json.dumps(arr, cls=NumpyEncoder))
        assert result == [1.0, 2.0, 3.0]
