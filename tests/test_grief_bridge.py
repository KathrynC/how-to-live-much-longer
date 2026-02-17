"""Tests for grief->mito bridge."""
from __future__ import annotations

import numpy as np
import pytest

from grief_bridge import GriefDisturbance, grief_trajectory, grief_scenarios
from disturbances import Disturbance, simulate_with_disturbances
from simulator import simulate


class TestGriefTrajectory:
    """Test grief_trajectory() helper."""

    def test_returns_five_curves(self):
        curves = grief_trajectory()
        assert set(curves.keys()) == {"infl", "cort", "sns", "slp", "cvd_risk", "times"}

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


# -- Task 2: GriefMitoSimulator tests -----------------------------------------

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


# -- Task 3: Grief-Mito Scenarios tests ---------------------------------------

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
