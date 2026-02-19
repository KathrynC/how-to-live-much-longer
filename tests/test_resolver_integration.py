"""Tests for resolver integration with simulator and disturbances."""
import pytest
import numpy as np


class TestSimulatorWithResolver:
    def test_simulate_accepts_resolver_kwarg(self):
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        result = simulate(resolver=pr)
        assert 'states' in result
        assert result['states'].shape[1] == 8

    def test_resolver_produces_different_result_than_default(self):
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        default = simulate()
        pr = ParameterResolver(
            patient_expanded={'apoe_genotype': 2, 'baseline_age': 70.0},
            intervention_expanded={'nr_dose': 1.0, 'rapamycin_dose': 0.5},
        )
        resolved = simulate(resolver=pr)
        assert not np.allclose(default['states'][-1], resolved['states'][-1])

    def test_resolver_none_is_backwards_compatible(self):
        from simulator import simulate
        r1 = simulate()
        r2 = simulate(resolver=None)
        assert np.allclose(r1['states'], r2['states'])

    def test_resolver_with_stochastic_single_trajectory(self):
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        result = simulate(resolver=pr, stochastic=True, rng_seed=42)
        assert 'states' in result
        assert result['states'].shape[1] == 8

    def test_resolver_with_stochastic_multi_trajectory(self):
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        result = simulate(resolver=pr, stochastic=True, n_trajectories=3,
                          rng_seed=42, sim_years=5.0)
        assert 'states' in result
        assert result['states'].shape[0] == 3  # 3 trajectories


class TestDisturbancesWithResolver:
    def test_simulate_with_disturbances_accepts_resolver(self):
        from disturbances import simulate_with_disturbances, IonizingRadiation
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        shock = IonizingRadiation(start_year=5.0, magnitude=0.5)
        result = simulate_with_disturbances(disturbances=[shock], resolver=pr)
        assert 'states' in result

    def test_disturbances_resolver_none_backwards_compatible(self):
        from disturbances import simulate_with_disturbances
        r1 = simulate_with_disturbances()
        r2 = simulate_with_disturbances(resolver=None)
        assert np.allclose(r1['states'], r2['states'])

    def test_disturbances_resolver_with_shock(self):
        from disturbances import simulate_with_disturbances, ChemotherapyBurst
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={'apoe_genotype': 2},
            intervention_expanded={'nr_dose': 0.5},
        )
        shock = ChemotherapyBurst(start_year=5.0, magnitude=0.8)
        result = simulate_with_disturbances(disturbances=[shock], resolver=pr)
        assert 'states' in result
        assert result['states'].shape[1] == 8
