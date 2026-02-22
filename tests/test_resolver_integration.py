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


class TestSleepAgeSweep:
    """End-to-end test: older patients experience worse sleep effects on mitochondria."""

    def test_sleep_age_sweep(self):
        """Older patients experience worse sleep effects on mitochondria."""
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        from constants import DEFAULT_PATIENT

        ages = [30, 50, 70, 80]
        final_atp = []
        final_nad = []

        for age in ages:
            patient = dict(DEFAULT_PATIENT, baseline_age=float(age))
            resolver = ParameterResolver(
                patient_expanded={'baseline_age': float(age), 'apoe_genotype': 0, 'sex': 'M'},
                intervention_expanded={'sleep_intervention': 0.5, 'alcohol_intake': 0.1},
            )
            result = simulate(
                patient=patient,
                resolver=resolver,
                sim_years=20,
            )
            final_atp.append(result['states'][-1, 2])
            final_nad.append(result['states'][-1, 4])

        # Older patients should have lower ATP and NAD
        # (combination of age effects + worse sleep quality at older ages)
        for i in range(len(ages) - 1):
            assert final_atp[i] >= final_atp[i + 1], \
                f"ATP at age {ages[i]} ({final_atp[i]:.6f}) should be >= ATP at age {ages[i+1]} ({final_atp[i+1]:.6f})"
        for i in range(len(ages) - 1):
            assert final_nad[i] >= final_nad[i + 1], \
                f"NAD at age {ages[i]} ({final_nad[i]:.6f}) should be >= NAD at age {ages[i+1]} ({final_nad[i+1]:.6f})"


class TestSleepInterventionDoseResponse:
    """Higher sleep intervention produces better mitochondrial outcomes."""

    def test_sleep_intervention_dose_response(self):
        """Higher sleep intervention produces better mitochondrial outcomes."""
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        from constants import DEFAULT_PATIENT

        interventions = [0.0, 0.25, 0.5, 0.75, 1.0]
        final_atp = []
        final_het = []

        patient = dict(DEFAULT_PATIENT, baseline_age=70.0)
        for si in interventions:
            resolver = ParameterResolver(
                patient_expanded={'baseline_age': 70.0, 'apoe_genotype': 0, 'sex': 'M'},
                intervention_expanded={'sleep_intervention': si},
            )
            result = simulate(patient=patient, resolver=resolver)
            final_atp.append(result['states'][-1, 2])
            final_het.append(result['heteroplasmy'][-1])

        # Higher intervention should give better ATP (monotonic)
        for i in range(len(interventions) - 1):
            assert final_atp[i] <= final_atp[i + 1], \
                f"ATP should increase with sleep intervention: {interventions[i]} ({final_atp[i]:.6f}) vs {interventions[i+1]} ({final_atp[i+1]:.6f})"
        # Higher intervention should give lower heteroplasmy (monotonic)
        for i in range(len(interventions) - 1):
            assert final_het[i] >= final_het[i + 1], \
                f"Het should decrease with sleep intervention: {interventions[i]} ({final_het[i]:.6f}) vs {interventions[i+1]} ({final_het[i+1]:.6f})"


class TestAPOE4SleepInteraction:
    """APOE4 carriers are more vulnerable to poor sleep."""

    def test_apoe4_sleep_interaction(self):
        """APOE4 carriers are more vulnerable to poor sleep."""
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        from constants import DEFAULT_PATIENT

        patient = dict(DEFAULT_PATIENT, baseline_age=70.0)

        # Wild type (apoe_genotype=0)
        resolver_wt = ParameterResolver(
            patient_expanded={'baseline_age': 70.0, 'apoe_genotype': 0, 'sex': 'M'},
            intervention_expanded={'sleep_intervention': 0.2},  # poor sleep
        )
        result_wt = simulate(patient=patient, resolver=resolver_wt)

        # APOE4 heterozygous (apoe_genotype=1)
        resolver_apoe = ParameterResolver(
            patient_expanded={'baseline_age': 70.0, 'apoe_genotype': 1, 'sex': 'M'},
            intervention_expanded={'sleep_intervention': 0.2},  # same poor sleep
        )
        result_apoe = simulate(patient=patient, resolver=resolver_apoe)

        # APOE4 should have lower final ATP and higher heteroplasmy
        assert result_apoe['states'][-1, 2] < result_wt['states'][-1, 2], \
            "APOE4 carrier should have lower ATP under poor sleep"
        assert result_apoe['heteroplasmy'][-1] > result_wt['heteroplasmy'][-1], \
            "APOE4 carrier should have higher heteroplasmy under poor sleep"
