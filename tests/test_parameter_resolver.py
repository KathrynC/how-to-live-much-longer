"""Tests for parameter resolver — 50D expanded → effective 12D core."""
import pytest
import numpy as np


class TestParameterResolverConstruction:
    def test_constructs_with_minimal_params(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        assert pr is not None

    def test_resolve_returns_two_dicts(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        intervention, patient = pr.resolve(t=0.0)
        assert isinstance(intervention, dict)
        assert isinstance(patient, dict)

    def test_resolve_returns_valid_core_keys(self):
        from parameter_resolver import ParameterResolver
        from constants import INTERVENTION_NAMES, PATIENT_NAMES
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        intervention, patient = pr.resolve(t=0.0)
        for k in INTERVENTION_NAMES:
            assert k in intervention
        for k in PATIENT_NAMES:
            assert k in patient


class TestGeneticResolution:
    def test_apoe4_increases_vulnerability(self):
        from parameter_resolver import ParameterResolver
        baseline = ParameterResolver(
            patient_expanded={'apoe_genotype': 0},
            intervention_expanded={},
        )
        apoe4 = ParameterResolver(
            patient_expanded={'apoe_genotype': 1},
            intervention_expanded={},
        )
        _, p_base = baseline.resolve(0.0)
        _, p_apoe = apoe4.resolve(0.0)
        assert p_apoe['genetic_vulnerability'] > p_base['genetic_vulnerability']


class TestSupplementResolution:
    def test_nr_increases_nad_supplement(self):
        from parameter_resolver import ParameterResolver
        without = ParameterResolver(
            patient_expanded={}, intervention_expanded={},
        )
        with_nr = ParameterResolver(
            patient_expanded={}, intervention_expanded={'nr_dose': 0.8},
        )
        i_base, _ = without.resolve(0.0)
        i_nr, _ = with_nr.resolve(0.0)
        assert i_nr['nad_supplement'] > i_base['nad_supplement']


class TestAlcoholResolution:
    def test_alcohol_increases_inflammation(self):
        from parameter_resolver import ParameterResolver
        sober = ParameterResolver(
            patient_expanded={}, intervention_expanded={'alcohol_intake': 0.0},
        )
        drinker = ParameterResolver(
            patient_expanded={}, intervention_expanded={'alcohol_intake': 0.8},
        )
        _, p_sober = sober.resolve(0.0)
        _, p_drink = drinker.resolve(0.0)
        assert p_drink['inflammation_level'] > p_sober['inflammation_level']


class TestTimeVaryingGrief:
    def test_grief_decays_over_time(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={'grief_intensity': 0.9, 'therapy_intensity': 0.5},
            intervention_expanded={},
        )
        _, p_early = pr.resolve(0.0)
        _, p_late = pr.resolve(20.0)
        assert p_late['inflammation_level'] < p_early['inflammation_level']


class TestTimeVaryingAlcohol:
    def test_alcohol_taper_reduces_over_time(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={},
            intervention_expanded={'alcohol_intake': 0.8},
            schedules={'alcohol_taper': {'start': 0.8, 'end': 0.0, 'taper_years': 2}},
        )
        _, p_before = pr.resolve(0.0)
        _, p_after = pr.resolve(5.0)
        assert p_after['inflammation_level'] < p_before['inflammation_level']


class TestCoreSchedulePassthrough:
    def test_rapamycin_passed_through(self):
        from parameter_resolver import ParameterResolver
        # Perfect sleep (1.0) ensures no sleep-related repair degradation
        pr = ParameterResolver(
            patient_expanded={},
            intervention_expanded={'rapamycin_dose': 0.8, 'sleep_intervention': 1.0},
        )
        intervention, _ = pr.resolve(0.0)
        assert intervention['rapamycin_dose'] >= 0.8


class TestOutputsClamped:
    def test_inflammation_clamped_to_one(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={
                'apoe_genotype': 2,
                'grief_intensity': 1.0,
                'sex': 'F',
                'menopause_status': 'post',
                'baseline_age': 90.0,
                'inflammation_level': 0.9,
            },
            intervention_expanded={'alcohol_intake': 1.0},
        )
        _, patient = pr.resolve(0.0)
        assert patient['inflammation_level'] <= 1.0
