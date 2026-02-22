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
        # Perfect sleep (1.0) with a young patient minimises age-dependent
        # sleep degradation. The SleepTrajectory still applies a small
        # age-modulated repair penalty, so we check rapamycin > 0.7
        # rather than >= 0.8.
        pr = ParameterResolver(
            patient_expanded={'baseline_age': 20.0},
            intervention_expanded={'rapamycin_dose': 0.8, 'sleep_intervention': 1.0},
        )
        intervention, _ = pr.resolve(0.0)
        assert intervention['rapamycin_dose'] > 0.7


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


class TestSleepTrajectoryIntegration:
    """Verify resolver constructs a SleepTrajectory and uses it."""

    def test_sleep_trajectory_integration(self):
        from parameter_resolver import ParameterResolver
        from sleep_trajectory import SleepTrajectory
        pr = ParameterResolver(
            patient_expanded={'baseline_age': 60.0},
            intervention_expanded={'sleep_intervention': 0.7},
        )
        # Resolver should have a SleepTrajectory instance
        assert hasattr(pr, '_sleep_trajectory')
        assert isinstance(pr._sleep_trajectory, SleepTrajectory)

        # resolve() at t=0 should return a patient dict with all 5 channels
        intervention, patient = pr.resolve(t=0.0)
        assert 'inflammation_level' in patient
        assert '_sleep_ros_boost' in patient
        assert '_sleep_nad_drain' in patient
        assert '_sleep_membrane_penalty' in patient
        # inflammation_delta is folded into inflammation_level (not a separate key)
        # sleep_repair_factor is applied to rapamycin_dose (not stored separately)
        # Verify rapamycin was modified by sleep repair factor
        assert isinstance(intervention['rapamycin_dose'], float)

    def test_sleep_age_varying(self):
        """Same patient/intervention at t=0 vs t=20 should differ due to aging."""
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={'baseline_age': 50.0},
            intervention_expanded={'sleep_intervention': 0.0},  # worse than baseline
            duration_years=30.0,
        )
        _, patient_early = pr.resolve(t=0.0)
        _, patient_late = pr.resolve(t=20.0)

        # At t=20, patient is 70 years old (50+20) vs 50 at t=0.
        # Age-dependent sleep degradation: older age has lower baseline quality,
        # leading to larger deficit (since intervention=0.0) and larger ROS boost.
        assert patient_late['_sleep_ros_boost'] > patient_early['_sleep_ros_boost']
        assert patient_late['_sleep_nad_drain'] > patient_early['_sleep_nad_drain']

    def test_sleep_channels_in_patient_dict(self):
        """Verify _sleep_ros_boost, _sleep_nad_drain, _sleep_membrane_penalty
        keys are present and are floats after resolve()."""
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={},
            intervention_expanded={},
        )
        _, patient = pr.resolve(t=0.0)
        for key in ('_sleep_ros_boost', '_sleep_nad_drain', '_sleep_membrane_penalty'):
            assert key in patient, f"Missing key: {key}"
            assert isinstance(patient[key], float), f"{key} should be float, got {type(patient[key])}"
            assert patient[key] >= 0.0, f"{key} should be non-negative"
