"""Tests for LEMURS->mito bridge."""
from __future__ import annotations

import numpy as np
import pytest

from lemurs_bridge import (
    LEMURSDisturbance, lemurs_trajectory, lemurs_scenarios,
    LEMURS_STUDENT_ARCHETYPES, LEMURS_DEFAULT_INTERVENTION,
    LEMURS_DEFAULT_PATIENT,
)
from disturbances import Disturbance, simulate_with_disturbances
from simulator import simulate


class TestLEMURSTrajectory:
    """Test lemurs_trajectory() helper."""

    def test_returns_six_keys(self):
        curves = lemurs_trajectory()
        assert set(curves.keys()) == {"times", "tst", "pss", "gad7", "hrv", "dac"}

    def test_curves_are_numpy_arrays(self):
        curves = lemurs_trajectory()
        for name, arr in curves.items():
            assert isinstance(arr, np.ndarray), f"{name} is not ndarray"

    def test_curves_have_matching_lengths(self):
        curves = lemurs_trajectory()
        lengths = [len(v) for v in curves.values()]
        assert len(set(lengths)) == 1, f"Mismatched lengths: {lengths}"

    def test_times_included(self):
        curves = lemurs_trajectory()
        assert "times" in curves

    def test_tst_in_valid_range(self):
        curves = lemurs_trajectory()
        assert np.all(curves["tst"] >= 4.0)
        assert np.all(curves["tst"] <= 12.0)

    def test_pss_in_valid_range(self):
        curves = lemurs_trajectory()
        assert np.all(curves["pss"] >= 0.0)
        assert np.all(curves["pss"] <= 40.0)

    def test_gad7_in_valid_range(self):
        curves = lemurs_trajectory()
        assert np.all(curves["gad7"] >= 0.0)
        assert np.all(curves["gad7"] <= 21.0)

    def test_hrv_in_valid_range(self):
        curves = lemurs_trajectory()
        assert np.all(curves["hrv"] >= 15.0)
        assert np.all(curves["hrv"] <= 120.0)

    def test_dac_in_valid_range(self):
        curves = lemurs_trajectory()
        assert np.all(curves["dac"] >= 0.0)
        assert np.all(curves["dac"] <= 1.0)

    def test_custom_patient(self):
        curves = lemurs_trajectory(
            lemurs_patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0},
        )
        assert len(curves["tst"]) > 0


class TestLEMURSDisturbance:
    """Test LEMURSDisturbance class."""

    def test_is_disturbance_subclass(self):
        d = LEMURSDisturbance()
        assert isinstance(d, Disturbance)

    def test_has_modify_state(self):
        d = LEMURSDisturbance()
        assert callable(d.modify_state)

    def test_has_modify_params(self):
        d = LEMURSDisturbance()
        assert callable(d.modify_params)

    def test_is_active_at_start(self):
        d = LEMURSDisturbance(start_year=0.0)
        assert d.is_active(0.0)
        assert d.is_active(1.0)
        assert d.is_active(15.0)

    def test_not_active_before_start(self):
        d = LEMURSDisturbance(start_year=5.0)
        assert not d.is_active(4.0)

    def test_not_active_after_duration(self):
        d = LEMURSDisturbance(start_year=0.0, duration=10.0)
        assert not d.is_active(11.0)

    def test_default_starts_at_year_zero(self):
        d = LEMURSDisturbance()
        assert d.start_year == 0.0

    def test_custom_label(self):
        d = LEMURSDisturbance(label="test_lemurs")
        assert d.name == "test_lemurs"

    def test_default_label(self):
        d = LEMURSDisturbance()
        assert d.name == "LEMURS"

    def test_modify_state_adds_ros(self):
        d = LEMURSDisturbance()
        state = np.array([0.5, 0.3, 0.8, 0.1, 0.6, 0.05, 0.9, 0.01])
        new_state = d.modify_state(state, 0.0)
        # GAD-7-driven ROS should increase state[3]
        assert new_state[3] >= state[3]

    def test_modify_state_does_not_mutate_input(self):
        d = LEMURSDisturbance()
        state = np.array([0.5, 0.3, 0.8, 0.1, 0.6, 0.05, 0.9, 0.01])
        original = state.copy()
        d.modify_state(state, 0.0)
        np.testing.assert_array_equal(state, original)

    def test_modify_params_no_change_outside_window(self):
        d = LEMURSDisturbance(start_year=5.0, duration=10.0)
        patient = {"inflammation_level": 0.1, "metabolic_demand": 1.0,
                   "genetic_vulnerability": 1.0}
        _, new_pat = d.modify_params({}, patient, 1.0)
        assert new_pat["inflammation_level"] == patient["inflammation_level"]


class TestSemesterPhase:
    """Test _semester_phase() logic."""

    @pytest.fixture
    def disturbance(self):
        return LEMURSDisturbance(start_year=0.0)

    def test_fall_semester_detected(self, disturbance):
        # t=0.1 years from start -> ~5.2 weeks into first year -> fall semester
        in_sem, week, n_sem = disturbance._semester_phase(0.1)
        assert in_sem is True
        assert week >= 0
        assert week < 15

    def test_spring_semester_detected(self, disturbance):
        # Spring semester: weeks 22-36 of the academic year
        # t = 25/52 ~ 0.481 years -> week 25 of year -> spring semester
        in_sem, week, n_sem = disturbance._semester_phase(25.0 / 52.0)
        assert in_sem is True
        assert week >= 0
        assert week < 15

    def test_winter_break_detected(self, disturbance):
        # Winter break: weeks 15-21 of the academic year
        # t = 18/52 ~ 0.346 -> week 18 -> winter break
        in_sem, week, n_sem = disturbance._semester_phase(18.0 / 52.0)
        assert in_sem is False

    def test_summer_break_detected(self, disturbance):
        # Summer break: weeks 37-51 of the academic year
        # t = 40/52 ~ 0.769 -> week 40 -> summer break
        in_sem, week, n_sem = disturbance._semester_phase(40.0 / 52.0)
        assert in_sem is False

    def test_semester_counting_year_one(self, disturbance):
        # After fall (week 15+), one semester completed
        _, _, n_sem = disturbance._semester_phase(16.0 / 52.0)
        assert n_sem == 1

    def test_semester_counting_year_two(self, disturbance):
        # Second year, early fall: 2 semesters completed from year 1
        _, _, n_sem = disturbance._semester_phase(1.05)  # slightly into year 2
        assert n_sem == 2

    def test_past_college(self, disturbance):
        # t > college_years (4.0): all semesters completed, not in semester
        in_sem, _, n_sem = disturbance._semester_phase(5.0)
        assert in_sem is False
        assert n_sem == 8  # 4 years x 2 semesters/year

    def test_before_start_year(self, disturbance):
        d = LEMURSDisturbance(start_year=5.0)
        in_sem, _, n_sem = d._semester_phase(3.0)
        assert in_sem is False
        assert n_sem == 0


class TestCouplingChannels:
    """Test each of the 5 coupling channels works in the right direction."""

    @pytest.fixture
    def disturbance(self):
        return LEMURSDisturbance(start_year=0.0)

    def _base_patient(self):
        return {
            "inflammation_level": 0.1,
            "metabolic_demand": 1.0,
            "genetic_vulnerability": 1.0,
        }

    def _base_intervention(self):
        return {"rapamycin_dose": 0.5}

    def test_sleep_deficit_increases_inflammation(self, disturbance):
        """Low TST should increase inflammation_level."""
        patient = self._base_patient()
        intervention = self._base_intervention()
        new_int, new_pat = disturbance.modify_params(intervention, patient, 0.1)
        # TST is below max, so sleep_deficit > 0, inflammation should increase
        assert new_pat["inflammation_level"] >= patient["inflammation_level"]

    def test_high_pss_increases_metabolic_demand(self, disturbance):
        """Stress should increase metabolic_demand."""
        patient = self._base_patient()
        intervention = self._base_intervention()
        _, new_pat = disturbance.modify_params(intervention, patient, 0.1)
        # PSS normalized > 0, so metabolic demand should increase
        assert new_pat["metabolic_demand"] >= patient["metabolic_demand"]

    def test_low_hrv_increases_vulnerability(self, disturbance):
        """Low HRV should increase genetic_vulnerability."""
        patient = self._base_patient()
        intervention = self._base_intervention()
        _, new_pat = disturbance.modify_params(intervention, patient, 0.1)
        # HRV is below max, so hrv_deficit > 0, vulnerability should increase
        assert new_pat["genetic_vulnerability"] >= patient["genetic_vulnerability"]

    def test_low_dac_reduces_rapamycin(self, disturbance):
        """Low DAC should reduce rapamycin_dose."""
        patient = self._base_patient()
        intervention = {"rapamycin_dose": 0.5}
        new_int, _ = disturbance.modify_params(intervention, patient, 0.1)
        # DAC is below 1.0, so dac_deficit > 0, repair scaling should reduce rapamycin
        assert new_int["rapamycin_dose"] <= intervention["rapamycin_dose"]

    def test_high_gad_adds_inflammation(self):
        """GAD-7 above clinical threshold should add extra inflammation."""
        # Use a vulnerable student with high anxiety
        d = LEMURSDisturbance(
            lemurs_patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0,
                            "trauma_load": 3.0},
        )
        patient = self._base_patient()
        intervention = self._base_intervention()
        _, new_pat = d.modify_params(intervention, patient, 0.1)
        # Vulnerable students develop high GAD-7, which should push inflammation higher
        assert new_pat["inflammation_level"] >= patient["inflammation_level"]


class TestAllostatic:
    """Test that later semesters produce larger perturbation magnitude."""

    def test_later_semesters_stronger(self):
        d = LEMURSDisturbance(start_year=0.0)
        patient_early = {
            "inflammation_level": 0.1,
            "metabolic_demand": 1.0,
            "genetic_vulnerability": 1.0,
        }
        patient_late = dict(patient_early)
        intervention = {"rapamycin_dose": 0.0}

        # Early semester: t=0.1 (fall of year 1, 0 semesters completed)
        _, pat_early = d.modify_params(dict(intervention), dict(patient_early), 0.1)

        # Later semester: t=2.1 (fall of year 3, 4 semesters completed)
        _, pat_late = d.modify_params(dict(intervention), dict(patient_late), 2.1)

        # Allostatic accumulation: later perturbation should be larger
        # (both are in fall semester, but more semesters completed for the later one)
        delta_early = pat_early["inflammation_level"] - patient_early["inflammation_level"]
        delta_late = pat_late["inflammation_level"] - patient_late["inflammation_level"]
        assert delta_late > delta_early


class TestPostCollegeDecay:
    """Test that perturbation decays exponentially after college."""

    def test_decay_after_college(self):
        d = LEMURSDisturbance(start_year=0.0)
        patient_base = {
            "inflammation_level": 0.1,
            "metabolic_demand": 1.0,
            "genetic_vulnerability": 1.0,
        }
        intervention = {"rapamycin_dose": 0.0}

        # During college (year 3, fall semester)
        _, pat_during = d.modify_params(dict(intervention), dict(patient_base), 2.1)

        # After college (year 10, well past college window)
        _, pat_after = d.modify_params(dict(intervention), dict(patient_base), 10.0)

        # Post-college perturbation should be much smaller
        delta_during = pat_during["inflammation_level"] - patient_base["inflammation_level"]
        delta_after = pat_after["inflammation_level"] - patient_base["inflammation_level"]
        assert delta_after < delta_during

    def test_deep_post_college_very_small(self):
        d = LEMURSDisturbance(start_year=0.0)
        patient_base = {
            "inflammation_level": 0.1,
            "metabolic_demand": 1.0,
            "genetic_vulnerability": 1.0,
        }
        intervention = {"rapamycin_dose": 0.0}

        # 20 years past college
        _, pat = d.modify_params(dict(intervention), dict(patient_base), 24.0)
        delta = pat["inflammation_level"] - patient_base["inflammation_level"]
        # Should be very close to zero (exp(-0.5 * 20) ~ 4.5e-5)
        assert delta < 0.01


class TestTimescale:
    """Test disturbance timing behavior."""

    def test_inactive_before_start_year(self):
        d = LEMURSDisturbance(start_year=5.0, duration=10.0)
        assert not d.is_active(4.9)

    def test_active_at_start_year(self):
        d = LEMURSDisturbance(start_year=5.0, duration=10.0)
        assert d.is_active(5.0)

    def test_inactive_after_duration(self):
        d = LEMURSDisturbance(start_year=5.0, duration=10.0)
        assert not d.is_active(15.1)


class TestDeterminism:
    """Test that same inputs produce same outputs."""

    def test_same_trajectory(self):
        c1 = lemurs_trajectory(
            lemurs_patient={"emotional_stability": 4.0},
        )
        c2 = lemurs_trajectory(
            lemurs_patient={"emotional_stability": 4.0},
        )
        np.testing.assert_array_equal(c1["tst"], c2["tst"])
        np.testing.assert_array_equal(c1["pss"], c2["pss"])
        np.testing.assert_array_equal(c1["gad7"], c2["gad7"])

    def test_same_modify_params(self):
        d = LEMURSDisturbance()
        patient = {"inflammation_level": 0.2, "metabolic_demand": 1.0,
                   "genetic_vulnerability": 1.0}
        intervention = {"rapamycin_dose": 0.5}

        _, p1 = d.modify_params(dict(intervention), dict(patient), 0.5)
        _, p2 = d.modify_params(dict(intervention), dict(patient), 0.5)
        assert p1["inflammation_level"] == p2["inflammation_level"]
        assert p1["metabolic_demand"] == p2["metabolic_demand"]
        assert p1["genetic_vulnerability"] == p2["genetic_vulnerability"]


class TestScenarios:
    """Test lemurs_scenarios() convenience function."""

    def test_returns_list_of_disturbances(self):
        scenarios = lemurs_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        for s in scenarios:
            assert isinstance(s, LEMURSDisturbance)

    def test_correct_count(self):
        scenarios = lemurs_scenarios()
        # 8 archetypes x 2 (with/without support) = 16
        assert len(scenarios) == len(LEMURS_STUDENT_ARCHETYPES) * 2

    def test_all_have_labels(self):
        scenarios = lemurs_scenarios()
        for s in scenarios:
            assert s.name is not None
            assert len(s.name) > 0

    def test_includes_no_support_and_full_support(self):
        scenarios = lemurs_scenarios()
        labels = [s.name for s in scenarios]
        has_no_support = any("no_support" in l for l in labels)
        has_full_support = any("full_support" in l for l in labels)
        assert has_no_support
        assert has_full_support

    def test_scenarios_run_without_crash(self):
        """Smoke test: first 4 scenarios run through simulate_with_disturbances."""
        scenarios = lemurs_scenarios()
        for s in scenarios[:4]:
            result = simulate_with_disturbances(disturbances=[s])
            assert not np.any(np.isnan(result["states"])), f"NaN in {s.name}"


class TestLEMURSDisturbanceIntegration:
    """Integration tests: LEMURS disturbance in the mito simulator."""

    @pytest.fixture(scope="class")
    def baseline(self):
        return simulate()

    @pytest.fixture(scope="class")
    def stressed_student(self):
        d = LEMURSDisturbance(
            lemurs_patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0,
                            "trauma_load": 3.0},
        )
        return simulate_with_disturbances(disturbances=[d])

    @pytest.fixture(scope="class")
    def supported_student(self):
        d = LEMURSDisturbance(
            lemurs_patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0,
                            "trauma_load": 3.0},
            lemurs_intervention={"nature_rx": 0.8, "exercise_rx": 0.8,
                                 "therapy_rx": 0.5, "sleep_hygiene": 0.8},
        )
        return simulate_with_disturbances(disturbances=[d])

    def test_no_nan_in_stressed(self, stressed_student):
        assert not np.any(np.isnan(stressed_student["states"]))

    def test_no_negative_states(self, stressed_student):
        assert np.all(stressed_student["states"] >= -1e-10)

    def test_stressed_higher_het_than_baseline(self, baseline, stressed_student):
        """College stress should cause more mitochondrial damage."""
        assert stressed_student["heteroplasmy"][-1] > baseline["heteroplasmy"][-1]

    def test_interventions_reduce_het(self, stressed_student, supported_student):
        """LEMURS interventions should reduce mitochondrial damage."""
        assert supported_student["heteroplasmy"][-1] < stressed_student["heteroplasmy"][-1]

    def test_composes_with_radiation(self, stressed_student):
        """LEMURS stress + radiation should stack."""
        from disturbances import IonizingRadiation
        shocks = [
            LEMURSDisturbance(
                lemurs_patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0},
            ),
            IonizingRadiation(start_year=10.0, magnitude=0.5),
        ]
        result = simulate_with_disturbances(disturbances=shocks)
        assert not np.any(np.isnan(result["states"]))
        # Combined should be worse than LEMURS alone
        assert result["heteroplasmy"][-1] > stressed_student["heteroplasmy"][-1]
