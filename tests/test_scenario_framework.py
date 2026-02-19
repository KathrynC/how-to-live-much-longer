"""Tests for scenario framework — definition, running, analysis, plotting."""
import pytest


class TestScenarioDefinitions:
    def test_intervention_profile_defaults_to_zero(self):
        from scenario_definitions import InterventionProfile
        ip = InterventionProfile()
        assert ip.rapamycin_dose == 0.0
        assert ip.nr_dose == 0.0

    def test_scenario_has_required_fields(self):
        from scenario_definitions import Scenario, InterventionProfile
        s = Scenario(
            name="test",
            description="a test scenario",
            patient_params={'baseline_age': 63},
            interventions=InterventionProfile(),
        )
        assert s.name == "test"
        assert s.duration_years == 30.0

    def test_example_scenarios_returns_four(self):
        from scenario_definitions import get_example_scenarios
        scenarios = get_example_scenarios()
        assert len(scenarios) == 4
        assert scenarios[0].name.startswith("A")
        assert scenarios[3].name.startswith("D")


class TestScenarioRunner:
    def test_run_scenario_returns_dict(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenario
        scenarios = get_example_scenarios()
        result = run_scenario(scenarios[0], years=5)
        assert 'core' in result
        assert 'downstream' in result
        assert 'scenario_name' in result

    def test_run_scenarios_returns_all(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        scenarios = get_example_scenarios()[:2]
        results = run_scenarios(scenarios, years=5)
        assert len(results) == 2


class TestScenarioAnalysis:
    def test_extract_milestones(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenario
        from scenario_analysis import extract_milestones
        result = run_scenario(get_example_scenarios()[0], years=5)
        milestones = extract_milestones(result)
        assert 'final_heteroplasmy' in milestones
        assert 'final_atp' in milestones
        assert 'final_memory_index' in milestones

    def test_compare_scenarios(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        from scenario_analysis import compare_scenarios
        results = run_scenarios(get_example_scenarios()[:2], years=5)
        comparison = compare_scenarios(results)
        assert len(comparison) == 2


class TestScenarioPlot:
    def test_plot_trajectories_creates_figure(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        from scenario_plot import plot_trajectories
        results = run_scenarios(get_example_scenarios()[:2], years=5)
        fig = plot_trajectories(results)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
