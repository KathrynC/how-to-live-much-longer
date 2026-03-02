"""Integration test -- run all 7 scenarios for the 63yo APOE4 patient end-to-end."""
import pytest
import numpy as np


class TestFourScenarioComparison:
    """Validate that the A-D scenario progression produces monotonically
    improving outcomes, and E/F (sensor-constrained) produce reasonable results."""

    @pytest.fixture(scope='class')
    def all_results(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        return run_scenarios(get_example_scenarios(), years=30)

    def test_seven_scenarios_complete(self, all_results):
        assert len(all_results) == 7

    def test_scenario_b_better_than_a_het(self, all_results):
        a_het = all_results[0]['core']['heteroplasmy'][-1]
        b_het = all_results[1]['core']['heteroplasmy'][-1]
        assert b_het < a_het, "B (OTC supplements) should have lower het than A (sleep only)"

    def test_scenario_c_better_than_b_het(self, all_results):
        b_het = all_results[1]['core']['heteroplasmy'][-1]
        c_het = all_results[2]['core']['heteroplasmy'][-1]
        assert c_het < b_het, "C (prescription) should have lower het than B"

    def test_scenario_d_better_than_c_het(self, all_results):
        c_het = all_results[2]['core']['heteroplasmy'][-1]
        d_het = all_results[3]['core']['heteroplasmy'][-1]
        assert d_het < c_het, "D (experimental) should have lower het than C"

    def test_memory_index_monotonically_better_a_through_d(self, all_results):
        """Scenarios A-D should show monotonically improving memory."""
        memory_finals = [all_results[i]['downstream'][-1]['memory_index'] for i in range(4)]
        for i in range(len(memory_finals) - 1):
            assert memory_finals[i+1] >= memory_finals[i], \
                f"Scenario {i+2} should have >= memory than scenario {i+1}"

    def test_scenario_a_memory_worse_than_b(self, all_results):
        """Scenario A (sleep only) should have worse memory than B (supplements)."""
        a_mi = all_results[0]['downstream'][-1]['memory_index']
        b_mi = all_results[1]['downstream'][-1]['memory_index']
        assert b_mi > a_mi

    def test_scenario_f_has_sensor_config(self, all_results):
        """Scenario F should have enhanced sensing config with 5 devices."""
        from scenario_definitions import get_example_scenarios
        scenarios = get_example_scenarios()
        f_config = scenarios[5].sensor_config
        assert f_config is not None
        assert len(f_config['devices']) == 5
        assert 'abbott_lingo' in f_config['devices']
        assert 'periodic_biomarker' in f_config['devices']


class TestBackwardCompatibilityNoResolver:
    """Simulation without resolver is unaffected by sleep changes."""

    def test_backward_compatibility_no_resolver(self):
        """Simulation without resolver produces identical, clean results."""
        from simulator import simulate

        # Run twice without resolver -- results must be identical
        r1 = simulate()
        r2 = simulate()
        np.testing.assert_array_equal(r1['states'], r2['states'])

        # No NaN or Inf
        assert not np.any(np.isnan(r1['states']))
        assert not np.any(np.isinf(r1['states']))
