"""Integration test -- run all 4 scenarios for the 63yo APOE4 patient end-to-end."""
import pytest


class TestFourScenarioComparison:
    """Validate that the A-D scenario progression produces monotonically
    improving outcomes, matching the projected trajectories from the handoff."""

    @pytest.fixture(scope='class')
    def all_results(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        return run_scenarios(get_example_scenarios(), years=30)

    def test_four_scenarios_complete(self, all_results):
        assert len(all_results) == 4

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

    def test_memory_index_monotonically_better(self, all_results):
        memory_finals = [r['downstream'][-1]['memory_index'] for r in all_results]
        for i in range(len(memory_finals) - 1):
            assert memory_finals[i+1] >= memory_finals[i], \
                f"Scenario {i+2} should have >= memory than scenario {i+1}"

    def test_scenario_a_memory_worse_than_b(self, all_results):
        """Scenario A (sleep only) should have worse memory than B (supplements)."""
        a_mi = all_results[0]['downstream'][-1]['memory_index']
        b_mi = all_results[1]['downstream'][-1]['memory_index']
        assert b_mi > a_mi
