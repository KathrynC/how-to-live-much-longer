"""Tests for the Cramer Toolkit bridge.

Verifies that domain-specific biological stress scenarios are valid,
the scenario bank is complete, and convenience analysis functions
produce correct outputs from the mitochondrial simulator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure toolkits are importable
PROJECT = Path(__file__).resolve().parent.parent
CRAMER_PATH = PROJECT.parent / "cramer-toolkit"
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"
for p in (CRAMER_PATH, ZIMMERMAN_PATH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cramer.base import Scenario, ScenarioSet, Simulator
from cramer import BASELINE, ScenarioSimulator

from kcramer_bridge import (
    MitoSimulator,
    INFLAMMATION_SCENARIOS,
    NAD_SCENARIOS,
    VULNERABILITY_SCENARIOS,
    DEMAND_SCENARIOS,
    AGING_SCENARIOS,
    COMBINED_SCENARIOS,
    ALL_STRESS_SCENARIOS,
    PROTOCOLS,
    run_vulnerability_analysis,
)
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sim():
    """Full 12D MitoSimulator (shared across module for speed)."""
    return MitoSimulator()


@pytest.fixture(scope="module")
def default_params():
    """Default 12D parameter dict."""
    return {**DEFAULT_INTERVENTION, **DEFAULT_PATIENT}


# ── Scenario bank structure ─────────────────────────────────────────────────

class TestScenarioBank:
    """Verify scenario bank is properly constructed."""

    def test_inflammation_is_scenario_set(self):
        assert isinstance(INFLAMMATION_SCENARIOS, ScenarioSet)

    def test_nad_is_scenario_set(self):
        assert isinstance(NAD_SCENARIOS, ScenarioSet)

    def test_vulnerability_is_scenario_set(self):
        assert isinstance(VULNERABILITY_SCENARIOS, ScenarioSet)

    def test_demand_is_scenario_set(self):
        assert isinstance(DEMAND_SCENARIOS, ScenarioSet)

    def test_aging_is_scenario_set(self):
        assert isinstance(AGING_SCENARIOS, ScenarioSet)

    def test_combined_is_scenario_set(self):
        assert isinstance(COMBINED_SCENARIOS, ScenarioSet)

    def test_all_stress_is_scenario_set(self):
        assert isinstance(ALL_STRESS_SCENARIOS, ScenarioSet)

    def test_all_stress_count(self):
        expected = (
            len(INFLAMMATION_SCENARIOS)
            + len(NAD_SCENARIOS)
            + len(VULNERABILITY_SCENARIOS)
            + len(DEMAND_SCENARIOS)
            + len(AGING_SCENARIOS)
            + len(COMBINED_SCENARIOS)
        )
        assert len(ALL_STRESS_SCENARIOS) == expected

    def test_all_scenarios_are_scenario_type(self):
        for s in ALL_STRESS_SCENARIOS:
            assert isinstance(s, Scenario)

    def test_all_scenarios_have_unique_names(self):
        names = [s.name for s in ALL_STRESS_SCENARIOS]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_scenario_names_are_descriptive(self):
        for s in ALL_STRESS_SCENARIOS:
            assert len(s.name) > 3, f"Name too short: {s.name}"
            assert "_" in s.name or len(s.name) > 8, f"Name not descriptive: {s.name}"


# ── Scenario application ────────────────────────────────────────────────────

class TestScenarioApplication:
    """Verify scenarios modify parameters correctly."""

    def test_inflammation_targets_correct_param(self, default_params):
        s = INFLAMMATION_SCENARIOS[0]  # mild_inflammaging
        modified = s.apply(default_params)
        assert modified["inflammation_level"] != default_params["inflammation_level"]
        # Other params should be unchanged
        for k in default_params:
            if k != "inflammation_level":
                assert modified[k] == default_params[k], f"{k} was modified"

    def test_nad_targets_correct_param(self, default_params):
        s = NAD_SCENARIOS[0]  # mild_nad_decline
        modified = s.apply(default_params)
        assert modified["baseline_nad_level"] != default_params["baseline_nad_level"]

    def test_combined_modifies_multiple_params(self, default_params):
        s = COMBINED_SCENARIOS["inflamed_nad_depleted"]
        modified = s.apply(default_params)
        assert modified["inflammation_level"] != default_params["inflammation_level"]
        assert modified["baseline_nad_level"] != default_params["baseline_nad_level"]

    def test_worst_case_modifies_four_params(self, default_params):
        s = COMBINED_SCENARIOS["worst_case_patient"]
        modified = s.apply(default_params)
        changed = sum(
            1 for k in default_params if modified[k] != default_params[k]
        )
        assert changed >= 4


# ── ScenarioSimulator integration ──────────────────────────────────────────

class TestScenarioSimulatorIntegration:
    """Verify ScenarioSimulator works with MitoSimulator."""

    def test_wrapped_satisfies_protocol(self, sim):
        wrapped = ScenarioSimulator(sim, BASELINE)
        assert isinstance(wrapped, Simulator)

    def test_baseline_is_identity(self, sim, default_params):
        wrapped = ScenarioSimulator(sim, BASELINE)
        base = sim.run(default_params)
        wrap = wrapped.run(default_params)
        # All numeric outputs should match (except _scenario tag)
        for k in base:
            assert wrap[k] == pytest.approx(base[k], abs=1e-10), f"{k} differs"

    def test_inflammation_changes_output(self, sim, default_params):
        wrapped = ScenarioSimulator(sim, INFLAMMATION_SCENARIOS["severe_inflammation"])
        base = sim.run(default_params)
        stressed = wrapped.run(default_params)
        # Severe inflammation should worsen ATP
        assert stressed["final_atp"] != base["final_atp"]

    def test_wrapped_produces_no_nan(self, sim, default_params):
        for s in list(ALL_STRESS_SCENARIOS)[:5]:  # spot-check first 5
            wrapped = ScenarioSimulator(sim, s)
            result = wrapped.run(default_params)
            for k, v in result.items():
                if k == "_scenario":
                    continue
                assert not (isinstance(v, float) and v != v), \
                    f"NaN in {k} under scenario {s.name}"


# ── Protocol bank ──────────────────────────────────────────────────────────

class TestProtocols:
    """Verify protocol definitions are valid."""

    def test_all_protocols_have_6_intervention_keys(self):
        from constants import INTERVENTION_NAMES
        for name, protocol in PROTOCOLS.items():
            for k in INTERVENTION_NAMES:
                assert k in protocol, f"Protocol {name} missing key {k}"

    def test_protocol_values_in_range(self):
        for name, protocol in PROTOCOLS.items():
            for k, v in protocol.items():
                assert 0.0 <= v <= 1.0, f"Protocol {name}: {k}={v} out of [0,1]"

    def test_no_treatment_is_all_zeros(self):
        assert all(v == 0.0 for v in PROTOCOLS["no_treatment"].values())

    def test_aggressive_has_nonzero_transplant(self):
        assert PROTOCOLS["aggressive"]["transplant_rate"] > 0

    def test_transplant_focused_maxes_transplant(self):
        assert PROTOCOLS["transplant_focused"]["transplant_rate"] == 1.0


# ── Convenience functions ───────────────────────────────────────────────────

class TestConvenienceFunctions:
    """Verify helper analysis wrappers return expected shapes."""

    def test_run_vulnerability_analysis_returns_ranked_list(self, sim):
        profile = run_vulnerability_analysis(
            sim=sim,
            protocol=PROTOCOLS["moderate"],
            scenarios=INFLAMMATION_SCENARIOS,
            output_key="final_atp",
        )
        assert isinstance(profile, list)
        assert len(profile) > 0
        assert "scenario" in profile[0]
        assert "impact" in profile[0]
