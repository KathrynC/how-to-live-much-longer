"""Tests for the Zimmerman Simulator protocol bridge.

Verifies that MitoSimulator satisfies the zimmerman-toolkit Simulator
protocol and produces valid, deterministic outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure zimmerman-toolkit is importable
PROJECT = Path(__file__).resolve().parent.parent
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"
if str(ZIMMERMAN_PATH) not in sys.path:
    sys.path.insert(0, str(ZIMMERMAN_PATH))

from zimmerman.base import Simulator
from zimmerman_bridge import MitoSimulator
from constants import (
    INTERVENTION_NAMES, PATIENT_NAMES,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sim_full():
    """Full 12D simulator (intervention + patient)."""
    return MitoSimulator()


@pytest.fixture
def sim_iv():
    """6D intervention-only simulator."""
    return MitoSimulator(intervention_only=True)


@pytest.fixture
def default_params():
    """Default 12D parameter dict."""
    return {**DEFAULT_INTERVENTION, **DEFAULT_PATIENT}


# ── Protocol compliance ──────────────────────────────────────────────────────

class TestProtocol:
    """Verify MitoSimulator satisfies the Simulator protocol."""

    def test_isinstance_full(self, sim_full):
        assert isinstance(sim_full, Simulator)

    def test_isinstance_intervention_only(self, sim_iv):
        assert isinstance(sim_iv, Simulator)

    def test_has_run_method(self, sim_full):
        assert callable(getattr(sim_full, "run", None))

    def test_has_param_spec_method(self, sim_full):
        assert callable(getattr(sim_full, "param_spec", None))


# ── param_spec ────────────────────────────────────────────────────────────────

class TestParamSpec:
    """Verify param_spec returns correct structure."""

    def test_full_mode_returns_12_keys(self, sim_full):
        spec = sim_full.param_spec()
        assert len(spec) == 12

    def test_intervention_only_returns_6_keys(self, sim_iv):
        spec = sim_iv.param_spec()
        assert len(spec) == 6

    def test_full_mode_has_all_param_names(self, sim_full):
        spec = sim_full.param_spec()
        for name in INTERVENTION_NAMES + PATIENT_NAMES:
            assert name in spec, f"Missing param: {name}"

    def test_intervention_only_has_intervention_names(self, sim_iv):
        spec = sim_iv.param_spec()
        for name in INTERVENTION_NAMES:
            assert name in spec, f"Missing param: {name}"
        for name in PATIENT_NAMES:
            assert name not in spec, f"Unexpected patient param: {name}"

    def test_bounds_are_tuples(self, sim_full):
        spec = sim_full.param_spec()
        for name, bounds in spec.items():
            assert isinstance(bounds, tuple), f"{name} bounds not tuple"
            assert len(bounds) == 2, f"{name} bounds not length 2"
            lo, hi = bounds
            assert lo < hi, f"{name}: lo={lo} >= hi={hi}"


# ── run() output ──────────────────────────────────────────────────────────────

class TestRunOutput:
    """Verify run() returns valid scalar metrics."""

    def test_returns_dict(self, sim_full, default_params):
        result = sim_full.run(default_params)
        assert isinstance(result, dict)

    def test_all_values_are_float(self, sim_full, default_params):
        result = sim_full.run(default_params)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is {type(v).__name__}, not float"

    def test_no_nan_values(self, sim_full, default_params):
        result = sim_full.run(default_params)
        for k, v in result.items():
            assert not np.isnan(v), f"{k} is NaN"

    def test_no_inf_values(self, sim_full, default_params):
        result = sim_full.run(default_params)
        for k, v in result.items():
            assert not np.isinf(v), f"{k} is inf"

    def test_has_pillar_prefixes(self, sim_full, default_params):
        result = sim_full.run(default_params)
        prefixes = {"energy_", "damage_", "dynamics_", "intervention_"}
        found = set()
        for k in result:
            for p in prefixes:
                if k.startswith(p):
                    found.add(p)
        assert found == prefixes, f"Missing pillar prefixes: {prefixes - found}"

    def test_has_final_endpoints(self, sim_full, default_params):
        result = sim_full.run(default_params)
        for key in ["final_heteroplasmy", "final_atp", "final_ros",
                     "final_nad", "final_senescent", "final_membrane_potential"]:
            assert key in result, f"Missing endpoint: {key}"

    def test_intervention_only_mode_works(self, sim_iv):
        params = {k: 0.5 for k in INTERVENTION_NAMES}
        result = sim_iv.run(params)
        assert isinstance(result, dict)
        assert len(result) > 0
        for v in result.values():
            assert isinstance(v, float)
            assert not np.isnan(v)


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    """Verify identical inputs produce identical outputs."""

    def test_same_input_same_output(self, sim_full, default_params):
        r1 = sim_full.run(default_params)
        r2 = sim_full.run(default_params)
        assert r1.keys() == r2.keys()
        for k in r1:
            assert r1[k] == r2[k], f"{k}: {r1[k]} != {r2[k]}"

    def test_deterministic_across_instances(self, default_params):
        sim1 = MitoSimulator()
        sim2 = MitoSimulator()
        r1 = sim1.run(default_params)
        r2 = sim2.run(default_params)
        for k in r1:
            assert r1[k] == r2[k], f"{k}: {r1[k]} != {r2[k]}"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Verify robustness at parameter boundaries."""

    def test_all_zero_intervention(self, sim_full):
        params = {**DEFAULT_INTERVENTION, **DEFAULT_PATIENT}
        for k in INTERVENTION_NAMES:
            params[k] = 0.0
        result = sim_full.run(params)
        assert all(not np.isnan(v) for v in result.values())

    def test_all_max_intervention(self, sim_full):
        params = {**{k: 1.0 for k in INTERVENTION_NAMES}, **DEFAULT_PATIENT}
        result = sim_full.run(params)
        assert all(not np.isnan(v) for v in result.values())

    def test_near_cliff_patient(self, sim_full):
        params = {
            **DEFAULT_INTERVENTION,
            "baseline_age": 80.0,
            "baseline_heteroplasmy": 0.65,
            "baseline_nad_level": 0.4,
            "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0,
            "inflammation_level": 0.5,
        }
        result = sim_full.run(params)
        assert all(not np.isnan(v) for v in result.values())

    def test_young_healthy_patient(self, sim_full):
        params = {
            **DEFAULT_INTERVENTION,
            "baseline_age": 25.0,
            "baseline_heteroplasmy": 0.05,
            "baseline_nad_level": 0.95,
            "genetic_vulnerability": 0.5,
            "metabolic_demand": 0.5,
            "inflammation_level": 0.0,
        }
        result = sim_full.run(params)
        assert all(not np.isnan(v) for v in result.values())

    def test_patient_override(self):
        custom_patient = {
            "baseline_age": 50.0,
            "baseline_heteroplasmy": 0.2,
            "baseline_nad_level": 0.8,
            "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0,
            "inflammation_level": 0.1,
        }
        sim = MitoSimulator(intervention_only=True, patient_override=custom_patient)
        spec = sim.param_spec()
        assert len(spec) == 6
        result = sim.run({k: 0.5 for k in INTERVENTION_NAMES})
        assert isinstance(result, dict)


# ── Biological sanity ─────────────────────────────────────────────────────────

class TestBiologicalSanity:
    """Verify outputs are biologically reasonable."""

    def test_heteroplasmy_in_range(self, sim_full, default_params):
        result = sim_full.run(default_params)
        het = result["final_heteroplasmy"]
        assert 0.0 <= het <= 1.0, f"het={het} out of [0, 1]"

    def test_atp_non_negative(self, sim_full, default_params):
        result = sim_full.run(default_params)
        assert result["final_atp"] >= 0.0

    def test_treatment_improves_outcomes(self):
        sim = MitoSimulator()
        no_treatment = {**DEFAULT_INTERVENTION, **DEFAULT_PATIENT}
        with_treatment = {
            "rapamycin_dose": 0.5, "nad_supplement": 0.75,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5,
            **DEFAULT_PATIENT,
        }
        r_none = sim.run(no_treatment)
        r_treat = sim.run(with_treatment)
        # Treatment should reduce heteroplasmy
        assert r_treat["final_heteroplasmy"] < r_none["final_heteroplasmy"]
