"""Tests for the CA Zimmerman protocol adapters."""
import pytest

from ca_zimmerman_bridge import (
    MitoCASimulator,
    MitoTissueSimulator,
    MitoCAEnsembleSimulator,
)


class TestMitoCASimulator:
    def test_param_spec(self):
        sim = MitoCASimulator()
        spec = sim.param_spec()
        assert len(spec) == 12
        assert "rapamycin_dose" in spec
        assert "baseline_age" in spec

    def test_run_default(self):
        sim = MitoCASimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())
        assert "ca_final_attractor" in result

    def test_no_nan_inf(self):
        sim = MitoCASimulator()
        result = sim.run({})
        for k, v in result.items():
            assert not (v != v), f"NaN in {k}"
            assert v != float("inf"), f"Inf in {k}"
            assert v != float("-inf"), f"-Inf in {k}"

    def test_has_final_state_bins(self):
        sim = MitoCASimulator()
        result = sim.run({})
        assert "ca_final_ATP" in result
        assert "ca_final_N_deletion" in result

    def test_deterministic(self):
        sim = MitoCASimulator()
        r1 = sim.run({})
        r2 = sim.run({})
        assert r1 == r2


class TestMitoTissueSimulator:
    def test_param_spec(self):
        sim = MitoTissueSimulator()
        spec = sim.param_spec()
        assert "tissue_coupling" in spec
        assert len(spec) == 13

    def test_run_default(self):
        sim = MitoTissueSimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_has_tissue_attractors(self):
        sim = MitoTissueSimulator()
        result = sim.run({})
        assert "tissue_brain_attractor" in result
        assert "tissue_muscle_attractor" in result

    def test_no_nan_inf(self):
        sim = MitoTissueSimulator()
        result = sim.run({})
        for k, v in result.items():
            assert not (v != v), f"NaN in {k}"
            assert v != float("inf"), f"Inf in {k}"


class TestMitoCAEnsembleSimulator:
    def test_param_spec(self):
        sim = MitoCAEnsembleSimulator(n_trials=5)
        spec = sim.param_spec()
        assert len(spec) == 12

    def test_run_default(self):
        sim = MitoCAEnsembleSimulator(n_trials=5)
        result = sim.run({})
        assert "ens_cliff_crossing_prob" in result
        assert isinstance(result, dict)

    def test_probabilities_valid(self):
        sim = MitoCAEnsembleSimulator(n_trials=5)
        result = sim.run({})
        for k, v in result.items():
            if "prob" in k:
                assert 0.0 <= v <= 1.0, f"{k}={v} out of range"

    def test_no_nan_inf(self):
        sim = MitoCAEnsembleSimulator(n_trials=5)
        result = sim.run({})
        for k, v in result.items():
            assert not (v != v), f"NaN in {k}"
