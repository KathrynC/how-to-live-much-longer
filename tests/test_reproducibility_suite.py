"""Reproducibility-focused tests for deterministic and stochastic paths."""
from __future__ import annotations

import numpy as np

from simulator import simulate


class TestDeterministicReproducibility:
    """Deterministic simulator should be bit-stable for fixed inputs."""

    def test_deterministic_repeatable(self):
        r1 = simulate(sim_years=8)
        r2 = simulate(sim_years=8)
        np.testing.assert_allclose(r1["states"], r2["states"], atol=0.0)
        np.testing.assert_allclose(r1["heteroplasmy"], r2["heteroplasmy"], atol=0.0)
        np.testing.assert_allclose(
            r1["deletion_heteroplasmy"], r2["deletion_heteroplasmy"], atol=0.0
        )


class TestStochasticReproducibility:
    """Stochastic simulator should be seed-reproducible and seed-sensitive."""

    def test_single_trajectory_same_seed_same_path(self):
        r1 = simulate(stochastic=True, noise_scale=0.01, rng_seed=123, sim_years=5)
        r2 = simulate(stochastic=True, noise_scale=0.01, rng_seed=123, sim_years=5)
        np.testing.assert_allclose(r1["states"], r2["states"], atol=0.0)
        np.testing.assert_allclose(r1["heteroplasmy"], r2["heteroplasmy"], atol=0.0)

    def test_single_trajectory_different_seed_differs(self):
        r1 = simulate(stochastic=True, noise_scale=0.01, rng_seed=123, sim_years=5)
        r2 = simulate(stochastic=True, noise_scale=0.01, rng_seed=124, sim_years=5)
        assert not np.allclose(r1["states"], r2["states"])

    def test_multi_trajectory_same_seed_same_ensemble(self):
        r1 = simulate(
            stochastic=True,
            noise_scale=0.01,
            n_trajectories=6,
            rng_seed=222,
            sim_years=5,
        )
        r2 = simulate(
            stochastic=True,
            noise_scale=0.01,
            n_trajectories=6,
            rng_seed=222,
            sim_years=5,
        )
        np.testing.assert_allclose(r1["states"], r2["states"], atol=0.0)
        np.testing.assert_allclose(r1["heteroplasmy"], r2["heteroplasmy"], atol=0.0)

    def test_multi_trajectory_different_seed_differs(self):
        r1 = simulate(
            stochastic=True,
            noise_scale=0.01,
            n_trajectories=6,
            rng_seed=222,
            sim_years=5,
        )
        r2 = simulate(
            stochastic=True,
            noise_scale=0.01,
            n_trajectories=6,
            rng_seed=223,
            sim_years=5,
        )
        assert not np.allclose(r1["states"], r2["states"])
