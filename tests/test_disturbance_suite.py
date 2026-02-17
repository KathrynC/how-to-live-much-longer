"""Additional disturbance-suite tests for robustness and composition."""
from __future__ import annotations

import numpy as np
import pytest

from disturbances import (
    ChemotherapyBurst,
    InflammationBurst,
    IonizingRadiation,
    ToxinExposure,
    simulate_with_disturbances,
)


def _assert_valid_result(result: dict) -> None:
    """Shared validity checks for disturbance simulation outputs."""
    states = result["states"]
    assert states.shape[1] == 8
    assert np.all(np.isfinite(states))
    assert np.all(states >= 0.0)
    assert np.all((0.0 <= result["heteroplasmy"]) & (result["heteroplasmy"] <= 1.0))
    assert np.all(
        (0.0 <= result["deletion_heteroplasmy"])
        & (result["deletion_heteroplasmy"] <= 1.0)
    )
    assert np.all(
        result["deletion_heteroplasmy"] <= result["heteroplasmy"] + 1e-12
    )


@pytest.mark.parametrize(
    "disturbance",
    [
        IonizingRadiation(start_year=3.0, magnitude=0.6),
        ToxinExposure(start_year=3.0, magnitude=0.6),
        ChemotherapyBurst(start_year=3.0, magnitude=0.6),
        InflammationBurst(start_year=3.0, magnitude=0.6),
    ],
)
def test_each_disturbance_runs_with_valid_outputs(disturbance):
    """Each disturbance type should yield numerically valid trajectories."""
    result = simulate_with_disturbances(disturbances=[disturbance], sim_years=8)
    _assert_valid_result(result)
    assert len(result["disturbances"]) == 1
    assert len(result["shock_times"]) == 1


def test_non_overlapping_disturbance_order_invariance():
    """Order of non-overlapping disturbances should not change trajectory."""
    d1 = IonizingRadiation(start_year=2.0, duration=0.5, magnitude=0.5)
    d2 = ToxinExposure(start_year=6.0, duration=0.5, magnitude=0.4)

    r12 = simulate_with_disturbances(disturbances=[d1, d2], sim_years=8)
    r21 = simulate_with_disturbances(disturbances=[d2, d1], sim_years=8)

    np.testing.assert_allclose(r12["states"], r21["states"], atol=1e-10)
    np.testing.assert_allclose(r12["heteroplasmy"], r21["heteroplasmy"], atol=1e-10)
    np.testing.assert_allclose(
        r12["deletion_heteroplasmy"], r21["deletion_heteroplasmy"], atol=1e-10
    )


def test_stacked_disturbances_still_stable():
    """A stacked multi-disturbance scenario should stay finite and bounded."""
    shocks = [
        InflammationBurst(start_year=1.5, magnitude=0.7),
        IonizingRadiation(start_year=3.0, magnitude=0.8),
        ChemotherapyBurst(start_year=5.0, magnitude=0.8),
        ToxinExposure(start_year=7.0, magnitude=0.7),
    ]
    result = simulate_with_disturbances(disturbances=shocks, sim_years=10)
    _assert_valid_result(result)
    assert len(result["disturbances"]) == 4
    assert len(result["shock_times"]) == 4
