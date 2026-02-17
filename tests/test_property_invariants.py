"""Property-style invariants across randomized valid parameter settings."""
from __future__ import annotations

import random

import numpy as np
import pytest

from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT, INTERVENTION_NAMES
from simulator import simulate


def _random_case(seed: int) -> tuple[dict[str, float], dict[str, float]]:
    """Generate one valid intervention/patient pair from documented bounds."""
    rng = random.Random(seed)
    intervention = dict(DEFAULT_INTERVENTION)
    patient = dict(DEFAULT_PATIENT)

    for k in INTERVENTION_NAMES:
        intervention[k] = rng.random()

    patient["baseline_age"] = rng.uniform(20.0, 90.0)
    patient["baseline_heteroplasmy"] = rng.uniform(0.0, 0.95)
    patient["baseline_nad_level"] = rng.uniform(0.2, 1.0)
    patient["genetic_vulnerability"] = rng.uniform(0.5, 2.0)
    patient["metabolic_demand"] = rng.uniform(0.5, 2.0)
    patient["inflammation_level"] = rng.uniform(0.0, 1.0)
    return intervention, patient


@pytest.mark.parametrize("seed", list(range(20)))
def test_core_invariants_hold_under_random_valid_inputs(seed: int):
    """Core numerical/biological invariants should hold over random cases."""
    intervention, patient = _random_case(seed)
    result = simulate(intervention=intervention, patient=patient, sim_years=8)

    states = result["states"]
    total_het = result["heteroplasmy"]
    deletion_het = result["deletion_heteroplasmy"]

    # Shape and metadata coherence
    assert states.ndim == 2
    assert states.shape[1] == 8
    assert len(result["time"]) == states.shape[0]
    assert len(total_het) == states.shape[0]
    assert len(deletion_het) == states.shape[0]

    # Numerical sanity
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(total_het))
    assert np.all(np.isfinite(deletion_het))

    # Non-negativity and bounded key fractions
    assert np.all(states >= 0.0)
    assert np.all((0.0 <= total_het) & (total_het <= 1.0))
    assert np.all((0.0 <= deletion_het) & (deletion_het <= 1.0))

    # Deletion heteroplasmy is always a subset of total heteroplasmy.
    assert np.all(deletion_het <= total_het + 1e-12)

    # Time should be monotonically non-decreasing.
    assert np.all(np.diff(result["time"]) >= 0.0)
