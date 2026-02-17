"""Tests for surrogate_optimizer add-on module."""

from __future__ import annotations

import numpy as np

import surrogate_optimizer as so


def test_random_intervention_in_bounds():
    rng = np.random.default_rng(0)
    iv = so.random_intervention(rng)
    assert set(iv.keys()) == set(so.INTERVENTION_NAMES)
    for k in so.INTERVENTION_NAMES:
        lo, hi = so.INTERVENTION_PARAMS[k]["range"]
        assert lo <= iv[k] <= hi


def test_knn_surrogate_fits_and_predicts_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 5))
    y = x[:, 0] - 0.5 * x[:, 1] + 0.1 * rng.normal(size=40)
    m = so.KNNRegressorSurrogate(k=5).fit(x, y)
    pred = m.predict(x[:7])
    assert pred.shape == (7,)
    assert np.isfinite(pred).all()


def test_build_training_data_with_toy_objective():
    patient = dict(so.DEFAULT_PATIENT)

    def toy_objective(iv):
        # Deterministic simple objective: mean intervention dose.
        vals = np.array([iv[k] for k in so.INTERVENTION_NAMES], dtype=float)
        return {"fitness": float(np.mean(vals))}

    ds = so.build_training_data(
        patient=patient,
        n_samples=12,
        seed=7,
        objective_fn=toy_objective,
    )
    assert ds["x"].shape[0] == 12
    assert ds["y"].shape == (12,)
    assert len(ds["interventions"]) == 12
