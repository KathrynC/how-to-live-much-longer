"""Tests for gradient_refiner local optimization helpers."""

from __future__ import annotations

import numpy as np

import gradient_refiner as gr


def test_extract_seed_protocol_from_single_best_params():
    data = {
        "algorithm": "cma_es",
        "best_params": {
            "rapamycin_dose": 1.2,  # clipped
            "nad_supplement": 0.9,
            "senolytic_dose": 0.8,
            "yamanaka_intensity": -0.2,  # clipped
            "transplant_rate": 0.7,
            "exercise_level": 0.2,
        },
    }
    seed = gr._extract_seed_protocol(data)
    assert set(seed.keys()) == set(gr.INTERVENTION_NAMES)
    assert 0.0 <= seed["rapamycin_dose"] <= 1.0
    assert 0.0 <= seed["yamanaka_intensity"] <= 1.0


def test_extract_seed_protocol_from_profiled_results():
    data = {
        "results": {
            "default": {
                "best_params": {
                    "rapamycin_dose": 0.6,
                    "nad_supplement": 0.7,
                    "senolytic_dose": 0.5,
                    "yamanaka_intensity": 0.0,
                    "transplant_rate": 0.4,
                    "exercise_level": 0.3,
                }
            }
        }
    }
    seed = gr._extract_seed_protocol(data, profile="default")
    assert seed["rapamycin_dose"] == 0.6
    assert seed["transplant_rate"] == 0.4


def test_refine_protocol_improves_toy_quadratic_objective():
    # Max at center 0.5 in all dimensions.
    target = np.full(len(gr.INTERVENTION_NAMES), 0.5, dtype=float)

    def objective(protocol: dict[str, float]) -> dict[str, float]:
        x = np.array([protocol[k] for k in gr.INTERVENTION_NAMES], dtype=float)
        fitness = -float(np.sum((x - target) ** 2))
        return {"fitness": fitness}

    start = {k: 0.0 for k in gr.INTERVENTION_NAMES}
    out = gr.refine_protocol(
        seed_protocol=start,
        objective_fn=objective,
        steps=80,
        lr=0.12,
        fd_rel_step=1e-3,
        patience=30,
    )

    assert out["best_fitness"] > out["seed_fitness"]
    x_best = np.array([out["best_params"][k] for k in gr.INTERVENTION_NAMES], dtype=float)
    # Should move meaningfully toward center from origin.
    assert float(np.mean(x_best)) > 0.2
