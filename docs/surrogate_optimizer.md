# surrogate_optimizer

Add-on surrogate modeling utilities for protocol ranking.

---

## Overview

`surrogate_optimizer.py` provides lightweight ML primitives that sit on top of
the existing simulator and analytics stack:

- Build supervised datasets from true simulator evaluations.
- Fit a KNN regressor surrogate (`numpy` only).
- Score candidate interventions before true re-evaluation.

This is intended to reduce expensive simulator calls during search while
keeping simulator + analytics as the ground truth.

---

## Key Functions

### `make_objective(patient, metric="combined") -> callable`

Creates the true simulator-backed objective function:

- `combined`: ATP benefit + 0.5 * heteroplasmy benefit
- `atp`
- `het`
- `crisis_delay`

### `build_training_data(patient, n_samples, seed, objective_fn=None) -> dict`

Generates supervised data (`x`, `y`) by random intervention sampling and true
objective evaluation.

### `encode_features(intervention, patient) -> np.ndarray`

Feature encoding for surrogate input:

- 6 intervention dimensions
- 6 patient dimensions

### `KNNRegressorSurrogate.fit(x, y) / predict(x)`

Distance-weighted KNN regressor for fast local approximation.

### `evaluate_candidates(candidates, objective_fn) -> list[dict]`

Re-evaluates candidate protocols with the true objective and returns metrics.

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
