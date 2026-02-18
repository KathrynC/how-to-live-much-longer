# Optimization Reference (Add-ons)

This document is part of the How to Live Much Longer simulation project.

Symbol reference for the additive optimization stack:

- `gradient_refiner.py`
- `surrogate_optimizer.py`
- `ml_prefilter_runner.py`

These functions are additive utilities and do not replace core simulator or
analytics logic.

---

## `load_seed_protocol[path, profile]`

### Usage

Load a seed intervention protocol from a JSON artifact.

### Arguments

- `path` (`str`) — Artifact JSON file path.
- `profile` (`str | None`) — Optional profile key for multi-profile artifacts.

### Details

- Supports multiple artifact schemas (`best_params`, `best_runs`, `results`).
- Normalizes and clips returned values to intervention bounds.

### Returns

- `dict[str, float]` — Full 6D intervention dictionary.

### See Also

- `refine_protocol`
- `run_prefilter`

---

## `make_objective[patient, metric]`

### Usage

Create a simulator-backed scalar objective function over intervention
parameters.

### Arguments

- `patient` (`dict[str, float]`) — Patient profile.
- `metric` (`str`) — One of `combined`, `atp`, `het`, `crisis_delay`.

### Details

- Caches no-treatment baseline for the selected patient.
- Preserves project metric semantics:
  - `combined = ATP benefit + 0.5 * heteroplasmy benefit`

### Returns

- `Callable[[dict[str, float]], dict[str, float]]`

### See Also

- `build_training_data`
- `evaluate_candidates`

---

## `finite_difference_gradient[x, objectiveFn, relStep]`

### Usage

Estimate the local gradient of scalar fitness by central finite differences.

### Arguments

- `x` (`np.ndarray`) — Current intervention vector.
- `objectiveFn` (`callable`) — Objective function from `make_objective`.
- `relStep` (`float`) — Relative step size per parameter range.

### Details

- Applies bounds-aware perturbations on each axis.
- Returns both gradient vector and L2 norm.

### Returns

- `(np.ndarray, float)` — `(gradient, gradient_norm)`.

### See Also

- `refine_protocol`

---

## `refine_protocol[seedProtocol, objectiveFn, ...]`

### Usage

Run bounded Adam ascent to locally improve a seed protocol.

### Arguments

- `seedProtocol` (`dict[str, float]`) — Initial intervention.
- `objectiveFn` (`callable`) — Scalar objective function.
- `steps` (`int`) — Maximum iterations.
- `lr` (`float`) — Learning rate.
- `fd_rel_step` (`float`) — Finite-difference relative step.
- `patience` (`int`) — Early-stop patience.

### Details

- Uses finite-difference gradients (no autograd dependency).
- Projects every update back into intervention bounds.
- Records full optimization history for auditability.

### Returns

- `dict` containing seed/best params, fitness improvement, and trajectory.

### See Also

- `load_seed_protocol`
- `finite_difference_gradient`

---

## `clip_intervention[intervention]`

### Usage

Clamp intervention values into valid configured parameter ranges.

### Arguments

- `intervention` (`dict[str, float]`) — Partial or full intervention dictionary.

### Details

- Ensures canonical 6D key set.

### Returns

- `dict[str, float]`

### See Also

- `random_intervention`

---

## `random_intervention[rng]`

### Usage

Sample one intervention uniformly in each parameter range.

### Arguments

- `rng` (`np.random.Generator`) — Random number generator.

### Returns

- `dict[str, float]`

### See Also

- `build_candidate_pool`
- `build_training_data`

---

## `encode_features[intervention, patient]`

### Usage

Encode intervention and patient dictionaries into one numeric feature vector.

### Arguments

- `intervention` (`dict[str, float]`)
- `patient` (`dict[str, float]`)

### Details

- Feature order is fixed:
  - `INTERVENTION_NAMES` then `PATIENT_NAMES`.

### Returns

- `np.ndarray`

### See Also

- `KNNRegressorSurrogate.fit`
- `KNNRegressorSurrogate.predict`

---

## `KNNRegressorSurrogate`

### Usage

Distance-weighted K-nearest-neighbor regressor (numpy implementation).

### Arguments

- `k` (`int`) — Number of neighbors.
- `distance_eps` (`float`) — Numerical stabilizer in inverse-distance weights.

### Details

- Minimal dependency profile for easy portability.
- Deterministic for fixed training/query arrays.

### Returns

- Fitted object with `fit` and `predict`.

### See Also

- `build_training_data`
- `run_prefilter`

---

## `build_training_data[patient, nSamples, seed, objectiveFn]`

### Usage

Generate surrogate training data by evaluating random interventions with the
true simulator objective.

### Arguments

- `patient` (`dict[str, float]`)
- `nSamples` (`int`)
- `seed` (`int`)
- `objectiveFn` (`callable | None`)

### Returns

- `dict` with `x`, `y`, and `interventions`.

### See Also

- `KNNRegressorSurrogate.fit`

---

## `evaluate_candidates[candidates, objectiveFn]`

### Usage

Evaluate candidate interventions using true simulation/analytics objective.

### Arguments

- `candidates` (`list[dict[str, float]]`)
- `objectiveFn` (`callable`)

### Details

- This is the ground-truth validation layer after surrogate ranking.

### Returns

- `list[dict]` with fitness and key outcome metrics.

### See Also

- `run_prefilter`

---

## `build_candidate_pool[poolSize, rng, seedProtocols, ...]`

### Usage

Construct candidate pool from random samples plus local perturbations around
seed protocols.

### Arguments

- `poolSize` (`int`) — Number of random samples.
- `rng` (`np.random.Generator`)
- `seedProtocols` (`list[dict] | None`)
- `perturbPerSeed` (`int`)
- `perturbSigma` (`float`)

### Returns

- `list[dict[str, float]]`

### See Also

- `run_prefilter`

---

## `run_prefilter[patientName, metric, trainSamples, poolSize, topK, ...]`

### Usage

Run end-to-end surrogate prefilter workflow and return artifact payload.

### Arguments

- `patientName` (`str`)
- `metric` (`str`)
- `trainSamples` (`int`)
- `poolSize` (`int`)
- `topK` (`int`)
- `seed` (`int`)
- `seedProtocols` (`list[dict] | None`)
- `knnK` (`int`)

### Details

- Trains surrogate from true labels.
- Ranks pool by predicted fitness.
- Re-evaluates top-K and random-control sets with true objective.
- Reports top-vs-random true-fitness lift.

### Returns

- `dict` — Full experiment report suitable for JSON artifact.

### See Also

- `build_candidate_pool`
- `evaluate_candidates`

---

## `main[]` (`gradient_refiner.py`, `ml_prefilter_runner.py`)

### Usage

CLI entry points for their respective modules.

### Details

- Parse command-line flags.
- Execute experiment.
- Save artifact JSON under `artifacts/` by default.

### See Also

- `gradient_refiner.md`
- `ml_prefilter_runner.md`
- `surrogate_optimizer.md`
