# gradient_refiner

Local gradient-based protocol refinement (EA -> GD hybrid).

---

## Overview

`gradient_refiner.py` performs bounded local optimization in the 6D
intervention space:

1. Load a seed protocol (typically `best_params` from EA artifacts).
2. Estimate local gradients using central finite differences.
3. Apply Adam-style gradient ascent with projection to parameter bounds.

This tool is intended as a second-stage exploit step after global exploration
by evolutionary algorithms.

---

## Key Functions

### `load_seed_protocol(path, profile=None) -> dict`

Loads a seed protocol from common artifact schemas:

- `best_params` (single-run artifacts)
- `best_runs[profile].best_params` (transfer artifacts)
- `results[...].best_params` (comparison / constrained artifacts)

### `make_objective(patient, metric="combined") -> callable`

Builds the scalar objective used by the refiner with cached no-treatment
baseline simulation.

Supported metrics:

- `combined`: ATP benefit + 0.5 * heteroplasmy benefit
- `atp`: ATP benefit
- `het`: heteroplasmy benefit
- `crisis_delay`: crisis-delay years from intervention analytics

### `finite_difference_gradient(x, objective_fn, rel_step=1e-3) -> (grad, norm)`

Central finite-difference gradient over intervention parameters.

### `refine_protocol(seed_protocol, objective_fn, ...) -> dict`

Runs bounded Adam ascent and returns:

- seed fitness
- best fitness + improvement
- best protocol parameters
- per-step history (`fitness`, `grad_norm`, `params`)

---

## CLI

```bash
python gradient_refiner.py
python gradient_refiner.py --seed-artifact artifacts/ea_cma_es.json
python gradient_refiner.py --seed-artifact artifacts/ea_protocol_transfer_2026-02-17.json --profile near_cliff_80 --patient near_cliff_80
```

Main options:

- `--patient`: patient profile key
- `--metric`: objective metric (`combined`, `atp`, `het`, `crisis_delay`)
- `--steps`: optimization iterations
- `--lr`: Adam learning rate
- `--fd-rel-step`: finite-difference step as range fraction
- `--patience`: early-stop patience
- `--out`: output JSON path

---

## Outputs

Writes `artifacts/gd_refiner_<patient>_<date>.json` (or `--out`) with:

- optimization settings
- seed and best protocols
- fitness improvement summary
- trajectory history for downstream plotting/audit

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
