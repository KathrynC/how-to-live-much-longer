# search_addons

Shared utilities for additive optimization/search tools.

---

## Overview

`search_addons.py` provides common primitives used by add-on methods:

- bounded protocol operators
- simulator-backed objective wrappers
- Pareto sorting / crowding utilities

---

## Key Symbols

- `bounds_dict`
- `clip_protocol`
- `vectorize` / `devectorize`
- `random_protocol`
- `gaussian_mutation`
- `blend_crossover`
- `make_scalar_objective`
- `multi_objectives`
- `dominates`
- `non_dominated_sort`
- `crowding_distance`

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
