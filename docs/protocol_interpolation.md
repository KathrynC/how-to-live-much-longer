# protocol_interpolation

Linear interpolation between champion intervention protocols in 6D space.

---

## Overview

Adapted from `gait_interpolation.py` in the parent Evolutionary-Robotics project. Interpolates between known-effective intervention protocols to map the fitness landscape and find synergistic combinations ("super-protocols" whose fitness exceeds both endpoints).

---

## Experiments

### 1. Pairwise Interpolation

5 champion protocols × C(5,2) = 10 pairs × 21 alpha steps. For each pair, linearly interpolate from protocol A (alpha=0) to protocol B (alpha=1). Detect "super-protocols" where peak ATP exceeds the better endpoint by > 0.005.

### 2. Radial Sweep

Interpolate from no-treatment center to each of 5 champions (5 × 21 steps). Measures marginal gain at each dose level and detects the alpha at which diminishing returns begin.

### 3. 3D Grid

10×10×10 grid through (rapamycin, NAD, exercise) subspace with other params at default. Finds optimal point in the most clinically relevant 3D slice.

---

## Champion Protocols

| Name | Strategy |
|------|----------|
| `rapamycin_heavy` | Rapamycin 0.75, moderate everything else |
| `nad_heavy` | NAD 0.75, moderate everything else |
| `full_cocktail` | Balanced 0.5 across rapamycin, NAD, senolytics, exercise |
| `transplant_focused` | Transplant 0.75, minimal everything else |
| `yamanaka_cautious` | Yamanaka 0.25 with NAD 0.5 support |

---

## Key Functions

### `interpolate_interventions(intv_a, intv_b, alpha) → dict`

Linear interpolation between two intervention dicts. `alpha=0` returns A, `alpha=1` returns B.

### `evaluate_intervention(intervention, patient) → dict`

Run simulation and return `final_atp`, `final_het`, `min_atp`, `mean_atp`, `max_het`, `final_copy_number`.

### `run_experiment()`

Execute all three experiments. Total: ~1325 simulations.

---

## Output

- `artifacts/protocol_interpolation.json` — Pairwise, radial, and 3D grid results with detected super-protocols

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
