# layer_viz

Population-level visualization tools for layer comparisons.

---

## Overview

`layer_viz.py` generates cohort-level plots comparing the core model against
additional layers:

- Agroecology-inspired disturbance stack
- Grief disturbance bridge
- Combined agro + grief

It focuses on two outputs:

1. **Layer delta forest plot** (per-patient effect size distribution)
2. **Outcome shift matrix** (core category -> layered category transitions)

Outputs are saved to `output/layers/`.

---

## Key Functions

### `simulate_layer_cohort(n_patients, seed) -> dict`

Runs the cohort under four conditions:

- `core`
- `agro`
- `grief`
- `agro_grief`

Returns per-patient final ATP, total/deletion heteroplasmy, category label, and
cliff/crisis flags for each condition.

### `plot_layer_delta_forest(cohort, filename) -> str`

Forest-style summary of per-patient deltas (`layer - core`) for:

- `final_atp`
- `final_het`
- `final_del_het`

Each layer shows median delta (dot) and 10th-90th percentile band.

### `plot_outcome_shift_matrix(cohort, filename) -> str`

Three transition heatmaps (`core -> agro`, `core -> grief`,
`core -> agro_grief`) using ATP-based outcome categories:

- `collapsed` (`ATP < 0.2`)
- `declining` (`0.2 <= ATP < 0.5`)
- `stable` (`0.5 <= ATP < 0.8`)
- `healthy` (`ATP >= 0.8`)

Rows are normalized so each row sums to 1.0.

### `generate_layer_effects_dashboard(n_patients, seed) -> list[str]`

Convenience runner for the full pipeline: cohort simulation + both plots.

---

## CLI

```bash
python layer_viz.py --n-patients 1000 --seed 2026
```

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
