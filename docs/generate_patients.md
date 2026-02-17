# generate_patients

Patient population generator with biologically plausible correlations.

---

## Overview

Generates two complementary patient populations: a normal population (100 patients) with biology-informed correlation structure, and an edge-case population (82 patients) for simulator robustness testing. Both include comprehensive evaluation metrics.

---

## Key Functions

### `generate_patients(n, seed) → list[dict]`

Generate `n` patients with biologically plausible correlations. Age drives heteroplasmy (nonlinear), NAD decline, and inflammation. Genetic vulnerability and metabolic demand are sampled independently. All values snapped to grid.

**Correlation structure:**

| Pair | Expected | Biology |
|------|----------|---------|
| age ↔ het | positive (r ~ +0.80) | Older → more mtDNA damage |
| age ↔ NAD | negative (r ~ -0.92) | Older → lower NAD+ (Ca16) |
| age ↔ inflammation | positive (r ~ +0.66) | Inflammaging |
| het ↔ NAD | negative (r ~ -0.70) | Damage → worse NAD state |
| genetic_vuln ↔ metabolic_demand | ~zero (r ~ +0.05) | Independent |

### `generate_edge_patients() → list[dict]`

Generate ~82 edge-case patients organized into 8 categories:

| Category | N | Purpose |
|----------|---|---------|
| `single_extreme` | 12 | One param at min or max, rest default |
| `corner` | 10 | Multiple params at extremes simultaneously |
| `cliff_boundary` | 14 | Het sweep across the 0.70 cliff threshold |
| `contradictory` | 12 | Biologically unlikely but must not crash |
| `max_stress` | 10 | Worst-case parameter combinations |
| `tissue_vuln_cross` | 8 | 2×2×2 factorial: demand × vulnerability × age |
| `age_at_cliff` | 8 | Every decade at het=0.70 |
| `near_limits` | 8 | Values at or near hard parameter boundaries |

Each patient includes `_label`, `_category`, and `_id` metadata.

### `evaluate_population(patients) → dict`

Evaluate normal population quality: distribution statistics, grid coverage, correlation plausibility (5 expected biological correlations), clinical plausibility checks, and simulation outcome diversity. Overall quality score composed of 4 sub-scores.

### `evaluate_edge_population(patients) → dict`

Evaluate edge-case population for simulator robustness: NaN/Inf detection, negative state detection, heteroplasmy out-of-range detection, per-category outcome summary. Overall robustness score composed of 3 sub-scores.

---

## CLI

```bash
python generate_patients.py           # Normal population (100 patients)
python generate_patients.py --edge    # Edge-case population (82 patients)
python generate_patients.py --both    # Both populations
```

---

## Output

- `artifacts/sample_patients_100.json` — 100 normal patients + evaluation
- `artifacts/sample_patients_edge.json` — 82 edge-case patients + evaluation

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
