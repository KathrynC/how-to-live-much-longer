# cliff_mapping

Heteroplasmy threshold mapping and cliff characterization.

---

## Overview

Maps the nonlinear relationship between heteroplasmy and ATP collapse — the "cliff" from Cramer (2026, Ch. V.K p.66). Provides 1D sweeps, bisection search for cliff edge, intervention-dependent cliff shift analysis, 2D heatmaps, and cliff feature extraction.

---

## Key Functions

### `sweep_heteroplasmy(n_points, sim_years, intervention, patient_base) → dict`

Sweep baseline heteroplasmy from 0 to 0.95, recording terminal ATP at each point. Returns `{"het_values": array, "terminal_atp": array}`.

### `find_cliff_edge(threshold_atp_frac, tol, max_iter, intervention, patient_base, sim_years) → dict`

Bisection search for the heteroplasmy value where ATP drops below a fraction of maximum. Default: 50% of reference ATP. Precision to 0.001 (3 decimal places).

Returns: `cliff_edge_het`, `cliff_edge_atp`, `reference_atp`, `precision`, `iterations`, `history`.

### `cliff_shift_analysis(patient_base, sim_years) → dict`

Compare cliff edge location under 5 interventions: no intervention, rapamycin-only, NAD-only, full cocktail, Yamanaka. Returns mapping of intervention name to cliff edge result.

### `heatmap_het_age(n_het, n_age, intervention, sim_years) → dict`

2D heatmap of terminal ATP across heteroplasmy × starting age. NAD level adjusted for age. Returns `{"het_axis", "age_axis", "atp_grid"}`.

### `heatmap_het_rapamycin(n_het, n_rapa, patient_base, sim_years) → dict`

2D heatmap of terminal ATP across heteroplasmy × rapamycin dose. Returns `{"het_axis", "rapa_axis", "atp_grid"}`.

### `extract_cliff_features(sweep_result) → dict`

Extract cliff geometry from a 1D sweep:

| Feature | Description |
|---------|-------------|
| `threshold` | Heteroplasmy at maximum ATP decline rate |
| `sharpness` | Maximum \|dATP/dHet\| (steepness of cliff) |
| `width` | Heteroplasmy range over which 80% of ATP drop occurs |
| `asymmetry` | Ratio of pre-cliff slope to post-cliff slope |

---

## Usage

```python
from cliff_mapping import sweep_heteroplasmy, find_cliff_edge, extract_cliff_features

sweep = sweep_heteroplasmy(n_points=50, sim_years=10)
features = extract_cliff_features(sweep)
edge = find_cliff_edge(sim_years=10)
shifts = cliff_shift_analysis(sim_years=10)
```

---

## Reference

Cramer, J.G. (forthcoming 2026). *How to Live Much Longer: The Mitochondrial DNA Connection*. Springer. ISBN 978-3-032-17740-7. Ch. V.K p.66.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
