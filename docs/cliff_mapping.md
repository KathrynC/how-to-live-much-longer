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

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag. Ch. V.K p.66.
