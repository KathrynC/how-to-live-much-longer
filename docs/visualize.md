# visualize

Matplotlib visualization for mitochondrial aging simulations.

---

## Overview

Headless (`Agg` backend) Matplotlib plotting for publication-quality figures. Generates trajectory subplots, cliff curves, 2D heatmaps, intervention comparison overlays, and TIQM experiment summaries. All functions save to PNG at 150 DPI.

---

## Key Functions

### `plot_trajectory(result, output_path, title)`

9-panel subplot: 8 state variables + heteroplasmy over time. Cliff threshold
line at 0.70 in heteroplasmy panel. Takes `simulate()` output dict.

### `plot_cliff_curve(sweep_result, output_path, cliff_features)`

ATP vs heteroplasmy (the cliff curve). Marks Cramer cliff at 0.70 and optionally the measured cliff threshold from `extract_cliff_features()`. Takes `sweep_heteroplasmy()` output.

### `plot_heatmap(heatmap_result, x_label, y_label, x_key, y_key, output_path, title)`

Generic 2D heatmap of terminal ATP with RdYlGn colormap and cliff line. Works with `heatmap_het_age()` or `heatmap_het_rapamycin()` output.

### `plot_intervention_comparison(results_dict, output_path, variable_idx, variable_name)`

Side-by-side overlay: left panel shows chosen state variable (default: ATP), right panel shows heteroplasmy. Takes a dict mapping intervention name to `simulate()` result.

### `plot_tiqm_summary(experiments, output_path)`

Grouped bar chart of behavior and trajectory resonance scores by clinical scenario. Takes list of TIQM experiment artifact dicts.

---

## Color Scheme

| Variable | Color |
|----------|-------|
| N_healthy | `#2ecc71` (green) |
| N_deletion | `#e74c3c` (red) |
| ATP | `#3498db` (blue) |
| ROS | `#e67e22` (orange) |
| NAD | `#9b59b6` (purple) |
| Senescent_fraction | `#95a5a6` (gray) |
| Membrane_potential | `#1abc9c` (teal) |
| N_point | `#8e44ad` (violet) |

---

## Usage

```python
from simulator import simulate
from visualize import plot_trajectory, plot_intervention_comparison

result = simulate()
plot_trajectory(result, "output/trajectory.png", title="Natural Aging")
```

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
