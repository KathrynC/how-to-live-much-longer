# resilience_viz

Matplotlib visualizations for mitochondrial resilience analysis.

---

## Overview

Generates plots showing disturbance response, recovery dynamics, and resilience metric comparisons. All plots use the Agg backend (headless, non-interactive) and save to `output/resilience/`. Provides four individual plot functions, a `generate_all_plots()` suite that produces the complete visualization set, and a CLI for single-disturbance or full-suite generation.

---

## Key Functions

### `plot_shock_response(shocked_result, baseline_result, title, filename) -> str`

Four-panel plot of state trajectories with shock window highlighted. Panels show ATP, heteroplasmy, ROS, and membrane potential for both baseline (thin, transparent) and shocked (bold) trajectories. Shock windows are shaded red with dashed onset markers. The heteroplasmy panel includes a dashed cliff line at 0.70.

**Args:**
- `shocked_result` — Dict from `simulate_with_disturbances()`
- `baseline_result` — Dict from `simulate()`
- `title` — Plot title (default: `"Shock Response"`)
- `filename` — Output filename (default: `"shock_response.png"`)

**Returns:** Path to saved plot (150 dpi).

### `plot_resilience_comparison(results, baseline_result, labels, state_idx, ylabel, title, filename) -> str`

Overlay plot comparing multiple shocked trajectories against baseline for a single state variable. Baseline rendered in black; each shocked trajectory in a distinct color from the Set1 colormap. Shock windows shaded per trajectory.

**Args:**
- `results` — List of dicts from `simulate_with_disturbances()`
- `baseline_result` — Dict from `simulate()`
- `labels` — Label for each result
- `state_idx` — State variable index to plot (default: 2 = ATP)
- `ylabel` — Y-axis label (default: `"ATP (MU/day)"`)
- `title` — Plot title (default: `"Resilience Comparison"`)
- `filename` — Output filename (default: `"resilience_comparison.png"`)

**Returns:** Path to saved plot (150 dpi).

### `plot_resilience_summary(sweep_results, title, filename) -> str`

Four-panel bar chart of resilience metrics across shock magnitudes from a `compute_resilience_sweep()` result:

| Panel | Metric | Color Coding |
|-------|--------|-------------|
| Top-left | Composite resilience score (0-1) | Steel blue |
| Top-right | Relative peak deviation (resistance) | Coral |
| Bottom-left | Recovery time (years, capped at 30) | Green (<5yr), orange (5-15yr), red (>15yr) |
| Bottom-right | Regime retention (binary) | Green (retained), red (shifted) |

**Args:**
- `sweep_results` — List of dicts from `compute_resilience_sweep()`
- `title` — Plot title (default: `"Resilience vs Shock Magnitude"`)
- `filename` — Output filename (default: `"resilience_summary.png"`)

**Returns:** Path to saved plot (150 dpi).

### `plot_recovery_landscape(disturbance_class, magnitudes, start_years, intervention, patient, filename) -> str`

2D heatmap of recovery time across shock magnitude (y-axis) vs shock timing (x-axis). Uses the `RdYlGn_r` colormap (green = fast recovery, red = slow/never). Recovery times capped at 30 years for visualization. Runs a full simulation grid internally.

**Args:**
- `disturbance_class` — `Disturbance` subclass to test
- `magnitudes` — Magnitude values (default: 0.1 to 1.0 in steps of 0.1)
- `start_years` — Shock start times (default: `[2, 5, 8, 11, 14, 17, 20, 24]`)
- `intervention` — Intervention dict (default: no treatment)
- `patient` — Patient dict (default: typical 70-year-old)
- `filename` — Output filename (default: `"recovery_landscape.png"`)

**Returns:** Path to saved plot (150 dpi).

### `generate_all_plots(intervention, patient) -> list[str]`

Generate the complete resilience visualization suite. Produces 9 plots:

| # | Plot | Filename | Description |
|---|------|----------|-------------|
| 1 | Radiation shock response | `shock_radiation.png` | 4-panel response (mag=0.8) |
| 2 | Toxin shock response | `shock_toxin.png` | 4-panel response (mag=0.6) |
| 3 | Chemo shock response | `shock_chemo.png` | 4-panel response (mag=0.8) |
| 4 | Inflammation shock response | `shock_inflammation.png` | 4-panel response (mag=0.7) |
| 5 | ATP comparison | `disturbance_comparison_atp.png` | All 4 disturbances overlaid on ATP |
| 6 | Heteroplasmy comparison | `disturbance_comparison_het.png` | All 4 disturbances overlaid on het with cliff line |
| 7 | Radiation magnitude sweep | `resilience_sweep_radiation.png` | 4-panel bar chart (7 magnitudes) |
| 8 | Chemo magnitude sweep | `resilience_sweep_chemo.png` | 4-panel bar chart (7 magnitudes) |
| 9 | Radiation recovery landscape | `recovery_landscape_radiation.png` | 2D heatmap (6 magnitudes x 6 start times) |
| 10 | Chemo recovery landscape | `recovery_landscape_chemo.png` | 2D heatmap (6 magnitudes x 6 start times) |

All disturbances applied at `start_year=10.0`. Magnitude sweep uses `[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]`. Recovery landscapes use magnitudes `[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` at start years `[2, 5, 10, 15, 20, 25]`.

**Returns:** List of paths to all generated plots.

---

## CLI Interface

```bash
# Full visualization suite (all 10 plots)
python resilience_viz.py

# Single disturbance plot
python resilience_viz.py --disturbance radiation --magnitude 0.8
python resilience_viz.py --disturbance chemo --magnitude 0.6
python resilience_viz.py --disturbance toxin --magnitude 0.5
python resilience_viz.py --disturbance inflammation --magnitude 0.7
```

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--disturbance` | `str` | `None` | Single disturbance to plot: `radiation`, `toxin`, `chemo`, `inflammation` |
| `--magnitude` | `float` | `0.8` | Disturbance magnitude for single-plot mode |

When `--disturbance` is provided, generates a single shock response plot. When omitted, runs `generate_all_plots()` for the full suite.

---

## Output

All plots are saved to `output/resilience/` relative to the project root. The directory is created automatically if it does not exist. All figures are saved at 150 dpi with tight bounding boxes.

---

## Usage

```python
from resilience_viz import (
    plot_shock_response, plot_resilience_comparison,
    plot_resilience_summary, plot_recovery_landscape,
    generate_all_plots,
)
from simulator import simulate
from disturbances import simulate_with_disturbances, IonizingRadiation
from resilience_metrics import compute_resilience_sweep

# Single shock response plot
baseline = simulate()
shock = IonizingRadiation(start_year=10.0, magnitude=0.8)
shocked = simulate_with_disturbances(disturbances=[shock])
path = plot_shock_response(shocked, baseline,
                           title="Radiation Response",
                           filename="my_radiation.png")

# Recovery landscape heatmap
path = plot_recovery_landscape(IonizingRadiation,
                               magnitudes=[0.2, 0.4, 0.6, 0.8, 1.0],
                               start_years=[5.0, 10.0, 15.0, 20.0])

# Full suite
paths = generate_all_plots()
print(f"Generated {len(paths)} plots")
```

---

## Reference

Cramer, J.G. (forthcoming 2026). *How to Live Much Longer: The Mitochondrial DNA Connection*. Springer. ISBN 978-3-032-17740-7.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
