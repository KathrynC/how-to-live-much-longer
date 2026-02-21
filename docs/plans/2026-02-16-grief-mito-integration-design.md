# Grief-to-Mitochondrial Aging Integration Design

## Goal

Connect the grief biological stress simulator to the mitochondrial aging simulator so we can answer: **"Can interventions during grief protect mitochondria?"**

## Architecture

The grief simulator (11-variable ODE, 10-year daily timestep) produces time-varying biological stress signals. These feed into the mitochondrial aging simulator (7-variable ODE, 30-year timestep) through a new `GriefDisturbance` class that implements the existing `Disturbance` ABC from `disturbances.py`.

```
grief-simulator/                   how-to-live-much-longer/
  simulator.py ──run──►              grief_bridge.py (NEW)
  constants.py                         ├── GriefDisturbance(Disturbance)
                                       ├── grief_trajectory()
                                       └── grief_scenarios()

                                     grief_mito_simulator.py (NEW)
                                       └── GriefMitoSimulator (Zimmerman 28D)

                                     grief_mito_viz.py (NEW)
                                       ├── plot_grief_mito_trajectory()
                                       ├── plot_intervention_comparison()
                                       └── plot_all_grief_scenarios()

                                     grief_mito_scenarios.py (NEW)
                                       ├── GRIEF_STRESS_SCENARIOS
                                       └── run_grief_resilience_analysis()

                                     disturbances.py (UNCHANGED)
                                       └── simulate_with_disturbances()
```

## Data Flow

1. User specifies grief params (14D) and mito params (12D)
2. `grief_trajectory()` runs the grief ODE (3,650 daily steps, ~5ms)
3. `GriefDisturbance` constructed with the 5 relevant grief curves
4. `simulate_with_disturbances()` runs the mito ODE, calling `GriefDisturbance.modify_params()` every timestep
5. Grief signals perturb mito parameters in real time
6. Result: mito trajectory reflecting grief's biological impact

## The Five Mappings

| Grief Signal | Mito Target | Channel | Coefficient | Rationale |
|---|---|---|---|---|
| `Infl` (inflammation) | `inflammation_level` | `modify_params` additive | `0.4 * grief_infl` | Direct: same IL-6/CRP pathway. Comparable to InflammationBurst (0.5). |
| `Cort` (cortisol) | `metabolic_demand` | `modify_params` additive | `0.2 * grief_cort` | Chronic cortisol elevation increases basal metabolic rate. Same as ToxinExposure (0.2). |
| `SNS` (sympathetic) | ROS state variable | `modify_state` additive | `0.05 * grief_sns` per step | Catecholamine metabolism + ETC stress. Lower than acute bursts (radiation: 0.15) — this is chronic. |
| `1 - Slp` (sleep disruption) | `inflammation_level` | `modify_params` additive | `0.15 * (1 - grief_slp)` | Poor sleep impairs autophagy/mitophagy → elevated inflammatory markers. Stacks with direct Infl mapping. |
| `d(CVD_risk)/dt` (damage rate) | `genetic_vulnerability` | `modify_params` multiplicative | `1.0 + 0.3 * cvd_rate` | Cardiovascular stress increases oxidative mtDNA damage. Comparable to IonizingRadiation (0.5). |

All signals interpolated from grief trajectory (daily, 10 years) to mito timestep (0.01 years ~ 3.65 days).

## Time Alignment

- Grief sim: 10 years, daily timestep (3,650 points)
- Mito sim: 30 years, dt=0.01 years (3,000 points)
- `GriefDisturbance` active window: configurable `start_year` and duration (default: start at year 0 of mito sim, duration = 10 years)
- After grief window ends, no further perturbation (grief has resolved or plateaued)
- Grief curves interpolated via `np.interp` to match mito time points within the active window

## Deliverables

### 1. `grief_bridge.py` (~150 lines)
- `grief_trajectory(grief_patient, grief_intervention)` — runs grief sim, returns dict of 5 numpy arrays
- `GriefDisturbance(Disturbance)` — the disturbance class
  - Constructor takes grief_patient, grief_intervention, start_year, magnitude
  - `modify_state()`: daily ROS increment from SNS
  - `modify_params()`: inflammation, metabolic demand, genetic vulnerability from Infl/Cort/Slp/CVD_risk
- `grief_scenarios()` — returns 8 clinical seeds as GriefDisturbance objects, with and without interventions

### 2. `grief_mito_simulator.py` (~80 lines)
- `GriefMitoSimulator` — Zimmerman protocol adapter for combined system
- 28D input: 14 grief params + 14 mito params (with sensible split convention)
- `param_spec()` returns all 28 params with bounds
- `run()` chains grief sim → GriefDisturbance → mito sim → combined analytics
- Compatible with zimmerman-toolkit and cramer-toolkit

### 3. `grief_mito_viz.py` (~200 lines)
- `plot_grief_mito_trajectory()` — 2x4 panel: grief panels (left) + mito panels (right), mapping channels highlighted
- `plot_intervention_comparison()` — same grief scenario with/without interventions, overlaid mito outcomes
- `plot_all_grief_scenarios()` — all 8 clinical seeds through mito sim, comparison overlay

### 4. `grief_mito_scenarios.py` (~100 lines)
- `GRIEF_STRESS_SCENARIOS` — grief-derived scenario bank for cramer-toolkit (8 scenarios x 2 intervention levels = 16)
- `run_grief_resilience_analysis()` — convenience wrapper combining grief scenarios with cramer resilience analysis
- `GRIEF_PROTOCOLS` — maps grief intervention profiles to named protocols

### 5. `tests/test_grief_bridge.py` (~150 lines)
- GriefDisturbance satisfies Disturbance ABC
- Grief trajectory returns correct shapes
- Bereaved > non-bereaved final heteroplasmy
- Interventions reduce mitochondrial impact
- Composes with existing disturbances (grief + radiation)
- No NaN/negative states
- GriefMitoSimulator satisfies Zimmerman protocol
- Scenarios produce valid disturbance objects

## Dependencies

- `grief_bridge.py` imports from `~/grief-simulator/` via `sys.path.insert`
- Both projects must be present on disk (sibling directories under `~`)
- No pip install, no package restructuring
- No changes to any existing files in either project

## Key Experiment

```python
from grief_bridge import GriefDisturbance
from disturbances import simulate_with_disturbances
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

# Same person, same loss — with and without grief interventions
no_help = GriefDisturbance(
    grief_patient={"B": 0.9, "M": 0.8, "age": 65.0, "E_ctx": 0.2},
    grief_intervention=None,  # defaults: no special support
)
with_help = GriefDisturbance(
    grief_patient={"B": 0.9, "M": 0.8, "age": 65.0, "E_ctx": 0.2},
    grief_intervention={"act_int": 0.7, "slp_int": 0.8, "soc_int": 0.7},
)

result_no = simulate_with_disturbances(disturbances=[no_help])
result_yes = simulate_with_disturbances(disturbances=[with_help])

# Compare: how much mitochondrial aging did the interventions prevent?
print(f"Without help: final het = {result_no['heteroplasmy'][-1]:.4f}")
print(f"With help:    final het = {result_yes['heteroplasmy'][-1]:.4f}")
```

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
