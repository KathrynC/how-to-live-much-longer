# analytics

4-pillar health analytics for mitochondrial aging simulations.

---

## Overview

Mirrors the Beer framework from the parent Evolutionary-Robotics project, adapted for cellular energetics. Computes comprehensive metrics across four clinical pillars from simulation output.

---

## Pillars

### Pillar 1: Energy — `compute_energy(result)`

| Metric | Description |
|--------|-------------|
| `atp_initial`, `atp_final` | Start and end ATP levels |
| `atp_min`, `atp_max`, `atp_mean` | Trajectory statistics |
| `atp_cv` | Coefficient of variation (stability) |
| `atp_slope` | Linear trend (MU/day/year) |
| `atp_terminal_slope` | Slope over final 10% of trajectory |
| `time_to_crisis` | Years until ATP drops below crisis threshold |
| `reserve_ratio` | min/max ATP ratio |

### Pillar 2: Damage — `compute_damage(result)`

| Metric | Description |
|--------|-------------|
| `het_initial`, `het_final` | Start and end heteroplasmy |
| `cliff_distance` | Distance from heteroplasmy cliff (0.70) |
| `time_to_cliff` | Years until het exceeds cliff threshold |
| `het_acceleration` | Second derivative of heteroplasmy trajectory |
| `fraction_above_cliff` | Fraction of simulation time above cliff |

### Pillar 3: Dynamics — `compute_dynamics(result)`

| Metric | Description |
|--------|-------------|
| `ros_dominant_freq` | FFT dominant frequency of ROS oscillations |
| `ros_dominant_amplitude` | FFT amplitude at dominant frequency |
| `membrane_potential_cv` | Membrane potential stability |
| `membrane_potential_slope` | Membrane potential trend |
| `nad_slope` | NAD+ trajectory trend |
| `ros_het_correlation` | ROS-heteroplasmy coupling strength |
| `ros_atp_correlation` | ROS-ATP coupling strength |
| `senescence_rate` | Rate of senescent cell accumulation |

### Pillar 4: Intervention — `compute_intervention(result, baseline)`

| Metric | Description |
|--------|-------------|
| `atp_benefit_terminal` | Final ATP gain vs no-treatment baseline |
| `atp_benefit_mean` | Mean ATP gain over trajectory |
| `het_benefit` | Heteroplasmy reduction vs baseline |
| `energy_cost` | Estimated intervention energy cost (Yamanaka-dominated) |
| `benefit_cost_ratio` | ATP benefit per unit energy cost |
| `crisis_delay_years` | Additional years before energy crisis |

---

## Main Entry Point

### `compute_all(result, baseline=None) → dict`

Compute all 4 pillars. If `baseline` is provided (no-treatment simulation), also computes intervention pillar.

```python
from simulator import simulate
from analytics import compute_all
result = simulate(intervention=cocktail, patient=patient)
baseline = simulate(patient=patient)  # no treatment
analytics = compute_all(result, baseline)
```

---

## Utilities

### `NumpyEncoder`

Custom JSON encoder that handles numpy types. All floats rounded to 6 decimal places.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
