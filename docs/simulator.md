# simulator

Numpy-only RK4 ODE integrator for mitochondrial aging dynamics.

---

## Overview

Simulates 7 coupled state variables over a configurable time horizon (default 30 years) using 4th-order Runge-Kutta integration. Models the core biology from Cramer (2026): the ROS-damage vicious cycle, the heteroplasmy cliff, age-dependent deletion doubling, and six intervention mechanisms.

**State variables:**

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | `N_healthy` | Healthy mtDNA copies (normalized, 1.0 = full) |
| 1 | `N_damaged` | Damaged mtDNA copies (normalized) |
| 2 | `ATP` | ATP production rate (MU/day) |
| 3 | `ROS` | Reactive oxygen species level |
| 4 | `NAD` | NAD+ availability |
| 5 | `Senescent_fraction` | Fraction of senescent cells |
| 6 | `Membrane_potential` | Mitochondrial membrane potential (normalized) |

---

## Key Functions

### `simulate(intervention, patient, sim_years, tissue_type, stochastic, n_trajectories, noise_scale) → dict`

Run a full simulation. Returns dict with `time`, `states` (n_steps × 7), `heteroplasmy`, `intervention`, `patient`, `tissue_type`.

**Modes:**
- **Default:** Deterministic RK4, single trajectory
- **Tissue-specific:** `tissue_type="brain"` / `"muscle"` / `"cardiac"` applies tissue-specific modifiers from `TISSUE_PROFILES`
- **Stochastic:** `stochastic=True` runs `n_trajectories` Euler-Maruyama paths. Returns `trajectories` array (n_traj × n_steps × 7)
- **Time-varying:** Pass an `InterventionSchedule` instead of a dict for phased or pulsed protocols

### `derivatives(state, time, intervention, patient, tissue_mods) → ndarray`

The ODE right-hand side. Computes derivatives of all 7 state variables given current state and parameters. Can be called directly for custom integration or multi-tissue coupling.

### `initial_state(patient) → ndarray`

Construct the 7-element initial state vector from patient parameters.

---

## Time-Varying Interventions

### `InterventionSchedule`

Holds a sorted list of `(start_year, intervention_dict)` phases. At any time `t`, returns the intervention dict for the most recent phase.

### `phased_schedule(phases) → InterventionSchedule`

Convenience constructor for sequential phases:
```python
schedule = phased_schedule([(0, no_treatment), (10, cocktail)])
```

### `pulsed_schedule(on_intervention, off_intervention, period, duty_cycle) → InterventionSchedule`

Creates a periodic on/off schedule.

---

## Key Dynamics (Post-Falsifier Fixes)

- **Cliff feeds back into replication and apoptosis** (C1) — ATP collapse halts healthy replication
- **Copy number regulated** (C2) — total N_h + N_d homeostatically targets 1.0
- **NAD selectively benefits healthy mitochondria** (C3) — quality control
- **Bistability past cliff** (C4) — damaged replication advantage creates irreversible collapse
- **Yamanaka gated by ATP** (M1) — no energy → no reprogramming
- **CD38 degrades NMN/NR** (C7) — NAD boost gated by CD38 survival factor
- **Transplant displaces damaged copies** (C8) — competitive displacement + addition

---

## Reference

Cramer, J.G. (forthcoming 2026). *How to Live Much Longer: The Mitochondrial DNA Connection*. Springer. ISBN 978-3-032-17740-7.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
