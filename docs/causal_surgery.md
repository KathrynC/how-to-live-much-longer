# causal_surgery

Mid-simulation intervention switching to find the point of no return.

---

## Overview

Adapted from `causal_surgery.py` in the parent Evolutionary-Robotics project (which did mid-simulation brain transplants). Switches between no-treatment and intervention at various time points to answer: "When is it too late to start treatment?" and "How long must treatment run to have lasting effect?"

---

## Experimental Design

| Dimension | Values |
|-----------|--------|
| **Patients** | healthy_30 (30yo, 10% het), moderate_60 (60yo, 40% het), near_cliff_75 (75yo, 60% het) |
| **Interventions** | rapamycin_only, nad_only, full_cocktail, transplant_only |
| **Switch times** | 0, 2, 5, 8, 10, 15, 20, 25 years |
| **Directions** | Forward (no-treatment → intervention), Reverse (intervention → no-treatment) |
| **Total sims** | ~195 (3 × 4 × 8 × 2 + 3 baselines) |

---

## Key Functions

### `simulate_with_surgery(patient, phase1_intervention, phase2_intervention, switch_year, sim_years, dt) → dict`

Run simulation with intervention switching at `switch_year`. Returns output matching `simulate()` format. Uses internal RK4 stepping with `_rk4_step`.

### `run_experiment()`

Execute full sweep. Computes and prints:
- **Point of no return**: Latest switch_year where forward surgery still improves ATP vs baseline
- **Treatment duration threshold**: Minimum years of treatment (reverse surgery) that produces lasting benefit

---

## Analysis Outputs

**Point of no return (forward surgery):** For each patient × intervention, the latest year at which starting treatment still helps. Near-cliff patients have earlier points of no return.

**Treatment duration threshold (reverse surgery):** Minimum treatment duration such that stopping treatment still leaves the patient better than no-treatment baseline.

---

## Output

- `artifacts/causal_surgery.json` — Full results with ATP and het trajectories

---

## Reference

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag.
