# perturbation_probing

Intervention fragility mapping via ±1 grid step perturbations.

---

## Overview

Adapted from the parent Evolutionary-Robotics project (which measured "cliffiness" at LLM-generated weight vectors). Perturbs each of 12 parameters ±1 grid step around probe vectors and measures outcome sensitivity. Maps which parameters are fragile (small changes → large outcome shifts) vs robust.

---

## Perturbation Protocol

For each probe vector:
1. Perturb each of 12 parameters up and down by 1 grid step (24 perturbations)
2. Run simulation at each perturbed point
3. Compute sensitivity = |delta_outcome| / |delta_param|
4. Record per-parameter and aggregate fragility

---

## Built-in Probes

| Probe | Description |
|-------|-------------|
| `no_treatment_moderate` | Baseline, 60yo moderate patient |
| `full_cocktail_moderate` | Balanced cocktail, same patient |
| `near_cliff_no_treatment` | 75yo, 65% het, untreated |
| `near_cliff_with_treatment` | Same patient, aggressive treatment |
| `young_biohacker` | 30yo prevention-oriented |
| `yamanaka_aggressive` | High Yamanaka (0.75), moderate patient |
| `high_inflammation` | High inflammation scenario |

Can also load vectors from OEIS or character experiment results.

---

## Key Functions

### `run_experiment()`

Execute perturbation sweep across all probe vectors. Reports:
- Per-parameter sensitivity rankings
- Most/least fragile probe points
- Cliff-adjacent vs far-from-cliff fragility comparison

---

## Scale

~50 probe points × 25 sims each = ~1250 simulations. Estimated time: ~5 minutes (pure simulation).

---

## Output

- `artifacts/perturbation_probing.json` — Per-probe sensitivity maps, fragility rankings

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
