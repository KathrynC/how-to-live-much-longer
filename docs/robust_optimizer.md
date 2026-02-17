# robust_optimizer

Robust protocol search under patient-parameter uncertainty.

---

## Overview

`robust_optimizer.py` evaluates each candidate across an ensemble of perturbed
patient profiles and optimizes a risk-adjusted score:

`robust_fitness = mean_fitness - std_fitness`

This seeks protocols that are both effective and stable under uncertainty.

---

## CLI

```bash
python robust_optimizer.py --patient default --budget 250 --ensemble-size 24
```

---

## Output

Writes `artifacts/robust_optimizer_<patient>_<date>.json` with:

- best robust protocol
- robust statistics (`mean`, `std`, `worst`, `best`)
- trajectory history

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
