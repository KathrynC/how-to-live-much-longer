# bayesian_optimizer

Constrained Bayesian-style search (surrogate + UCB acquisition).

---

## Overview

`bayesian_optimizer.py` trains a lightweight surrogate from evaluated points and
selects new candidates by upper-confidence-bound (UCB) acquisition:

`acquisition = predicted_mean + kappa * predicted_std`

Top acquisition candidates are then evaluated with the true simulator-backed
objective.

---

## CLI

```bash
python bayesian_optimizer.py --patient default --metric combined
```

Key options:

- `--n-init`
- `--iterations`
- `--candidate-pool`
- `--kappa`

---

## Output

Writes `artifacts/bayes_optimizer_<patient>_<date>.json`.

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
