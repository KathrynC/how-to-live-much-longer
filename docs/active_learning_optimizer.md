# active_learning_optimizer

Iterative surrogate retraining with uncertainty-aware proposal.

---

## Overview

`active_learning_optimizer.py` alternates:

1. fit surrogate on current evaluated set
2. propose candidates by surrogate utility (`mean + uncertainty`)
3. evaluate proposed candidates with true simulator objective
4. append to training set and repeat

This targets sample efficiency by focusing new evaluations where expected
benefit and uncertainty are both high.

---

## CLI

```bash
python active_learning_optimizer.py --patient default --n-init 24 --rounds 20 --propose-per-round 8
```

---

## Output

Writes `artifacts/active_learning_<patient>_<date>.json` with:

- evaluated records
- round summaries
- best protocol

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
