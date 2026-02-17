# nsga2_optimizer

NSGA-II multi-objective search for intervention protocols.

---

## Overview

`nsga2_optimizer.py` performs evolutionary Pareto search over intervention
space. Objectives include:

- maximize `atp_benefit`
- maximize `het_benefit`
- maximize `crisis_delay_years`
- minimize `total_dose` (via maximize `neg_total_dose`)
- minimize `final_het` (via maximize `neg_final_het`)

---

## CLI

```bash
python nsga2_optimizer.py --patient default --pop-size 48 --generations 40
```

---

## Output

Writes `artifacts/nsga2_<patient>_<date>.json` with:

- Pareto front
- scalar-best point
- run metadata

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
