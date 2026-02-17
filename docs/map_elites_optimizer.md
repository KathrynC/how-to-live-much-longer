# map_elites_optimizer

Quality-diversity search via MAP-Elites.

---

## Overview

`map_elites_optimizer.py` builds an archive of diverse high-quality protocols
over a behavior descriptor grid:

- descriptor 1: `final_atp`
- descriptor 2: `final_het`

Each grid cell stores the best protocol found for that niche.

---

## CLI

```bash
python map_elites_optimizer.py --patient default --budget 600 --bins-atp 14 --bins-het 14
```

---

## Output

Writes `artifacts/map_elites_<patient>_<date>.json` with:

- archive coverage
- elite count
- best global fitness elite
- full elite table

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
