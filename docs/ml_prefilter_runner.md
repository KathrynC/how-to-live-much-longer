# ml_prefilter_runner

Add-on ML prefilter pipeline for candidate prioritization.

---

## Overview

`ml_prefilter_runner.py` runs an additive surrogate-first workflow:

1. Train surrogate on simulator-labeled samples.
2. Build candidate pool (random + optional EA-seed perturbations).
3. Rank by surrogate prediction.
4. Re-evaluate top-K with true simulator objective.
5. Compare against random-control candidates from same pool.

No existing analytics equations or simulator mechanisms are modified.

---

## CLI

```bash
python ml_prefilter_runner.py --patient default
python ml_prefilter_runner.py \
  --patient near_cliff_80 \
  --seed-artifact artifacts/ea_protocol_transfer_2026-02-17.json \
  --profile near_cliff_80 \
  --train-samples 200 \
  --pool-size 2000 \
  --top-k 32
```

---

## Output Artifact

Writes `artifacts/ml_prefilter_<patient>_<date>.json` (or `--out`) including:

- training and pool settings
- top-K predicted and true-evaluated candidates
- random-control true-evaluated candidates
- mean true-fitness lift of top-K vs random baseline

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
