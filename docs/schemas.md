# schemas

Pydantic validation models for LLM-generated parameter vectors.

---

## Overview

Provides strict per-field range constraints derived from `constants.py`. Used as a validation layer between JSON parsing and grid snapping in `llm_common.py`.

---

## Models

### `InterventionVector`

6 intervention parameters, each constrained to [0.0, 1.0]:
`rapamycin_dose`, `nad_supplement`, `senolytic_dose`, `yamanaka_intensity`, `transplant_rate`, `exercise_level`

### `PatientProfile`

6 patient parameters with per-field ranges:
- `baseline_age`: [20, 90]
- `baseline_heteroplasmy`: [0.0, 0.95]
- `baseline_nad_level`: [0.2, 1.0]
- `genetic_vulnerability`: [0.5, 2.0]
- `metabolic_demand`: [0.5, 2.0]
- `inflammation_level`: [0.0, 1.0]

### `FullProtocol`

Combined 12D vector. All fields optional (LLMs may omit some). Used for flexible validation that doesn't reject partial outputs.

---

## Usage

```python
from schemas import FullProtocol, InterventionVector
from llm_common import validate_llm_output

# Direct validation
protocol = FullProtocol(rapamycin_dose=0.5, baseline_age=70)

# In pipeline (via llm_common)
snapped, warnings = validate_llm_output(raw_dict)
```

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
