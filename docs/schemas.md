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
