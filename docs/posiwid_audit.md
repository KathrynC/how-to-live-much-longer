# posiwid_audit

POSIWID alignment auditor: "The Purpose Of a System Is What It Does."

---

## Overview

Measures the gap between what an LLM *says* it intends and what the simulation *actually produces*. Stafford Beer's POSIWID principle (1974) — "The Purpose Of a System Is What It Does" — appears in Zimmerman's thesis (§3.5.2) as a framework for analyzing objective function misalignment: the tokenizer's purpose (compress orthographically) is not aligned with the model's purpose (learn semantics).

This script applies POSIWID to the LLM→simulator pipeline: when an LLM generates parameters intending to reduce heteroplasmy, and the simulation shows heteroplasmy increasing, that gap between intended and actual is quantifiable.

---

## Pipeline

For each clinical scenario × model:

```
1. INTENTION QUERY    "What outcome do you intend for this patient?"
   │                  → expected het change, expected ATP change, confidence
   │
2. PROTOCOL QUERY     "Generate a 12D intervention+patient vector"
   │                  → parsed, validated, snapped to grid
   │
3. SIMULATE           Run ODE with generated parameters
   │                  + baseline (no treatment) for comparison
   │
4. SCORE ALIGNMENT    Compare intention vs actual on 4 axes
```

**Scale:** 10 scenarios × 2 models × 2 queries = 40 LLM calls + 20 simulations. ~15-20 min with Ollama.

---

## Alignment Scoring

### `score_alignment(intention, actual_het_change, actual_atp_change, actual_final_het, actual_final_atp) → dict`

Four scoring components:

| Component | Type | What It Measures |
|-----------|------|-----------------|
| `het_direction` | Binary (0/1) | Did heteroplasmy move in the expected direction? |
| `het_magnitude` | Continuous (0–1) | How close was predicted final het to actual? |
| `atp_direction` | Binary (0/1) | Did ATP move in the expected direction? |
| `atp_magnitude` | Continuous (0–1) | How close was predicted final ATP to actual? |
| `overall` | Mean of above 4 | Overall alignment score |

**Calibration:**
- Magnitude scaling: ×2, so an error of 0.5 (the full healthy-to-cliff range for heteroplasmy) scores zero
- "Expected no change" scaling: ×5, so an unexpected change of 0.2 over 30 years scores zero (clinically significant)

### What Low Alignment Reveals

Low alignment scores indicate the LLM generates "plausible-sounding" parameters without understanding the ODE dynamics. Common failure modes:

- **Direction errors:** LLM says "heteroplasmy will decrease" but prescribes parameters that increase it
- **Magnitude errors:** LLM correctly predicts direction but dramatically over/underestimates the effect
- **Confidence-alignment mismatch:** High-confidence predictions with low actual alignment

---

## Usage

```python
from posiwid_audit import run_audit

# Full audit (requires Ollama)
result = run_audit()

# With specific models/scenarios
from llm_common import MODELS
from constants import CLINICAL_SEEDS
result = run_audit(models=MODELS[:1], seeds=CLINICAL_SEEDS[:3])
```

```bash
python posiwid_audit.py
# Outputs: artifacts/posiwid_audit.json
```

---

## Output

```json
{
  "n_trials": 20,
  "summary": {
    "mean_overall": 0.45,
    "het_direction_accuracy": 0.60,
    "atp_direction_accuracy": 0.55
  },
  "results": [
    {
      "seed_id": "S01",
      "model": "qwen3-coder:30b",
      "intention": {"expected_het_change": -0.1, ...},
      "alignment": {"overall": 0.52, "het_direction": 1.0, ...}
    }
  ]
}
```

---

## Reference

- Beer, Stafford (1974). "Designing Freedom." CBC Massey Lectures.
- Zimmerman, J.W. (2025). PhD dissertation, University of Vermont. §3.5.2.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
