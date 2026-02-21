# ten_types_audit

Deterministic Ten Types of Innovation audit for this repository archive.

---

## Purpose and Scope

`ten_types_audit.py` evaluates the repository as a **whole platform** using the Ten Types framework and emits repeatable JSON/Markdown artifacts.

Scope in v1 is fixed to:

- `whole_platform`

No external data sources or LLM judgments are used.

---

## Ten Types and Categories

The audit preserves canonical type IDs while adapting display labels for this context.

| Category | Type IDs | Display Labels |
|---|---|---|
| Configuration | `profit_model`, `network`, `structure`, `process` | Value Model, Network, Structure, Process |
| Offering | `product_performance`, `product_system` | Product Performance, Product System |
| Experience | `service`, `channel`, `brand`, `customer_engagement` | Service, Channel, Brand, Customer Engagement |

`profit_model` is shown as **Value Model** (non-commercial adaptation), while `canonical_label` remains `Profit Model`.

---

## Strict 3-Level Rubric

Maturity levels:

- `no_activity` = 0
- `me_too` = 1
- `differentiating` = 2

Scoring logic per type:

- `differentiating`: at least one differentiating probe hit and sufficient total evidence
- `me_too`: some evidence present but differentiating threshold not met
- `no_activity`: no meaningful evidence

Strict counting rule:

- only `differentiating` counts as true innovation in coverage metrics

---

## Value Model Adaptation

This repository is research infrastructure, not a commercial product line. Therefore, the Profit Model type is interpreted as **Value Model**:

- sustainability of scientific utility
- traceable evidence quality
- reproducible and reusable outputs

This keeps the Ten Types structure intact while avoiding a forced revenue lens.

---

## Usage

```bash
python ten_types_audit.py
```

Default outputs:

- `artifacts/ten_types_innovation_audit.json`
- `artifacts/ten_types_innovation_audit.md`

CLI options:

```bash
python ten_types_audit.py \
  --scope whole_platform \
  --format json|markdown|both \
  --out-json <path> \
  --out-md <path> \
  --as-of YYYY-MM-DD
```

Python API:

```python
from ten_types_audit import run_audit

report = run_audit(
    scope="whole_platform",
    scoring_model="strict_3_level",
    as_of="2026-02-18",
)
```

---

## Output Schema

Top-level keys:

- `timestamp`
- `as_of`
- `scope`
- `scoring_model`
- `type_assessments`
- `summary`
- `category_balance`
- `priority_gaps`
- `evidence_index`

`type_assessments` includes per-type maturity, evidence refs, and strict innovation flag.

---

## Interpretation Guidance

Use the report in three passes:

1. **Strengths**: types scoring `differentiating` and their evidence clusters
2. **Blind spots**: `me_too` and `no_activity` types in `priority_gaps`
3. **Portfolio balance**: `category_balance` distribution across Configuration, Offering, Experience

The goal is not maximal counts; the goal is defensible differentiation across the platform.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
