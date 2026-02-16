# constants

Central configuration: biological constants, 12D parameter space, Ollama model config.

---

## Overview

All simulation parameters and biological constants derived from Cramer (2026), with chapter/page citations for each value. Defines the 12-dimensional parameter space (6 intervention + 6 patient), discrete grid points for LLM output snapping, type aliases, and 10 clinical scenario seeds.

---

## Biological Constants

| Constant | Value | Source |
|----------|-------|--------|
| `HETEROPLASMY_CLIFF` | 0.70 | Rossignol 2003; Cramer Ch. V.K p.66 |
| `CLIFF_STEEPNESS` | 15.0 | Simulation calibration |
| `DOUBLING_TIME_YOUNG` | 11.8 yr | Appendix 2 p.155, Va23 data |
| `DOUBLING_TIME_OLD` | 3.06 yr | Appendix 2 p.155, Va23 data |
| `AGE_TRANSITION` | 65 | Appendix 2 p.155 (dynamically coupled to ATP/mitophagy via C10) |
| `BASELINE_ATP` | 1.0 MU/day | Ch. VIII.A Table 3 p.100 |
| `BASELINE_ROS` | 0.1 | Ch. IV.B p.53, Ch. II.H p.14 |
| `NAD_DECLINE_RATE` | 0.01/yr | Ch. VI.A.3 pp.72-73, Ca16 |
| `SENESCENCE_RATE` | 0.005/yr | Ch. VII.A pp.89-92 |
| `DAMAGED_REPLICATION_ADVANTAGE` | 1.05× | Appendix 2 pp.154-155 (book: ≥1.21) |
| `YAMANAKA_ENERGY_COST` | 3-5 MU | Ch. VIII.A Table 3 p.100 |
| `TRANSPLANT_ADDITION_RATE` | 0.30 | Ch. VIII.G pp.104-107 |
| `TRANSPLANT_DISPLACEMENT_RATE` | 0.12 | Competitive displacement of damaged copies |
| `CD38_BASE_SURVIVAL` | 0.4 | Ch. VI.A.3 p.73 |

---

## Parameter Space

### Intervention Parameters (`INTERVENTION_PARAMS`)

6 parameters, each 0.0-1.0, snapped to grid `[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]`.

### Patient Parameters (`PATIENT_PARAMS`)

6 parameters with per-parameter ranges and grid definitions.

### Grid Snapping

```python
from constants import snap_param, snap_all
snap_param("rapamycin_dose", 0.37)  # → 0.25
snap_all({"rapamycin_dose": 0.37, "baseline_age": 67})
```

---

## Clinical Seeds (`CLINICAL_SEEDS`)

10 hardcoded clinical scenarios (S01-S10) used across all LLM experiments for reproducible comparison. Each has an `id`, `description`, and `category`.

---

## Type Aliases

- `ParamDict = dict[str, float]`
- `InterventionDict = dict[str, float]`
- `PatientDict = dict[str, float]`

---

## Tissue Profiles (`TISSUE_PROFILES`)

| Tissue | `metabolic_demand` | `ros_sensitivity` | `biogenesis_rate` |
|--------|-------------------|--------------------|-------------------|
| brain | 2.0 | 1.5 | 0.7 |
| muscle | 1.5 | 1.0 | 1.3 |
| cardiac | 1.8 | 1.2 | 0.9 |
| default | 1.0 | 1.0 | 1.0 |
