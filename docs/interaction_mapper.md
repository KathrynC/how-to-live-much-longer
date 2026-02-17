# interaction_mapper

D4: Intervention interaction mapper — synergy and antagonism discovery.

---

## Overview

Discovers synergistic and antagonistic intervention combinations by running 2D grid sweeps of all 15 intervention pairs (C(6,2)) across 3 patient types. Uses the Bliss independence model to quantify super-additivity (synergy) and sub-additivity (antagonism).

**Synergy = actual_fitness(A,B) - (fitness_A_alone + fitness_B_alone)**

Positive → pathways complement each other. Negative → pathways compete for the same resource.

---

## Biological Motivation

| Combination | Expected Interaction | Mechanism |
|-------------|---------------------|-----------|
| Rapamycin + transplant | Synergy | Clear damaged + add healthy (complementary) |
| Yamanaka + exercise | Antagonism | Both consume ATP → energy crisis (Ch. VIII.A Table 3) |
| NAD + exercise | Complex | Exercise-induced ROS may deplete NAD faster |
| Rapamycin + NAD | Synergy | Enhanced mitophagy + restored cofactor |

Key question: do synergy patterns **reverse** between young and near-cliff patients, due to nonlinear cliff dynamics?

---

## Patient Profiles

| ID | Description |
|----|-------------|
| `young_25` | Young prevention (25yo, 10% het) |
| `moderate_70` | Moderate aging (70yo, 30% het) |
| `near_cliff_80` | Near cliff (80yo, 65% het) |

---

## Key Functions

### `run_experiment()`

Execute 2D grid sweeps. For each pair × patient: 6×6 grid of doses + 6+6 singles for baseline. Reports synergy/antagonism matrix, strongest interactions, and patient-dependent reversals.

---

## Scale

15 pairs × 6×6 grid × 3 patients = ~1620 combination sims + ~540 singles = ~2160 total. Estimated time: ~3 minutes.

---

## Output

- `artifacts/interaction_mapper.json` — Synergy matrices, per-pair dose-response surfaces, patient-dependent interaction patterns

---

## Reference

Cramer, J.G. (forthcoming from Springer Verlag in 2026). *How to Live Much Longer*. Ch. VI-VIII.
Bliss, C.I. (1939). The toxicity of poisons applied jointly. *Annals of Applied Biology*.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
