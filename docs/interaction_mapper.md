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

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag. Ch. VI-VIII.
Bliss, C.I. (1939). The toxicity of poisons applied jointly. *Annals of Applied Biology*.
