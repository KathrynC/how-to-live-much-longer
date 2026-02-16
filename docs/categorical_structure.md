# categorical_structure

Formal functor validation: Sem → Vec → Beh.

---

## Overview

Adapted from the parent Evolutionary-Robotics project. Validates the categorical structure of the LLM→simulation pipeline as a pair of functors:

```
F: ClinicalSeed → InterventionVector   (LLM's clinical reasoning)
G: InterventionVector → PatientOutcome  (ODE simulation)
G∘F: ClinicalSeed → PatientOutcome     (end-to-end)
```

Pure computation on existing data from seed experiments. No simulations or LLM queries.

---

## Validation Tests

| Test | Question |
|------|----------|
| **Functoriality** | Do nearby seeds produce nearby vectors? |
| **Continuity** | Do nearby vectors produce nearby outcomes? |
| **Faithfulness** | Do different seeds produce different outcomes? |
| **Sheaf consistency** | Is the mapping locally coherent? |
| **Information geometry** | Fisher-like analysis of the mapping |

---

## Data Sources

Loads successful results from prior experiments:
- `artifacts/oeis_seed_experiment.json` — OEIS sequence → vector → outcome
- `artifacts/character_seed_experiment.json` — Character → vector → outcome

Outcome vector: `[atp_final, atp_mean, atp_slope, het_final, delta_het, time_to_cliff]`

---

## Key Functions

### `load_experiment_data() → list[dict]`

Load and flatten successful results from seed experiments into records with `seed_label`, `vector_flat` (12D), `outcome_flat` (6D), and `model`.

---

## Scale

Pure computation on existing data. Estimated time: ~5 seconds.

---

## Output

- `artifacts/categorical_structure.json` — Functor validation metrics, distance correlations, faithfulness scores
