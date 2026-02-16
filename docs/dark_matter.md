# dark_matter

Random sampling of the 12D intervention space to classify futile interventions.

---

## Overview

Adapted from `analyze_dark_matter.py` in the parent Evolutionary-Robotics project (which studied "dead gaits"). Randomly samples intervention vectors and classifies outcomes into a taxonomy. The most interesting category is "paradoxical" — interventions that actively harm the patient.

---

## Outcome Taxonomy

| Category | Definition |
|----------|------------|
| **thriving** | Final ATP > 0.8 AND het < 0.5 |
| **stable** | Final ATP > 0.5 AND het < 0.7 |
| **declining** | Final ATP > 0.2, het > 0.5 |
| **collapsed** | Final ATP < 0.2 |
| **paradoxical** | Worse than no-treatment baseline on BOTH ATP and het |

---

## Key Functions

### `random_intervention(rng) → dict`

Generate a random intervention vector by sampling uniformly from each parameter's grid.

### `classify_outcome(result, baseline_result) → (category, detail_dict)`

Classify a simulation outcome by comparing to no-treatment baseline.

### `identify_culprit(intervention, patient, baseline_result) → (param_name, improvement)`

For paradoxical outcomes, removes each nonzero parameter one at a time to find which is most responsible for harm.

### `run_experiment(n_moderate, n_cliff, seed)`

Run full experiment: 500 random vectors for a moderate patient + 200 for a near-cliff patient. Reports taxonomy breakdown, culprit parameters for paradoxical cases, and comparison of mean intervention doses between paradoxical and thriving outcomes.

---

## Scale

| Patient | Trials | Total |
|---------|--------|-------|
| moderate_60 (60yo, 40% het) | 500 | |
| near_cliff_75 (75yo, 60% het) | 200 | |
| **Total** | | **~700 sims** |

---

## Output

- `artifacts/dark_matter.json` — Full taxonomy with per-trial intervention vectors and culprit analysis
