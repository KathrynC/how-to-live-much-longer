# competing_evaluators

D5: Competing evaluators — multi-criteria robust protocol search.

---

## Overview

Finds protocols robust across 4 competing clinical criteria — the TI "transaction" that resonates across all "absorbers." A "transaction" protocol is one in the top-25% for ALL 4 evaluators simultaneously. Robustness score = harmonic mean of percentile ranks (penalizes any single weak dimension).

---

## Evaluators

| Evaluator | Clinical Priority | Score Formula |
|-----------|-------------------|---------------|
| **ATP_Guardian** | Energy preservation | `atp_benefit_mean - 0.5 × energy_cost_per_year` |
| **Het_Hunter** | Root cause reduction | Heteroplasmy benefit vs baseline |
| **Crisis_Delayer** | Buy time | Years of crisis delay |
| **Efficiency_Auditor** | First, do no harm | Benefit per unit intervention intensity |

**TIQM analogy:** Each evaluator is a different clinical "absorber." A transaction protocol must resonate with all 4 simultaneously — the rare intervention that satisfies every clinical priority at once.

---

## Key Functions

### `atp_guardian(analytics) → float`

Maximize energy preservation. Penalizes ATP-consuming interventions (especially Yamanaka at 3-5 MU/day).

### `het_hunter(analytics) → float`

Aggressively reduce heteroplasmy — the upstream driver of all downstream pathology.

### `crisis_delayer(analytics) → float`

Maximize years before the heteroplasmy cliff triggers irreversible collapse.

### `efficiency_auditor(analytics, intervention) → float`

Minimize intervention burden per unit benefit.

### `run_experiment()`

Generate ~500 candidate protocols via Latin hypercube sampling. Score each with all 4 evaluators. Find transaction protocols (top-25% on all 4). Compute 4D Pareto frontier and evaluator agreement matrix. Optionally ingests Pareto/synergy data from D1/D4.

---

## Scale

~500 candidates × 1 patient + baselines = ~1000 simulations. Estimated time: ~2 minutes.

---

## Output

- `artifacts/competing_evaluators.json` — Transaction protocols, evaluator scores, agreement matrix, 4D Pareto frontier

---

## Reference

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag.
