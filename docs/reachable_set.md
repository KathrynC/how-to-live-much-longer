# reachable_set

D1: Reachable set mapper — the "advanced wave" of achievable outcomes.

---

## Overview

Maps the achievable outcome space using Latin hypercube sampling of the 6D intervention space. Discovers what outcomes are possible per patient type, computes the Pareto frontier (minimize het, maximize ATP), and finds minimum-intervention paths to predefined health targets.

**TIQM analogy:** The "advanced wave" propagates backward from desired futures to constrain which transactions are possible. The reachable set IS the space of possible transactions between interventions and outcomes.

---

## Computations

### 1. Reachable Region Boundary

Envelope of all achievable (heteroplasmy, ATP) outcomes from Latin hypercube sampling. Shows what biology permits.

### 2. Pareto Frontier

Optimal trade-off curve: no protocol on the frontier can improve one metric without worsening the other. Computed by non-dominated sorting.

### 3. Minimum-Intervention Paths

Least aggressive protocol achieving each health target:

| Target | het constraint | ATP constraint | Clinical Goal |
|--------|---------------|----------------|---------------|
| `maintain_health` | < 0.4 | > 0.7 | Stay well below cliff |
| `significant_reversal` | < 0.3 | > 0.6 | Roll back decades of damage |
| `aggressive_reversal` | < 0.2 | > 0.5 | Approach youthful het |
| `cliff_escape` | < 0.6 | > 0.4 | Pull back from precipice |

---

## Patient Profiles

| ID | Description |
|----|-------------|
| `young_25` | Young prevention (25yo, 10% het) |
| `moderate_70` | Moderate aging (70yo, 30% het) |
| `near_cliff_80` | Near cliff (80yo, 65% het) |

---

## Scale

400 samples × 3 patients = 1200 sims + target searches ~1200 = ~2400 total. Estimated time: ~5 minutes.

---

## Output

- `artifacts/reachable_set.json` — Reachable boundaries, Pareto frontiers, minimum-intervention paths per patient

---

## Reference

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag.
