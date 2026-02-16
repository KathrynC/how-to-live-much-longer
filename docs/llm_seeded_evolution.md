# llm_seeded_evolution

Hill-climbing from LLM-generated seeds vs random seeds.

---

## Overview

Adapted from the parent Evolutionary-Robotics project. Tests whether LLM-generated intervention vectors are better starting points for optimization than random vectors — the "launchpad vs trap" hypothesis. Uses single-parameter hill-climbing mutations.

---

## Experimental Design

| Population | Seeds | Evaluations | Total Sims |
|------------|-------|-------------|------------|
| LLM seeds | 20 | 100 each | 2000 |
| Random seeds | 20 | 100 each | 2000 |
| **Total** | 40 | | **~4000** |

**Fitness function:** ATP benefit over no-treatment baseline for a fixed test patient (60yo, 40% het).

**Mutation:** Single-parameter ±1 grid step per evaluation (hill-climbing).

---

## Key Functions

### `fitness(intervention) → float`

Compute fitness as ATP benefit over cached no-treatment baseline. Higher = better.

### `run_experiment()`

Execute full comparison. Reports:
- Final fitness distributions (LLM vs random seeds)
- Convergence speed (evaluations to reach 90% of final fitness)
- Whether LLM seeds reach higher peaks or get trapped in local optima

---

## Modes

| Flag | Description |
|------|-------------|
| (none) | Load LLM seeds from prior experiment data, pure simulation |
| `--narrative` | Generate LLM seeds with trajectory-feedback evolution (requires Ollama) |

---

## Scale

40 seeds × 100 evaluations = ~4000 simulations. Estimated time: ~10 minutes.

---

## Output

- `artifacts/llm_seeded_evolution.json` — Per-seed fitness trajectories, final rankings, launchpad/trap classification
