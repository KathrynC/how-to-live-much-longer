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

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
