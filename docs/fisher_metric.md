# fisher_metric

LLM output variance measurement to quantify clinical certainty.

---

## Overview

Adapted from the parent Evolutionary-Robotics project. Queries each model 10 times per clinical scenario with identical prompts and measures output variance. High variance indicates genuine clinical ambiguity; low variance indicates the LLM has a strong opinion. Constructs a 12×12 covariance matrix per (scenario, model) pair — the Fisher information metric of the LLM's "clinical intuition."

---

## Experimental Design

| Dimension | Values |
|-----------|--------|
| **Scenarios** | 10 clinical seeds from `CLINICAL_SEEDS` |
| **Repeats** | 10 per (scenario, model) pair |
| **Models** | 4 local Ollama models |
| **Total queries** | 400 |

---

## Key Metrics

For each (scenario, model) pair:
- **Per-parameter variance**: Which parameters the LLM is uncertain about
- **12×12 covariance matrix**: Parameter correlation structure in LLM outputs
- **Total variance** (trace of covariance): Scalar certainty measure
- **Fisher information** (inverse of covariance): Precision of clinical intuition

---

## Key Functions

### `run_experiment(seeds, n_repeats)`

Execute full experiment with checkpointing. For each scenario × model, query `n_repeats` times and compute variance statistics. Reports high/low certainty scenarios and per-model consistency.

---

## Interpretation

| Variance Level | Meaning |
|----------------|---------|
| Low (all params consistent) | LLM has clear clinical reasoning for this scenario |
| High (intervention params vary) | Genuine ambiguity about treatment approach |
| High (patient params vary) | LLM uncertain about patient characterization |
| Correlated (e.g., rapamycin ↔ NAD) | LLM treats these as trade-offs |

---

## Scale

400 Ollama queries. Estimated time: ~30-60 minutes.

---

## Output

- `artifacts/fisher_metric.json` — Per-(scenario, model) covariance matrices and summary statistics
