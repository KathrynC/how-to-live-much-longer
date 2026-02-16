# clinical_consensus

Multi-model agreement analysis for mitochondrial intervention design.

---

## Overview

Queries all 4 local LLMs with the same clinical scenario and measures agreement. Consensus indicates robust clinical reasoning; disagreement flags genuinely ambiguous decisions. Adapted from multi-model comparison in the parent project.

---

## Experimental Design

| Dimension | Values |
|-----------|--------|
| **Scenarios** | 10 clinical seeds from `CLINICAL_SEEDS` |
| **Models** | 4 local Ollama models |
| **Queries** | 40 (10 × 4) |
| **Simulations** | 40 (one per model-scenario pair) |

---

## Analysis Methods

### Pairwise Cosine Similarity

For each scenario, compute cosine similarity between all C(4,2)=6 model pairs in 12D parameter space. High similarity = models agree.

### Per-Parameter Agreement

Standard deviation across models for each of 12 parameters. Low std = consensus; high std = controversial parameter.

### Outcome Comparison

Simulate all 40 proposals and compare resulting ATP/het outcomes. Tests whether protocol disagreement leads to outcome disagreement.

---

## Key Functions

### `cosine_similarity(a, b) → float`

Cosine similarity between two 12D vectors.

### `param_agreement(vectors_dict, param_names) → dict`

Per-parameter standard deviation across models.

### `run_experiment(seeds)`

Execute full consensus analysis. Reports:
- Per-scenario consensus score (mean pairwise cosine similarity)
- Most/least controversial scenarios
- Per-parameter agreement ranking
- Outcome diversity across model proposals

---

## Scale

40 Ollama queries + 40 simulations. Estimated time: ~15-20 minutes.

---

## Output

- `artifacts/clinical_consensus.json` — Per-scenario model agreement, pairwise similarities, parameter-level analysis
