# oeis_seed_experiment

OEIS integer sequences as semantic seeds for LLM-mediated intervention design.

---

## Overview

Adapted from the parent Evolutionary-Robotics project. Presents ~99 curated OEIS sequences to 4 local Ollama models, asking each to translate the mathematical "character" of the sequence into a 12D intervention + patient vector. Tests whether mathematical structure influences clinical reasoning.

---

## Pipeline

```
OEIS sequence (ID, name, first 16 terms)
    ↓
LLM prompt: "Translate this sequence's character into a protocol"
    ↓
Parse 12D vector → snap to grid
    ↓
Simulate 30-year ODE → 4-pillar analytics
    ↓
Record: sequence → vector → outcome
```

---

## Sequence Categories

| Category | Examples | Expected Mapping |
|----------|----------|-----------------|
| Growth & Accumulation | Factorials, powers, squares | Aggressive intervention |
| Fibonacci & Golden Ratio | Fibonacci, Lucas, Padovan | Balanced, moderate |
| Primes & Divisibility | Primes, totient, Moebius | Targeted single-agent |
| Combinatorial | Catalan, Bell, partitions | Complex combinations |
| Oscillating & Periodic | Thue-Morse, period 2, dragon curve | Pulsed treatment |
| Chaotic & Pseudorandom | Pi digits, sqrt(2) | Experimental combos |
| Constant & Minimal | All 1's, all 0's | Watchful waiting |

---

## Key Functions

### `fetch_oeis_sequence(seq_id, cache_dir) → dict|None`

Fetch sequence from OEIS API, caching locally. Polite 0.3s sleep between requests.

### `run_experiment()`

Execute full experiment with checkpointing. Saves checkpoint every 50 trials. Reports per-model statistics and top 10 most effective / most harmful sequences.

---

## Scale

~99 sequences × 4 models = ~396 trials. Estimated time: ~2 hours.

---

## Output

- `artifacts/oeis_seed_experiment.json` — Full results
- `artifacts/oeis_cache/*.json` — Cached OEIS data
- `artifacts/oeis_seed_experiment_checkpoint.json` — Resumable checkpoint (deleted on completion)

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
