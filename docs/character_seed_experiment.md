# character_seed_experiment

Fictional character identities as semantic seeds for LLM-mediated intervention design.

---

## Overview

Adapted from the parent Evolutionary-Robotics project. Presents ~2000 fictional characters to 4 local Ollama models, asking each to reason about the character's personality, lifestyle, age, and health trajectory to design a personalized 12D intervention + patient vector. The largest-scale LLM experiment in the project.

---

## Pipeline

```
Character (name, story) from archetypometrics_characters.tsv
    ↓
LLM prompt: "Design a mitochondrial protocol for [Character/Story]"
    ↓
Parse 12D vector → snap to grid
    ↓
Simulate 30-year ODE → 4-pillar analytics
    ↓
Record: character → vector → outcome
```

---

## Prompt Design

The LLM is asked to consider each character's:
- Age and life stage
- Energy level and physical activity
- Stress and inflammation (villains, warriors → high; peaceful → low)
- Resilience and genetic robustness
- Tissue vulnerability (intellectual → brain/high demand; physical → muscle)

---

## Key Functions

### `run_experiment()`

Execute full experiment with checkpointing. Loads characters from `archetypometrics_characters.tsv` (copied from parent project if not found locally). Saves checkpoint periodically. Reports per-model statistics and top/bottom characters by ATP benefit.

---

## Scale

~2000 characters × 4 models = ~8000 trials. Estimated time: ~4-5 hours (overnight run).

---

## Output

- `artifacts/character_seed_experiment.json` — Full results
- `artifacts/character_seed_experiment_checkpoint.json` — Resumable checkpoint

---

## Data Source

Character data: `archetypometrics_characters.tsv` from the parent Evolutionary-Robotics project. Contains character name and story/work fields.
