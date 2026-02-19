# Handoff Batch 2: Epigenetic Sustenance & Neuroplasticity Module

**Date:** 2026-02-19
**Status:** Received, not yet implemented. Additional batches may follow.

## Scope

Adds a neuroplasticity pipeline driven by grief-coping intellectual engagement: grief → MEF2 activation → histone acetylation → synaptic plasticity → memory consolidation. Creates a virtuous cycle where intellectual engagement rewires the brain at genetic, epigenetic, synaptic, and functional levels.

## New State Variables

| Variable | Range | Baseline | Description |
|----------|-------|----------|-------------|
| `histone_acetylation` (HA) | 0.0–1.0 | 0.2 | Global histone acetylation at MEF2 target loci (plasticity genes) |
| `synaptic_strength` | 0.0–2.0 | 1.0 | Net synaptic strength in memory circuits (hippocampus) |

## Derived Variable

| Variable | Range | Description |
|----------|-------|-------------|
| `memory_index` | 0.0–1.0 | Composite memory performance score (not an ODE state — computed from synaptic_strength + MEF2) |

## ODE Equations

### Histone Acetylation

```python
dHA/dt = MEF2_activity * HA_INDUCTION_RATE * (1 - HA) - HA * HA_DECAY_RATE
```

- Saturates at 1.0 (logistic-like via `1 - HA` term)
- Slow induction (0.2/yr), very slow decay (0.05/yr) — epigenetic marks are persistent
- MEF2 recruits histone acetyltransferases (HATs) to keep chromatin open

### Synaptic Plasticity

```python
plasticity_factor = PLASTICITY_FACTOR_BASE + PLASTICITY_FACTOR_HA_MAX * HA
# Range: 0.5 (no acetylation) to 1.0 (full acetylation)

dsynaptic_strength/dt = LEARNING_RATE_BASE * engaged * plasticity_factor * (1 - synaptic_strength / MAX_SYNAPTIC_STRENGTH) - SYNAPTIC_DECAY_RATE * (synaptic_strength - 1)
```

- `engaged` is a 0-1 signal for active intellectual engagement (driven by grief coping)
- Capacity-limited growth (logistic toward MAX_SYNAPTIC_STRENGTH = 2.0)
- Decay pulls back toward baseline 1.0 without reinforcement

### Memory Index (derived, not ODE)

```python
memory_index = min(1.0, BASELINE_MEMORY + (synaptic_strength - 1) * SYNAPSES_TO_MEMORY + MEF2_activity * MEF2_MEMORY_BOOST)
```

- Three components: age-adjusted baseline (0.5), synaptic gain contribution, MEF2 direct consolidation boost

## New Constants for constants.py

```python
# Epigenetics (histone acetylation)
HA_INDUCTION_RATE = 0.2           # per unit MEF2 per year
HA_DECAY_RATE = 0.05              # per year
PLASTICITY_FACTOR_BASE = 0.5      # plasticity with zero acetylation
PLASTICITY_FACTOR_HA_MAX = 1.0    # max plasticity factor at full HA (total = 0.5 + 1.0*HA)

# Synaptic plasticity
LEARNING_RATE_BASE = 0.3          # per year of intense engagement
SYNAPTIC_DECAY_RATE = 0.1         # per year (drift back to baseline)
MAX_SYNAPTIC_STRENGTH = 2.0       # ceiling
SYNAPSES_TO_MEMORY = 0.3          # contribution of synaptic gain to memory index

# MEF2 direct memory boost
MEF2_MEMORY_BOOST = 0.2           # direct effect of MEF2 on memory consolidation
BASELINE_MEMORY = 0.5             # age-adjusted baseline memory performance
```

## Causal Chain (Full Interaction Network)

```
grief_intensity → intellectual_engagement (coping mechanism)
    → MEF2 activation
        → ion channel expression (KCNQ, Kv4) → reduced excitability → neuroprotection
        → histone_acetylation (HATs recruited to target loci)
            → enhanced plasticity_factor
                → greater synaptic_strength from learning
                    → improved memory_index
        → direct memory consolidation boost (MEF2_MEMORY_BOOST)
    reduced_excitability → protects synapses/neurons from degeneration
    memory + cognitive performance → reduces stress → supports continued engagement (positive feedback)
```

## Reference Trajectory (Validation Target)

Based on a specific patient case: age 53–63, grief onset at 53, intellectual engagement as coping.

| Age | Grief | MEF2 | HA | Synaptic Strength | Memory Index |
|-----|-------|------|----|-------------------|--------------|
| 53 | 0.90 | 0.20 | 0.20 | 1.00 | 0.50 |
| 55.5 | 0.74 | 0.20 | 0.22 | 1.03 | 0.52 |
| 57 | 0.58 | 0.37 | 0.30 | 1.10 | 0.58 |
| 59 | 0.45 | 0.52 | 0.45 | 1.22 | 0.68 |
| 60 | 0.35 | 0.60 | 0.55 | 1.30 | 0.74 |
| 63 | 0.18 | 0.82 | 1.00 | 1.60 | 0.84 |

Key outcomes at age 63:
- Grief reduced from 0.9 → 0.18
- HA saturated at 1.0 (maximal epigenetic priming)
- Synaptic strength 60% above baseline
- Memory index 0.84 (superior performance for age, cognitive reserve potentially delaying decline by a decade)

## Coupling to Batch 1 Modules

- **Grief module** (batch 1): `grief_intensity` drives `intellectual_engagement` which drives `MEF2_activity`
- **Genetics module** (batch 1): APOE4 vulnerabilities may interact with amyloid clearance, tau pathology (noted as future extension)
- **Sleep module** (batch 1): sleep quality likely affects synaptic consolidation (not yet specified)

## Code Structure

Functions to add to ODE system:

```python
def epigenetic_ode(HA, MEF2, t):
    dHA = MEF2 * HA_INDUCTION_RATE * (1 - HA) - HA * HA_DECAY_RATE
    return dHA

def synaptic_ode(synaptic_strength, HA, engaged, t):
    plasticity_factor = PLASTICITY_FACTOR_BASE + PLASTICITY_FACTOR_HA_MAX * HA
    learning = LEARNING_RATE_BASE * engaged * plasticity_factor * (1 - synaptic_strength / MAX_SYNAPTIC_STRENGTH)
    decay = SYNAPTIC_DECAY_RATE * (synaptic_strength - 1)
    return learning - decay

def memory_index(synaptic_strength, MEF2):
    return min(1.0, BASELINE_MEMORY + (synaptic_strength - 1) * SYNAPSES_TO_MEMORY + MEF2 * MEF2_MEMORY_BOOST)
```

## Design Notes

- HA uses saturation dynamics (`1 - HA` term) — biologically appropriate since chromatin has finite acetylation sites
- Synaptic strength decays toward 1.0 (not 0) — baseline connectivity is maintained even without stimulation
- Memory index is clamped at 1.0 — represents ceiling of measurable performance
- The positive feedback loop (engagement → MEF2 → HA → plasticity → performance → engagement) is the core mechanism explaining resilience; may need dampening to prevent runaway in simulation
