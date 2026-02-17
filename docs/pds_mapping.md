# pds_mapping

Power-Danger-Structure dimension mapping from fictional character archetypes to patient parameters.

---

## Overview

Maps Zimmerman's PDS dimensions — discovered through ousiometric analysis of word meanings (Dodds et al. 2023) — from 2000 Open Psychometrics fictional characters to the 6D patient parameter space of the mitochondrial aging simulator.

The PDS framework (Zimmerman 2025 §4.6.4) identifies three bipolar dimensions that emerge from SVD of character trait ratings:

| Dimension | Poles | Character-Space Mapping |
|-----------|-------|------------------------|
| **Power** | Fool ↔ Hero | Capability, status, agency |
| **Danger** | Angel ↔ Demon | Threat, morality, harm potential |
| **Structure** | Traditionalist ↔ Adventurer | Order, conventionality, discipline |

The alignment between these lexical dimensions and character-space dimensions is *emergent*: none of the words "powerful," "weak," "dangerous," "safe," "structured," or "unstructured" explicitly appear in the 464 bipolar adjective pairs used to construct character space.

---

## PDS → Patient Mapping

Each PDS dimension maps to patient parameters through biologically motivated coefficients:

| PDS Dimension | Patient Parameter | Weight | Biological Rationale |
|---------------|-------------------|--------|---------------------|
| Power | `metabolic_demand` | +0.40 | Powerful characters → high-energy tissues (brain, muscle) |
| Power | `genetic_vulnerability` | -0.20 | Powerful characters → narratively resilient |
| Danger | `inflammation_level` | +0.25 | Dangerous characters → chronic stress → inflammaging |
| Danger | `genetic_vulnerability` | +0.30 | Danger exposure → damage accumulation |
| Structure | `baseline_nad_level` | +0.20 | Disciplined characters → better cellular maintenance |
| Structure | `inflammation_level` | -0.15 | Chaotic lifestyles → chronic inflammation |

The mapping formula for each patient parameter:

```
value = base + Σ(coefficient × PDS_dimension_value)
```

where base values are biologically calibrated midpoints (e.g., `baseline_heteroplasmy` base = 0.2, `genetic_vulnerability` base = 1.0).

**Important:** These coefficients are empirically calibrated to produce biologically plausible distributions when applied to the 2000-character dataset. They are NOT derived from clinical data — this is a semantic bridge for hypothesis generation.

---

## Functions

### `compute_pds(character, available_columns) → dict`

Compute Power, Danger, Structure scores for a single character from trait ratings.

**Trait Mapping:**
- Power: heroic, assertive, dominant, competent, leader, strong (positive); meek, helpless, follower (negative)
- Danger: villainous, cruel, threatening, aggressive, chaotic, dark (positive); innocent, gentle, kind (negative)
- Structure: orderly, traditional, proper, methodical, disciplined (positive); rebellious, chaotic, spontaneous (negative)

Each trait is normalized from 0–100 to -1 to +1 range.

### `pds_to_patient(pds_scores) → dict`

Convert PDS scores to patient parameters. Returns a dict with all 6 patient parameter values, clipped to valid ranges.

### `compare_predictions(characters, experiment_data) → dict`

Compare PDS-predicted patient parameters against LLM-generated parameters from `character_seed_experiment.py`. Reports per-parameter correlation and systematic bias.

---

## Usage

```python
from pds_mapping import compute_pds, pds_to_patient

# Canonical archetypes
hero_pds = {"power": 0.8, "danger": -0.3, "structure": 0.2}
patient = pds_to_patient(hero_pds)
# → metabolic_demand=1.32, genetic_vulnerability=0.71, inflammation_level=0.195, ...

villain_pds = {"power": 0.9, "danger": 0.9, "structure": 0.5}
patient = pds_to_patient(villain_pds)
# → inflammation_level=0.525, genetic_vulnerability=1.09, ...
```

```bash
# Standalone analysis
python pds_mapping.py
# Outputs: artifacts/pds_mapping.json
```

---

## Reference

- Zimmerman, J.W. (2025). PhD dissertation, University of Vermont. §4.6.4.
- Dodds, P.S., et al. (2023). "Ousiometrics and telegnomics." *arXiv*.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
