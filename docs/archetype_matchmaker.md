# archetype_matchmaker

Which character archetypes produce the best intervention protocols for which patient types?

---

## Overview

Combines PDS mapping (Zimmerman 2025 §4.6.4) with character seed experiment data to identify archetype→outcome patterns. Given that different fictional characters produce different intervention protocols when used as LLM prompt seeds, this script answers: which PDS regions produce the best clinical outcomes for each patient profile?

This is a Tier 4 synthesis script: it requires prior data from `character_seed_experiment.py` (no Ollama needed at runtime).

---

## Pipeline

```
1. Load character seed experiment results
   │
2. Compute PDS scores for each character (from archetypometrics TSV)
   │
3. Classify simulation outcomes into clinical tiers
   │
4. Bin PDS into 3³ = 27 discrete cells (low/mid/high per dimension)
   │
5. Analyze: which PDS bins → which outcome distributions?
   │
6. Recommend: for each patient profile, which archetype bin is best?
```

---

## Outcome Classification

### `classify_outcome(analytics) → str`

Four clinically meaningful tiers grounded in Cramer's biological model:

| Tier | Criteria | Clinical Meaning |
|------|----------|-----------------|
| **Collapsed** | het ≥ 0.70 | Past the heteroplasmy cliff; ATP collapses (Cramer Ch. V.K p.66) |
| **Declining** | ATP < 0.5 OR het > 0.6 | Severe energy deficit or dangerously close to cliff |
| **Thriving** | ATP benefit > 0.05 AND het < 0.4 | Active improvement with well-controlled damage |
| **Stable** | Everything else | Maintenance-level outcome |

---

## PDS Binning

### `pds_bin(pds, n_bins=3) → tuple`

Discretizes PDS scores into equal-width bins at ±0.33:

| Bin | Range | Meaning |
|-----|-------|---------|
| `low` | [-1.0, -0.33) | Strongly negative |
| `mid` | [-0.33, +0.33] | Neutral / ambiguous |
| `high` | (+0.33, +1.0] | Strongly positive |

Equal-width bins are chosen over quantile bins because PDS dimensions have interpretable zero points (Zimmerman 2025 §4.6.4). The 3³ = 27 cells provide enough resolution for pattern detection while maintaining statistical power per cell.

---

## Analysis Outputs

### Per-Tier PDS Statistics

Mean and standard deviation of Power, Danger, Structure for each outcome tier. Reveals whether thriving outcomes cluster in particular PDS regions.

### PDS Bin → Success Rates

For each of the 27 bins with sufficient data (n ≥ 5): what fraction of trials are thriving vs. collapsed?

### Patient Profile → Best Archetype

For each patient profile (e.g., "young_healthy", "elderly_inflamed"): which PDS bin produces the highest median ATP benefit?

### Top Character-Protocol Pairs

The 15 character-protocol pairs with highest ATP benefit, with their PDS coordinates.

### Intervention Patterns by Archetype

Mean intervention values (rapamycin, NAD, senolytics, etc.) aggregated by PDS archetype bin.

---

## Usage

```python
from archetype_matchmaker import run_matchmaker

result = run_matchmaker()
# result["profile_recommendations"]["elderly_damaged"] →
#   {"best_pds_bin": ("mid", "low", "high"), "median_benefit": +0.042}
```

```bash
python archetype_matchmaker.py
# Outputs: artifacts/archetype_matchmaker.json
# Requires: artifacts/character_seed_experiment.json (from prior run)
```

---

## Reference

- Zimmerman, J.W. (2025). PhD dissertation, University of Vermont. §4.6.4.
- Dodds, P.S., et al. (2023). "Ousiometrics and telegnomics." *arXiv*.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
