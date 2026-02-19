# Finding: Energy Budget Trumps Heteroplasmy as Outcome Determinant

**Date:** 2026-02-19
**Source:** 700-protocol dark_matter sweep through protocol dictionary pipeline
**Branch:** `calibration`

## Summary

Analysis of 700 randomly sampled intervention protocols reveals that
heteroplasmy reduction is **not** a useful predictor of clinical outcome.
Outcome classes (thriving/stable/declining) are determined primarily by
the cell's energy budget, not by how much damaged mtDNA is cleared.

**het-ATP correlation across 700 protocols: r = 0.092** (essentially zero)

99% of all interventions reduce heteroplasmy below baseline, regardless of
whether the patient ends up thriving, stable, or declining. The model's
post-falsifier dynamics (copy number homeostasis, selective mitophagy,
1.21x deletion replication advantage) naturally drive het down over 30 years
under almost any intervention.

## Three-Tier Outcome Hierarchy

### Tier 1: Yamanaka Energy Cost (dominant)

Yamanaka intensity almost perfectly predicts outcome class:

| Outcome    | n   | Yamanaka median | Energy cost | Final ATP | Final het |
|------------|-----|-----------------|-------------|-----------|-----------|
| Thriving   |  28 | 0.00            | 0.13 MU/yr  | 0.823     | 0.200     |
| Stable     | 510 | 0.25            | 1.28 MU/yr  | 0.667     | 0.223     |
| Declining  | 162 | 1.00            | 4.43 MU/yr  | 0.396     | 0.206     |

Note that het is nearly identical across all three classes (0.200-0.223).

Binary test:
- Yamanaka OFF (<0.05): 0/116 declining, 20/116 thriving
- Yamanaka HIGH (>=0.5): 162/380 declining, 0/380 thriving

The causal chain: High Yamanaka (3-5 MU/day, Cramer Ch. VIII.A Table 3)
drains ATP -> cells pushed into senescence (declining: 0.238 vs thriving:
0.123) -> SASP inflammation -> further ATP drain -> declining outcome,
despite het being well-controlled.

### Tier 2: NAD Supplementation (secondary)

Among energy-neutral protocols (Yamanaka <= 0.1, n=230):

| Outcome   | n   | NAD supplement | Final ATP | Final het |
|-----------|-----|----------------|-----------|-----------|
| Thriving  |  27 | 0.89           | 0.824     | 0.201     |
| Stable    | 203 | 0.34           | 0.736     | 0.251     |
| Declining |   0 | -              | -         | -         |

Without Yamanaka energy drain, NO protocol produces a declining outcome.
The thriving/stable split is driven by NAD supplement level (0.89 vs 0.34).
het-ATP correlation improves to r = -0.328 when Yamanaka is excluded, but
remains modest.

### Tier 3: Heteroplasmy (tertiary)

Het shows a weak signal only after controlling for energy budget:
thriving 0.201 vs stable 0.251 (delta = 0.05). This is a real but small
effect compared to the ATP differences driven by Tiers 1 and 2.

## By Patient Type

Both patient profiles show the same pattern:

**moderate_60** (baseline het=0.40, n=500):
- 99% of protocols reduce het below 0.40
- No-treatment baseline: ATP=0.731, het=0.484
- Even declining protocols reduce het (mean 0.178 vs baseline 0.40)

**near_cliff_75** (baseline het=0.60, n=200):
- 99% of protocols reduce het below 0.60
- No-treatment baseline: ATP=0.648, het=0.701 (crosses cliff!)
- Even declining protocols reduce het (mean 0.259 vs baseline 0.60)

Only 6 out of 700 protocols (0.9%) raised heteroplasmy above baseline.
These were minimal-intervention protocols (near-zero across all params)
where natural aging outpaced the negligible intervention.

## Mechanism

The post-falsifier ODE dynamics explain why het universally drops:

1. **Copy number homeostasis** (fix C2): Total N_h + N_d regulated toward
   1.0 via TFAM-mediated copy number pressure. Prevents unbounded growth.

2. **Selective mitophagy** (fix C3): PINK1/Parkin pathway preferentially
   degrades deletion-bearing mitochondria (defective ETC -> low membrane
   potential -> PINK1 accumulation). This quality control mechanism
   continuously removes damaged copies.

3. **Deletion replication advantage** (1.21x, Appendix 2 pp.154-155):
   While deletions replicate faster, the homeostatic ceiling means this
   advantage mainly displaces healthy copies rather than increasing total
   damaged count. Under any intervention that boosts mitophagy (rapamycin)
   or adds healthy copies (transplant), the net flow favors het reduction.

4. **ATP determines everything else**: Once het drops below the cliff,
   ATP production depends on NAD availability, senescence burden, and
   energy costs — not on the specific het level. A patient at het=0.20
   has essentially the same ATP capacity as one at het=0.15.

## Implications

### For the Simulator

The model correctly captures Cramer's emphasis on Yamanaka energy cost
(Ch. VIII.A: "demanding 3 to 10 times as much ATP energy as normal
somatic cell operation"). The finding validates that the energy budget
dynamics are working as designed.

However, the dark_matter sweep uniformly samples Yamanaka 0-1, meaning
~50% of protocols have clinically unrealistic high-Yamanaka doses. Future
sweeps should use weighted sampling reflecting clinical feasibility
(most protocols would use Yamanaka <= 0.25).

### For Clinical Interpretation

**Testable prediction**: NAD supplementation level matters more than
heteroplasmy reduction rate for patient outcomes, provided the
intervention protocol does not exceed the cell's energy budget.

**Practical guideline**: Evaluate protocols primarily by:
1. Net energy cost (must be < 1 MU/yr for favorable outcome)
2. NAD restoration level (higher = better, gated by CD38 suppression)
3. Het reduction (a bonus, but not the primary target)

### For the Protocol Dictionary Pipeline

The classifier prototypes were recalibrated based on this finding:
- Old prototypes used het as a primary discriminator (het 0.20-0.85)
- New prototypes reflect the ATP-dominant reality (het 0.20-0.22 for
  thriving/stable/declining, separated by ATP 0.40-0.82)
- Classifier agreement improved from 16.7% to 79.1%

## Data

- Source: `artifacts/dark_matter.json` (700 trials, 2 patient profiles)
- Pipeline output: `artifacts/protocol_pipeline/protocol_dictionary.json`
- Classifier calibration commit: `5c46201`
