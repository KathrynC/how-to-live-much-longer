# Finding: Post-Recalibration Outcomes

**Date:** 2026-02-19
**Source:** NATURAL_HEALTH_REF recalibration, dark_matter re-run, causal_surgery re-run
**Branch:** `causal-recal`

## Summary

After the NAD coefficient reduction (0.4→0.2), cliff recalibration (0.70→0.50), and
NATURAL_HEALTH_REF update (0.77→0.91), three experiments were re-run to assess the
recalibrated model's behavior.

## 1. NATURAL_HEALTH_REF Recalibration

The C10 calibration constant ensures that a naturally aging person has shift ≈ 0 at
age 65 (the Va23 empirical transition point). After changing the NAD coefficients and
cliff threshold, the default patient's ATP at age 65 rose from 0.774 to 0.908.

Iterative convergence:
```
Iteration 0: ref=0.7700, ATP@65=0.9077
Iteration 1: ref=0.9077, ATP@65=0.9076
Converged: ref=0.91, residual shift < 0.01 years (<1 day)
```

## 2. Dark Matter Re-run (700 Trials, Clinical Yamanaka Weighting)

**Clinical Yamanaka weighting:** Sampling now 75% at dose ≤0.25, reflecting that
high-intensity reprogramming is experimental and costs 3-5 MU ATP.

| Category | Old (pre-recal) | New | Change |
|----------|----------------|-----|--------|
| thriving | — | 430 (61%) | — |
| stable | — | 234 (33%) | — |
| declining | — | 36 (5%) | — |
| collapsed | — | 0 (0%) | Eliminated |
| paradoxical | — | 0 (0%) | Eliminated |

**Key findings:**
- 0 paradoxical cases — the reduced NAD coefficients eliminated false hype-driven harm
- 0 collapsed cases — the moderate/near-cliff patients don't cross the point of no return
- All worst outcomes have yamanaka_intensity=1.0 — Yamanaka energy cost remains
  the primary harm vector

### Energy Budget Finding Update

The previous finding (energy_budget_trumps_heteroplasmy) noted that het didn't
discriminate outcomes. With the recalibrated cliff:

| Class | Mean ATP | Mean het | Mean Yamanaka | High Yamanaka (≥0.5) |
|-------|----------|----------|---------------|---------------------|
| thriving | 0.849 | 0.248 | 0.083 | 0 |
| stable | 0.717 | 0.251 | 0.451 | 148 |
| declining | 0.617 | 0.465 | 0.535 | 19 |

**Het is now a secondary predictor for declining patients** (het=0.47 vs 0.25 for
thriving/stable), but the primary discriminator remains Yamanaka intensity. The
energy budget finding is **nuanced, not invalidated**: energy cost (Yamanaka) is
still the #1 outcome driver, but het now provides additional signal for the worst
outcomes. Overall ATP-het correlation remains weak (r=-0.13).

## 3. Causal Surgery Re-run (195 Sims)

### Point of No Return (Forward Surgery)

With the recalibrated parameters, all interventions are helpful at all time points
for these three patients — there is no "too late" within the 30-year window.
This is because the moderate_60 (het=0.40) and near_cliff_75 (het=0.60) patients
don't cross the bistability threshold at het~0.93.

### Treatment Duration Threshold (Reverse Surgery)

| Patient | Intervention | Min Duration for Lasting Benefit |
|---------|-------------|--------------------------------|
| Moderate 60yo | Rapamycin only | 8 years |
| Moderate 60yo | Full cocktail | 5 years |
| Moderate 60yo | Transplant only | 5 years |
| Near-cliff 75yo | Rapamycin only | 2 years |
| Near-cliff 75yo | Full cocktail | 2 years |
| Near-cliff 75yo | Transplant only | 2 years |

**Key insight:** Near-cliff patients benefit from even short treatment durations
(2 years) because transplant and mitophagy quickly displace damaged copies.
The closer to the cliff, the more urgently treatment should begin — but also
the faster the benefit accumulates.

### Comparison with Pre-Recalibration

The old causal_surgery data (with cliff=0.70, NAD=0.4) showed different dynamics
because the cliff was unreachable under C11. The new results with the activated
cliff show that treatment timing matters more for patients closer to the
het=0.50 deletion cliff.

## Pipeline Integration

- Dark matter: 700 protocols ingested, 76 sent to review
- Causal surgery: 192 protocols ingested, 48 sent to review
- Total dictionary: 892 protocols with enrichment, classification, and rewrite rules
