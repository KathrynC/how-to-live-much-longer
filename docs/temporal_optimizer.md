# temporal_optimizer

D2: Temporal protocol optimizer — discovering optimal intervention timelines.

---

## Overview

Uses a (1+lambda) evolutionary strategy to discover optimal time-varying intervention schedules. The key insight: mitochondrial aging is highly nonlinear in time (AGE_TRANSITION at 65, cliff at ~0.70), so a phased protocol can outperform a constant-dose protocol with the same total drug exposure.

---

## Biological Motivation

Two key temporal transitions create optimization opportunities:

1. **AGE_TRANSITION at 65** (Cramer Appendix 2 p.155): deletion doubling drops from 11.8yr to 3.06yr — interventions may need to intensify around this age
2. **Heteroplasmy cliff at ~0.70** (Cramer Ch. V.K p.66): once crossed, bistability makes return difficult — timing before crossing is critical

---

## Genotype Representation

3 phases × 6 intervention params + 2 boundary years = 20D genotype.

```
[boundary_1, boundary_2, phase1_rapa, phase1_nad, ..., phase3_exercise]
```

**Mutation operators (50/50 split):**
- Mutate a boundary: Gaussian perturbation (sigma=2yr), clamp and sort
- Mutate one intervention param in one phase: ±1 grid step

---

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `N_PHASES` | 3 | Early/middle/late temporal structure |
| `N_GENERATIONS` | 50 | Typically converges by gen 30-40 |
| `LAMBDA` | 10 | Children per generation |
| `BOUNDARY_SIGMA` | 2.0 yr | Matches deletion doubling timescale (3.06yr) |

---

## Key Metrics

**Timing importance** = (phased_fitness - constant_fitness) / constant_fitness

Measures how much temporal structure matters compared to optimal constant dosing.

---

## Scale

3 patients × (50 gen × 10 lambda + 1 constant) = ~1530 sims + extras ≈ ~3000. Estimated time: ~7 minutes.

---

## Output

- `artifacts/temporal_optimizer.json` — Optimal schedules, phase boundaries, timing importance scores, convergence curves

---

## Reference

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag. Appendix 2 p.155.
