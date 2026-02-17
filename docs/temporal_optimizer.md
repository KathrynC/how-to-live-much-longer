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

Cramer, J.G. (forthcoming from Springer Verlag in 2026). *How to Live Much Longer*. Appendix 2 p.155.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
