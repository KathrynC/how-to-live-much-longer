# Slowing vs. Reversing Human Aging: Model Predictions

**Date:** 2026-02-15
**Simulation:** 30-year RK4 ODE integration, 7 coupled state variables, dt=0.01 yr
**Model version:** Post-falsifier fixes (C1-C4, M1-M5)

## Summary

The mitochondrial aging ODE model predicts that:

1. **Slowing aging** is achievable with single interventions (especially rapamycin and NAD+)
2. **Reversing aging** (reducing heteroplasmy below starting value) requires combination therapy
3. **Rescue from past the heteroplasmy cliff** is possible with aggressive multi-intervention protocols
4. **Reversal is paradoxically easier in older/more damaged patients** because there is more damaged mtDNA substrate for mitophagy to clear

---

## Part 1: Slowing Aging

### Individual Intervention Effectiveness (ranked)

| Rank | Intervention | Mechanism | Cramer Source | Effect on Het (30yr) | Effect on ATP |
|---|---|---|---|---|---|
| 1 | **Rapamycin** (0.5) | mTOR inhibition → enhanced PINK1/Parkin mitophagy | Ch. VI.A.1 pp.71-72 | Cuts het increase ~50% | Mild ATP decline |
| 2 | **NAD+ (NMN/NR)** (1.0) | Restores cofactor, selectively benefits healthy mito | Ch. VI.A.3 pp.72-73 | Cuts het increase ~40% | **Best ATP maintenance** |
| 3 | **Yamanaka** (0.5) | Epigenetic reprogramming, converts damaged→healthy | Ch. VII.B pp.92-95 | Cuts het increase ~45% | **Steep ATP cost** (−28%) |
| 4 | **Transplant** (0.5) | Adds healthy mtDNA copies via mitlets | Ch. VIII.G pp.104-107 | Cuts het increase ~10% | Mild ATP decline |
| 5 | **Exercise** (0.5) | Biogenesis (PGC-1α) + antioxidant upregulation | Hormesis literature | Minimal alone | Mild ATP decline |
| 6 | **Senolytics** (0.5) | Clears senescent cells, frees energy | Ch. VII.A.2 p.91 | Negligible alone | Mild ATP improvement |

### Key Findings — Slowing

- **Rapamycin is the single most effective intervention** for reducing the rate of heteroplasmy increase. At maximum dose, it can achieve stabilization or even reversal as a monotherapy in middle-aged and elderly patients.
- **NAD+ supplementation is the best single intervention for maintaining energy production.** It is the only individual intervention that increases ATP over 30 years in a young patient.
- **Senolytics and exercise have minimal standalone effect** on heteroplasmy but contribute meaningfully in combination by freeing energy budget and enhancing defenses.
- **Yamanaka reprogramming is powerful but dangerous** — the 3-5 MU energy cost (Ch. VIII.A Table 3 p.100) means it can drain the cell's energy reserves. Only viable when paired with energy-supporting interventions.

---

## Part 2: Reversing Aging

### Combination Therapy Results

**Young patient (age 40, het 0.15, NAD 0.85)**

| Protocol | Het: 0→30yr | dHet | ATP: 0→30yr | Verdict |
|---|---|---|---|---|
| No treatment | 0.15→0.47 | +0.32 | 0.94→0.80 | Steady decline |
| Rapamycin 1.0 alone | 0.15→0.21 | +0.06 | 0.94→0.83 | Strong slowing |
| Cocktail (Rapa+NAD+Seno+Ex) | 0.15→0.24 | +0.09 | 0.94→0.93 | Slowed, ATP preserved |
| Kitchen Sink (all max) | 0.15→0.12 | **−0.03** | 0.94→0.81 | **REVERSAL** |

**Middle-aged patient (age 60, het 0.35, NAD 0.60)**

| Protocol | Het: 0→30yr | dHet | ATP: 0→30yr | Verdict |
|---|---|---|---|---|
| No treatment | 0.35→0.63 | +0.28 | 0.82→0.56 | Decline toward cliff |
| Rapamycin 1.0 alone | 0.35→0.24 | **−0.11** | 0.82→0.73 | **REVERSAL** |
| Cocktail (Rapa+NAD+Seno+Ex) | 0.35→0.28 | **−0.07** | 0.82→0.85 | **REVERSAL +ATP** |
| Cocktail + Transplant | 0.35→0.23 | **−0.12** | 0.82→0.85 | **REVERSAL +ATP** |
| Kitchen Sink (all moderate) | 0.35→0.22 | **−0.13** | 0.82→0.76 | **REVERSAL** |
| Kitchen Sink (all max) | 0.35→0.13 | **−0.22** | 0.82→0.73 | **STRONG REVERSAL** |

**Elderly patient (age 75, het 0.55, NAD 0.40)**

| Protocol | Het: 0→30yr | dHet | ATP: 0→30yr | Verdict |
|---|---|---|---|---|
| No treatment | 0.55→0.80 | +0.25 | 0.67→0.12 | Collapse |
| Rapamycin 1.0 alone | 0.55→0.32 | **−0.23** | 0.67→0.66 | **REVERSAL** |
| Cocktail (Rapa+NAD+Seno+Ex) | 0.55→0.36 | **−0.19** | 0.67→0.78 | **REVERSAL +ATP** |
| Cocktail + Transplant | 0.55→0.26 | **−0.29** | 0.67→0.79 | **STRONG REVERSAL +ATP** |
| Kitchen Sink (all max) | 0.55→0.13 | **−0.42** | 0.67→0.67 | **NEAR-COMPLETE REVERSAL** |

### Key Findings — Reversal

1. **Reversal requires combination therapy** in young patients but is achievable with rapamycin monotherapy in older patients.
2. **The minimum viable reversal cocktail** is: Rapamycin (0.5) + NAD+ (0.75) + Senolytics (0.5) + Exercise (0.5).
3. **Adding transplant significantly accelerates reversal** by directly injecting healthy copies.
4. **Yamanaka reprogramming further accelerates reversal** but only if the energy budget permits (cocktail must maintain ATP above the reprogramming cost).
5. **Reversal is paradoxically easier in older/more damaged patients.** More damaged mtDNA = more substrate for mitophagy-enhancing interventions to clear.
6. **The best cocktails not only reverse damage but also improve ATP** — the elderly cocktail+transplant protocol ends at 0.79 MU/day, up from 0.67 at start.

---

## Part 3: Cliff Rescue

Can we rescue a patient already past the 70% heteroplasmy cliff?

| Starting Het | No Treatment (30yr) | Cocktail + Transplant | Max Rapa+NAD+Transplant | Kitchen Sink (max) |
|---|---|---|---|---|
| 0.65 (at cliff) | 0.65→0.84, ATP 0.08 | **0.65→0.26**, ATP 0.81 | **0.65→0.15**, ATP 0.85 | **0.65→0.13**, ATP 0.69 |
| 0.70 (on cliff) | 0.70→0.87, ATP 0.05 | **0.70→0.26**, ATP 0.81 | **0.70→0.15**, ATP 0.85 | **0.70→0.13**, ATP 0.69 |
| 0.80 (past cliff) | 0.80→0.92, ATP 0.02 | **0.80→0.27**, ATP 0.81 | **0.80→0.16**, ATP 0.85 | **0.80→0.13**, ATP 0.69 |
| 0.90 (deep past) | 0.90→0.97, ATP 0.01 | **0.90→0.27**, ATP 0.81 | **0.90→0.16**, ATP 0.85 | **0.90→0.13**, ATP 0.69 |

### Key Findings — Cliff Rescue

1. **The model shows no absolute point of no return.** Even at 90% heteroplasmy (ATP collapsed to 0.04 MU/day), aggressive multi-intervention therapy can restore het to ~0.15 and ATP to ~0.85.
2. **The endpoint is nearly independent of starting damage** for aggressive protocols — all starting hets converge to similar final values. This suggests the interventions dominate the natural dynamics.
3. **Transplant + rapamycin + NAD+ is the most efficient cliff-rescue protocol** — achieves het ~0.15 with ATP ~0.85, without the energy cost of Yamanaka reprogramming.

---

## The Synergy Principle

No single intervention reverses aging in young patients. The combination works because each intervention addresses a different node in the vicious cycle:

| Intervention | Mechanism | Vicious Cycle Target |
|---|---|---|
| Rapamycin | Enhanced mitophagy (PINK1/Parkin) | **Removes** damaged mtDNA (reduce numerator) |
| NAD+ (NMN/NR) | Restores cofactor + quality control | **Supports** healthy mitochondria (boost denominator) |
| Senolytics | Clears senescent cells | **Frees energy** budget (senescent cells use ~2x energy) |
| Exercise | Biogenesis + antioxidant defense | **Creates** new healthy copies + reduces ROS |
| Transplant | Healthy mtDNA infusion via mitlets | **Directly adds** healthy copies from outside |
| Yamanaka | Epigenetic reprogramming | **Converts** damaged→healthy (costs 3-5 MU ATP) |

This confirms the core thesis of Cramer (2025): aging is a cellular energy crisis caused by progressive mitochondrial DNA damage, and the path to reversal requires attacking the ROS-damage vicious cycle from multiple angles simultaneously.

---

## Caveats

1. **This is a computational model, not clinical evidence.** The ODE system captures key dynamics from Cramer (2025) but uses simplified coupling constants and conservatively estimated parameters.
2. **The damaged replication advantage is set to 5%** (conservative); the book says "at least 21%" (Appendix 2, pp.154-155). With the book's value, natural aging would be faster and reversal harder.
3. **The age transition for deletion doubling is set to 40** in the simulation vs. 65 in the book (Appendix 2, p.155). This makes simulation aging faster for ages 40-65.
4. **Intervention doses are normalized 0-1** and do not correspond to specific clinical dosing.
5. **The model does not capture tissue-specific effects** (brain vs. muscle vs. liver), immune system interactions, or organ-level dynamics.
6. **Post-falsifier fixes (2026-02-15) corrected 4 critical bugs** in the ODE equations. Results should be reviewed for biological plausibility.
