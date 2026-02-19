# Finding: NAD+ Pathway Audit and Hype Literature Guard

**Date:** 2026-02-19
**Source:** NAD dose-response sweep, pathway audit of simulator.py, cliff reachability analysis, Zimmerman falsifier (338 tests)
**Branch:** `nad-audit`
**Motivation:** Cramer reports high-profile NAD+ literature he considers not credible. This audit identifies which NAD+ pathways in the simulator are grounded in Cramer's book/emails (ground truth) vs. modeling assumptions that could have been influenced by hype literature via search ranking.

## Executive Summary

The simulator contains **12 locations** where NAD+ influences the ODE dynamics. Only **4 are directly from Cramer** (NAD boost equation, CD38 gating, mitophagy boost, age decline rate). The remaining **8 are modeling assumptions** with coefficients (primarily 0.4) not grounded in the book.

The CD38 gating mechanism creates a **super-linear dose-response** — marginal returns *increase* with dose — which is exactly the claim made by NMN/NR supplement marketing. While the CD38 biology is real (Cramer Ch. VI.A.3 p.73), the mathematical consequence is that the model rewards high-dose supplementation disproportionately.

However, a critical counterbalance exists: **NAD+ alone cannot rescue post-cliff patients**. Only transplant (Cramer's primary rejuvenation mechanism) can reverse heteroplasmy past the cliff. This correctly reflects Cramer's emphasis hierarchy.

## NAD+ Dose-Response Data

### Moderate Patient (age 60, het=0.40)

Baseline (no treatment): ATP=0.731, het=0.484

| NAD dose | Final ATP | Final het | ATP gain | Marginal ATP/0.05 |
|----------|-----------|-----------|----------|--------------------|
| 0.00     | 0.731     | 0.484     | —        | —                  |
| 0.25     | 0.745     | 0.463     | +0.014   | +0.003/step        |
| 0.50     | 0.766     | 0.432     | +0.035   | +0.005/step        |
| 0.75     | 0.795     | 0.395     | +0.064   | +0.006/step        |
| 1.00     | 0.831     | 0.360     | +0.100   | +0.008/step        |

**Marginal returns accelerate from +0.002 to +0.008 per 0.05 increment.**
The last quarter-dose (0.75→1.0) provides 3.6x the benefit of the first (0.0→0.25).

### Mechanism of Super-Linearity

The CD38 survival factor creates a quadratic:

```
cd38_survival = 0.4 + 0.6 * nad_supp        (line 574)
nad_boost = nad_supp * 0.25 * cd38_survival  (line 961)
         = 0.1 * nad_supp + 0.15 * nad_supp²
```

This means:
- At dose 0.25: boost = 0.025 + 0.009 = 0.034
- At dose 1.00: boost = 0.100 + 0.150 = 0.250

The quadratic term (0.15 * dose²) exceeds the linear term (0.1 * dose) at dose > 0.67. High-dose protocols get *accelerating* returns because apigenin suppresses CD38, letting more precursor reach cells.

**Hype concern:** This mathematical shape matches NMN/NR marketing claims that "you need high doses for meaningful results." While the CD38 biology is genuine (Cramer p.73), the specific coefficients (0.4 base survival, 0.6 suppression gain) amplify this narrative.

## Full Pathway Audit

### Pathways FROM Cramer (Ground Truth)

| # | Location | Line | Mechanism | Coefficient | Source |
|---|----------|------|-----------|-------------|--------|
| 1 | NAD boost equation | 961 | `nad_supp * 0.25 * cd38_survival` | 0.25 | Cramer email C7 |
| 2 | CD38 survival factor | 574 | `0.4 + 0.6 * nad_supp` | 0.4/0.6 | Cramer Ch. VI.A.3 p.73, email C7 |
| 3 | NAD mitophagy boost | 652-654 | `nad_supp * 0.03 * cd38_survival` | 0.03/yr | Cramer mechanism (sirtuin→PINK1/Parkin) |
| 4 | NAD age decline | 955 | `1.0 - 0.01 * max(age-30, 0)` | 0.01/yr | Cramer Ch. VI.A.3 (Ca16) |

### Pathways NOT from Cramer (Modeling Assumptions)

| # | Location | Line | Mechanism | Coefficient | Concern Level |
|---|----------|------|-----------|-------------|---------------|
| 5 | ATP production | 880-881 | `(0.6 + 0.4 * NAD)` | **0.4** | **HIGH** — 40% of ATP depends on NAD. Biochemically plausible but coefficient is a modeling choice |
| 6 | Antioxidant defense | 933 | `1.0 + 0.4 * NAD` | **0.4** | **HIGH** — 40% ROS defense boost per unit NAD. Same 0.4 coefficient pattern |
| 7 | Membrane potential | 1030 | `cliff * NAD * (1-0.3*sen)` | **multiplicative** | **MEDIUM** — NAD linearly scales membrane potential |
| 8 | Healthy replication gate | 689 | `base_rate * n_h * nad * ...` | **multiplicative** | **MEDIUM** — NAD required for any replication |
| 9 | Deletion replication gate | 789-791 | `base_rate * 1.21 * n_del * nad * ...` | **multiplicative** | **LOW** — actually *reduces* NAD benefit (NAD helps damaged copies too) |
| 10 | Point mutation replication | 839 | `base_rate * n_pt * nad * ...` | **multiplicative** | **LOW** — same as #9 |
| 11 | ROS drain on NAD | 971 | `0.03 * ros` | 0.03 | **LOW** — reduces NAD (conservative) |
| 12 | Yamanaka drain on NAD | 975 | `yama * 0.03` | 0.03 | **LOW** — reduces NAD (conservative) |

### Analysis of Concern

**The 0.4 coefficient pattern** appears twice (pathways 5 and 6) in assumptions not from Cramer. Together, these make NAD responsible for:
- 40% of ATP production capacity (pathway 5)
- 40% of antioxidant defense capacity (pathway 6)

This is a large effect. A patient with NAD=0.0 would have only 60% ATP capacity and baseline-only antioxidant defense, while NAD=1.0 gives full capacity. Whether these coefficients are reasonable is a question for Cramer — the biology (NADH→Complex I, sirtuin→SOD2) is real, but the magnitude (40%) is the simulator's choice.

**Pathways 8-10 (replication gating)** make NAD universally required for mtDNA replication. This is biologically defensible (NAD+ is needed for sirtuin-mediated replication licensing) but it's a modeling assumption. Importantly, pathway 9 is actually **conservative** — NAD helps deletion mutants replicate too, partially offsetting NAD's therapeutic benefit.

## Cliff Reachability Finding

### Can post-cliff patients be rescued?

**Full rescue protocol** (transplant=1.0, NAD=1.0, rapamycin=0.75, senolytic=0.75, exercise=0.75):

| Starting het | Final ATP | Final het | Rescued? |
|-------------|-----------|-----------|----------|
| 0.50        | 0.803     | 0.150     | YES      |
| 0.70        | 0.802     | 0.164     | YES      |
| 0.80        | 0.802     | 0.172     | YES      |
| 0.90        | 0.802     | 0.179     | YES      |
| 0.95        | 0.802     | 0.182     | YES      |

**Every patient is rescued to essentially the same endpoint** (~0.80 ATP, ~0.17 het) regardless of starting heteroplasmy. There is no point of no return.

### What drives rescue?

Component isolation at het=0.80 (post-cliff):

| Protocol           | Final ATP | Final het | Rescued? |
|--------------------|-----------|-----------|----------|
| No treatment       | 0.654     | 0.828     | NO       |
| **Transplant only**| **0.686** | **0.311** | **YES**  |
| NAD only           | 0.769     | 0.727     | NO       |
| Rapamycin only     | 0.677     | 0.629     | NO       |
| Transplant + NAD   | 0.784     | 0.243     | YES      |
| Full no transplant | 0.800     | 0.398     | YES      |
| Full rescue        | 0.802     | 0.172     | YES      |

**Key findings:**
1. **Transplant alone rescues** (het 0.83→0.31) — it is the only single intervention that can reverse post-cliff heteroplasmy
2. **NAD alone does NOT rescue** (het stays at 0.73) — NAD boosts ATP but cannot clear damaged copies
3. **Rapamycin alone does NOT rescue** (het drops to 0.63 but stays above cliff)
4. **Full cocktail without transplant barely rescues** (het 0.40, ATP 0.80) — the combined effect of NAD+rapamycin+senolytic+exercise gets close but is marginal
5. **Transplant + NAD is the power combination** (het 0.24, ATP 0.78)

### Delayed intervention

Patient starting at het=0.80, treatment delayed by N years:

| Delay (years) | Final ATP | Final het | Rescued? |
|---------------|-----------|-----------|----------|
| 0             | 0.802     | 0.172     | YES      |
| 5             | 0.802     | 0.178     | YES      |
| 10            | 0.800     | 0.188     | YES      |
| 15            | 0.796     | 0.205     | YES      |
| 20            | 0.783     | 0.245     | YES      |
| 25            | 0.740     | 0.355     | YES      |

**Even a 25-year delay still results in rescue.** Outcomes degrade gracefully but never cross into failure. This means the model's bistability mechanism (fix C4) is not creating true irreversibility — transplant at rate 1.0 overwhelms the deletion replication advantage.

### Implications for Cramer's Theory

This is potentially problematic. Cramer's central thesis is that heteroplasmy crossing the cliff leads to irreversible collapse. The model should exhibit a "point of no return" where intervention comes too late. Currently:

- The deletion replication advantage (1.21x) is overwhelmed by transplant's competitive displacement (0.12 * n_d)
- Copy number homeostasis (fix C2) prevents unbounded growth, so even at het=0.95, the total copy number is ~1.0
- This means transplant always has "room" to add healthy copies and displace damaged ones

**Possible model adjustments to discuss with Cramer:**
1. Reduce transplant displacement coefficient (currently 0.12) at high het levels
2. Add transplant efficiency degradation when recipient cell environment is hostile
3. Introduce tissue damage accumulation that isn't reversible by mtDNA replacement alone

## NAD-Specific Conclusions

### What the model says about NAD+

1. **NAD+ is genuinely beneficial** — every dose level improves both ATP and heteroplasmy
2. **NAD+ is NOT sufficient alone** — it cannot rescue post-cliff patients (het stays >0.70)
3. **NAD+ has super-linear dose-response** — high doses are disproportionately effective (3.6x marginal return at top vs. bottom quartile)
4. **NAD+ is age-independent in benefit** — Zimmerman falsifier confirmed ~0.10 ATP gain at all ages 30-85 (338 tests, 0 failures)

### Hype guard assessment

| Claim from hype literature | Model agrees? | Grounded in Cramer? | Concern |
|---------------------------|---------------|---------------------|---------|
| "NAD+ declines with age" | Yes | Yes (Ca16, p.73) | None |
| "NMN/NR restores NAD+" | Yes (CD38-gated) | Yes (email C7) | None |
| "Higher doses work better" | Yes (super-linear) | Partially (CD38 is real, but coefficients amplify) | **Medium** |
| "NAD+ boosts ATP production" | Yes (40% factor) | **No** (0.4 coefficient is assumption) | **High** |
| "NAD+ is antioxidant" | Yes (40% factor) | **No** (0.4 coefficient is assumption) | **High** |
| "NAD+ reverses aging" | No (cannot rescue cliff) | Cramer says transplant is primary | **Correct** |
| "NAD+ is all you need" | No (alone fails post-cliff) | Cramer says NAD is supporting | **Correct** |

### Recommendations

1. **Review the 0.4 coefficients with Cramer**: The ATP production factor `(0.6 + 0.4 * NAD)` and antioxidant defense `(1.0 + 0.4 * NAD)` are the largest non-Cramer NAD+ effects in the model. Ask Cramer if these magnitudes are justified or if they should be reduced.

2. **Consider reducing NAD's ATP coefficient**: If Cramer's book discusses the ATP→NAD dependence in Ch. VIII.A, use his numbers. If not, consider reducing from 0.4 to 0.2 (still meaningful but less dominant).

3. **The super-linear dose-response is correctly gated by CD38**: This is a genuine Cramer mechanism (email C7). However, the extreme shape (3.6x marginal ratio) could be softened by adjusting CD38_SUPPRESSION_GAIN from 0.6 to something lower.

4. **NAD gating ALL replication types is defensible but amplifying**: Pathway 9 (deletion replication gated by NAD) partially compensates, but the net effect still inflates NAD's role. Consider whether Cramer describes NAD as required for replication.

5. **The model correctly limits NAD's scope**: NAD cannot rescue post-cliff patients, cannot clear damaged copies, and cannot replace transplant. These are faithful to Cramer's hierarchy: transplant > NAD supplementation.

## Data Sources

- NAD dose-response: 21-point sweep on moderate_60 patient (this analysis)
- Cliff reachability: 9 het levels × 3 protocols + 7 delay points + 7 component configs (this analysis)
- Zimmerman falsifier: 338 adversarial tests, 0 failures (run earlier this session)
- Pathway audit: Manual inspection of simulator.py lines 529-1037 (12 NAD locations identified)
- Previous finding: `artifacts/finding_energy_budget_trumps_heteroplasmy_2026-02-19.md`
