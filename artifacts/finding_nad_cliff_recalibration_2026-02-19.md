# Finding: NAD+ Coefficient Reduction and Cliff Recalibration

**Date:** 2026-02-19
**Source:** NAD audit follow-up, cliff dynamics investigation
**Branch:** `nad-cliff-fixes`
**Commit:** `5d8ae88`

## Summary

Three coordinated changes restore the simulator's fidelity to Cramer's theory:

1. **NAD coefficient reduction** — Two non-Cramer coefficients (0.4) halved to 0.2, reducing NAD+ effect by 35%
2. **Cliff recalibration for C11** — Deletion het cliff lowered from 0.70 to 0.50, restoring cliff dynamics broken by the C11 mutation type split
3. **Bistability mechanisms** — ATP-gated mitophagy + transplant het penalty create a genuine point of no return

## Problem Statement

The NAD audit (finding_nad_audit_hype_guard_2026-02-19.md) identified two issues:

1. Two modeling-assumption coefficients (not from Cramer) inflated NAD's effect by ~35%
2. The model had no point of no return — even patients at het=0.95 were fully rescued

Investigation revealed a deeper issue: the C11 mutation type split (Feb 17) inadvertently broke the cliff dynamics. When all damage was deletions (pre-C11), the cliff at deletion het=0.70 was reachable. After C11 split damage into deletions (~60%) and point mutations (~40%), deletion het maxed out at ~0.57 (at total het=0.95), never reaching the 0.70 cliff. ATP never collapsed, so mitophagy never failed, so het always dropped.

## Changes Made

### 1. NAD Coefficient Reduction

| Constant | Old | New | Location | Effect |
|----------|-----|-----|----------|--------|
| NAD_ATP_DEPENDENCE | 0.4 | 0.2 | constants.py | 40%→20% of ATP depends on NAD |
| NAD_DEFENSE_BOOST | 0.4 | 0.2 | constants.py | 40%→20% antioxidant defense from NAD |

**Impact on NAD dose-response (moderate_60 patient):**

| NAD dose | Old ATP gain | New ATP gain | Reduction |
|----------|-------------|-------------|-----------|
| 0.25     | +0.014      | +0.012      | 17%       |
| 0.50     | +0.035      | +0.027      | 22%       |
| 0.75     | +0.064      | +0.045      | 29%       |
| 1.00     | +0.100      | +0.066      | 34%       |

The reduction is larger at higher doses because the hype coefficients amplified the super-linear dose-response.

**Side effect:** Patients with low NAD now have *higher* baseline ATP (because their low NAD matters less). A 70-year-old patient at het=0.80 went from ATP=0.654 to ATP=0.778 without treatment. This is because reducing NAD dependence means NAD deficiency is less catastrophic — the correct consequence of reducing hype.

### 2. Cliff Recalibration

| Constant | Old | New | Rationale |
|----------|-----|-----|-----------|
| HETEROPLASMY_CLIFF | 0.70 | 0.50 | Adjusted for deletion-only het (C11) |

The literature value of 0.70 (Rossignol et al. 2003) refers to **total** heteroplasmy. After C11, the cliff uses **deletion** het only, and deletions are ~60% of total damage. The equivalent deletion cliff is 0.70 * 0.60 = 0.42. We use 0.50 as a conservative estimate, accounting for the fact that deletion fraction increases with age.

**Impact on no-treatment outcomes:**

| Total het | Old ATP | New ATP | Cliff active? |
|-----------|---------|---------|---------------|
| 0.50      | 0.807   | 0.753   | Starting to bite |
| 0.70      | 0.794   | 0.498   | **Full collapse** |
| 0.80      | 0.778   | 0.254   | Deep collapse |
| 0.95      | 0.704   | 0.073   | Near-zero |

### 3. Bistability Mechanisms

Two new mechanisms create the positive feedback loop that traps cells past the cliff:

**ATP-gated mitophagy** (`simulator.py`):
```
mitophagy_efficiency = sigmoid(energy_available, midpoint=0.6, steepness=8)
```
Biology: Autophagosome formation requires ATP (membrane nucleation, cargo recognition, lysosomal fusion). At low ATP, PINK1 still flags damaged mitos, but the cell can't execute clearance. This creates: low ATP → impaired mitophagy → damaged mitos persist → ATP stays low.

**Transplant het penalty** (`simulator.py`):
```
transplant_penalty = sigmoid(het_total, midpoint=0.75, steepness=25)
```
Biology: At high total het, the intracellular environment (low ATP, high ROS, SASP) impairs engraftment of transplanted mitochondria. They can't compete effectively in a hostile cell.

## Results: Point of No Return

### Full rescue protocol (transplant=1.0, NAD=1.0, rapamycin=0.75)

| Starting het | Final ATP | Final het | Rescued? |
|-------------|-----------|-----------|----------|
| 0.70        | 0.884     | 0.184     | YES      |
| 0.80        | 0.884     | 0.194     | YES      |
| 0.85        | 0.884     | 0.201     | YES      |
| 0.90        | 0.882     | 0.222     | YES (marginal) |
| **0.95**    | **0.201** | **0.867** | **NO**   |

**The model now has a point of no return at het~0.93-0.95.** Below this, transplant can still rescue. Above it, the cliff dynamics trap the cell.

### Delayed intervention (het=0.80)

| Delay (years) | Final ATP | Final het | Rescued? |
|---------------|-----------|-----------|----------|
| 0             | 0.884     | 0.194     | YES      |
| 10            | 0.881     | 0.211     | YES      |
| 20            | 0.855     | 0.332     | YES      |
| **25**        | **0.527** | **0.670** | **NO**   |

**Delayed rescue now fails.** At het=0.80, waiting 25 years (het rises to ~0.87 by year 25) crosses the point of no return.

### Component isolation (het=0.80)

| Protocol | Final ATP | Final het | Rescued? |
|----------|-----------|-----------|----------|
| No treatment | 0.254 | 0.880 | NO |
| Transplant only | 0.810 | 0.324 | YES |
| NAD only | 0.311 | 0.870 | NO |
| Full rescue | 0.884 | 0.194 | YES |

NAD alone cannot rescue post-cliff patients. Transplant remains the primary mechanism — consistent with Cramer Ch. VIII.G.

## Tests Updated

| Test | Change | Reason |
|------|--------|--------|
| `test_cliff_factor_sigmoid` | Thresholds 0.3/0.7/0.9 → 0.2/0.5/0.7 | Cliff shifted from 0.70 to 0.50 |
| `test_declining_patient_has_crisis` | Check ATP < 0.85 instead of ATP declining | Reduced NAD dependence raises baseline ATP |
| `test_near_cliff_patient` | ATP < 0.8 instead of < 0.7 | Same reason |
| `test_high_damage_collapses` | ATP < 0.80 instead of < 0.65 | Same reason |
| `test_cliff_driven_by_deletions` | ATP < 0.55 instead of < 0.5 | Marginal shift |
| `test_cliff_distance_consistency` | Uses HETEROPLASMY_CLIFF constant | Don't hardcode 0.7 |

All 453 tests pass.

## Discussion

### For Cramer Review

1. **HETEROPLASMY_CLIFF at 0.50**: Is this the right deletion-het threshold? The literature's 70% total het ≈ 42-56% deletion het depending on age. We used 0.50 as a compromise. Cramer may prefer a different value based on the ETC subunit composition of common deletions.

2. **NAD coefficients at 0.2**: The ATP formula is now `(0.8 + 0.2 * NAD)`. This means 80% of ATP comes from glycolysis/TCA (NAD-independent) and 20% from oxphos (NAD-dependent). The biochemistry textbook split is roughly 60/40 for most tissues. Does Cramer have a preference?

3. **The point of no return at het~0.93**: Is this consistent with the clinical literature? The model now says you can rescue a patient at het=0.90 but not at het=0.95. Is this boundary approximately correct?

### Relationship to Energy Budget Finding

The previous finding (energy_budget_trumps_heteroplasmy) showed that het didn't discriminate outcomes in the dark_matter sweep. This was because the old cliff (0.70 deletion het) was unreachable under C11. With the recalibrated cliff (0.50), het should now be a more meaningful predictor. A re-run of the dark_matter sweep through the pipeline would confirm.
