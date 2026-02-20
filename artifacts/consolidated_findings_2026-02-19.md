# Consolidated Findings: Model Validation Sprint (Feb 15-19, 2026)

**Date:** 2026-02-19
**Scope:** ODE bug fixes, Cramer corrections, NAD audit, cliff recalibration, sleep pathway activation, APOE4 wiring, robustness sweep, literature validation

## Executive Summary

Over five days, the mitochondrial aging simulator underwent a comprehensive validation cycle: adversarial review, author corrections, coefficient audit, pathway activation, and robustness analysis. The model went from 7-state/81-test to 8-state/453-test, with four critical ODE bugs fixed, five author corrections applied, two hype-literature coefficients reduced, and two disconnected pathways wired up. The result is a simulator whose dynamics are faithful to Cramer's book, robust under 25 biological stress scenarios, and honest about its modeling assumptions.

## Phase 1: Adversarial Review (Feb 15)

**Source:** `artifacts/falsifier_report_2026-02-15.md`

The falsifier agent found 4 critical bugs that collectively meant the model's central claim (heteroplasmy cliff causes irreversible ATP collapse) was cosmetic, not dynamical:

| Bug | Problem | Fix |
|-----|---------|-----|
| C1: Cliff cosmetic | Cliff didn't feed back into replication/apoptosis | ATP collapse now halts replication |
| C2: Unbounded N_h | Transplant grew copy number to 3.96 | Homeostatic regulation toward total ~1.0 |
| C3: NAD inverted | NAD boosted damaged replication more than healthy | Selective quality control for healthy copies |
| C4: Universal attractor | All initial conditions converged to het~0.60 | Bistability via replication advantage |

Plus 5 major issues (M1-M5) and 6 minor issues. All fixed same day. Tests: 81 → 85.

## Phase 2: Cramer Author Corrections (Feb 15-17)

**Source:** `artifacts/cramer_corrections_2026-02-15.md`

John Cramer reviewed the simulation and provided two critiques, leading to five corrections:

| Correction | Change | Source |
|------------|--------|--------|
| C7: CD38 gating | NAD boost gated by CD38 survival factor (0.4 + 0.6*dose) | Cramer Ch. VI.A.3 p.73 |
| C8: Transplant primary | Addition rate doubled (0.15→0.30), competitive displacement added | Cramer Ch. VIII.G pp.104-107 |
| C9: AGE_TRANSITION | Restored from 40 to 65 years | Cramer Appendix 2 p.155 |
| C10: Dynamic transition | Fixed age replaced with ATP+mitophagy-coupled sigmoid | Cramer email |
| C11: Mutation split | N_damaged split into N_deletion + N_point (7D→8D) | Cramer Appendix 2 pp.152-155 |

Key outcome: transplant benefit (0.416 het reduction) is 2.3x NAD benefit (0.183), confirming Cramer's assertion that transplant is the primary rejuvenation mechanism.

## Phase 3: NAD Audit and Hype Guard (Feb 19)

**Source:** `artifacts/finding_nad_audit_hype_guard_2026-02-19.md`

Audited 12 locations where NAD+ influences ODE dynamics:
- **4 grounded in Cramer** (NAD boost, CD38 gating, mitophagy boost, age decline)
- **8 modeling assumptions** (ATP production, antioxidant defense, replication gating, etc.)

Two non-Cramer coefficients (both 0.4) inflated NAD's therapeutic effect:

| Coefficient | Old | New | Effect |
|------------|-----|-----|--------|
| NAD_ATP_DEPENDENCE | 0.4 | 0.2 | 40%→20% of ATP depends on NAD |
| NAD_DEFENSE_BOOST | 0.4 | 0.2 | 40%→20% antioxidant defense from NAD |

NAD gain at max dose fell from +0.100 to +0.066 ATP (34% reduction). The super-linear dose-response (CD38 mechanism) remains, but its magnitude is reduced.

## Phase 4: Cliff Recalibration (Feb 19)

**Source:** `artifacts/finding_nad_cliff_recalibration_2026-02-19.md`

The C11 mutation split broke cliff dynamics: deletion het maxed at ~0.57 (at total het=0.95), never reaching the old 0.70 cliff. Literature value 0.70 (Rossignol 2003) refers to total het; with deletions ~60% of total, equivalent deletion cliff is ~0.42-0.56.

| Constant | Old | New | Rationale |
|----------|-----|-----|-----------|
| HETEROPLASMY_CLIFF | 0.70 | 0.50 | Deletion-only equivalent of literature 0.70 total het |

Bistability restored via ATP-gated mitophagy + transplant het penalty. Point of no return now at het~0.93-0.95. Delayed rescue fails after ~25 years at het=0.80.

**For Cramer review:** Is 0.50 the right deletion threshold? Is het~0.93 point of no return clinically realistic?

## Phase 5: Energy Budget Finding (Feb 19)

**Source:** `artifacts/finding_energy_budget_trumps_heteroplasmy_2026-02-19.md`

Analysis of 700 random intervention protocols through the pipeline revealed that heteroplasmy is NOT a useful outcome predictor. Three-tier outcome hierarchy:

1. **Yamanaka energy cost** (dominant) — Yamanaka OFF: 0/116 declining. Yamanaka HIGH: 162/380 declining.
2. **NAD supplementation** (secondary) — Among energy-neutral protocols, thriving vs stable split by NAD (0.89 vs 0.34).
3. **Heteroplasmy** (tertiary) — Het nearly identical across all classes (0.20-0.25). Correlation r=-0.13.

After cliff recalibration, het becomes a secondary predictor for declining patients (het=0.47 vs 0.25 for thriving), but the finding is nuanced, not invalidated.

## Phase 6: Sleep Pathway Activation (Feb 19)

**Source:** `artifacts/finding_sleep_coefficient_audit_2026-02-19.md`

Audit revealed:
- `SLEEP_DISRUPTION_IMPACT = 0.7` was imported but never used (dead code)
- Section header falsely attributed coefficients to "UVM LEMURS"
- Coefficients originated from a ChatGPT conversation, not published research

Fixes applied:
1. Activated SLEEP_DISRUPTION_IMPACT as mitophagy repair modifier (0.7→0.5, literature-approximated)
2. Sleep and alcohol now independent efficacy variables with secondary interaction
3. Fixed LEMURS/grief conflation in constants.py

**Awaiting:** LEMURS Oura ring data from Dodds & Danforth (3 specific coefficient requests documented in CLAUDE.md).

## Phase 7: APOE4 Vulnerability Wiring (Feb 19)

Two disconnected pathways found and fixed:

1. **mitophagy_efficiency** defined in GENOTYPE_MULTIPLIERS (0.65 het, 0.45 hom) but never exported by `compute_genetic_modifiers()` — dead data
2. **Step 9 ordering bug** — core schedule passthrough (`max()`) overwrote sleep-modified rapamycin_dose

After fixes, proper genotype-dependent vulnerability:

| Genotype | Repair Loss (poor sleep+alcohol) | Inflammation |
|----------|--------------------------------|--------------|
| Wildtype | 40% | 0.515 |
| APOE4 het | 62% | 0.603 |
| APOE4 hom | 89% | 0.677 |

APOE4 homozygotes lose nearly all repair capacity under sleep disruption + alcohol — biologically plausible and consistent with literature on APOE4 vulnerability.

## Phase 8: Robustness Sweep (Feb 19)

**Source:** `artifacts/kcramer_tools_resilience_default_2026-02-19.json`

25 biological stress scenarios x 5 intervention protocols = 125 simulations.

### Robustness Scores (higher = more stress-resistant)

| Protocol | Robustness | Worst-Case ATP | Average ATP |
|----------|-----------|----------------|-------------|
| aggressive | 0.995 | 0.776 | 0.802 |
| transplant_focused | 0.994 | 0.811 | 0.841 |
| moderate | 0.994 | 0.810 | 0.839 |
| conservative | 0.952 | 0.441 | 0.776 |
| no_treatment | 0.919 | 0.339 | 0.730 |

### Key Findings

**Rankings depend on metric:**
- **Worst-case (maximin):** transplant_focused > moderate > aggressive > conservative > none
- **Average:** transplant_focused > moderate > aggressive > conservative > none
- **Robustness score:** aggressive > transplant_focused > moderate > conservative > none

**Transplant_focused has ZERO regret** — it is the best or tied-best protocol under every scenario. This validates Cramer's thesis that transplant is the primary rejuvenation mechanism.

**Most damaging scenarios** (across all protocols):
1. worst_case_patient (het=0.75, vuln=2.0, demand=2.0, infl=0.8)
2. past_cliff (het=0.75)
3. two_decades_older (+20 years)
4. near_cliff_vulnerable (het=0.65, vuln=1.5x)

**NAD scenarios have minimal impact** — even critical_nad_crisis (set NAD=0.2) barely affects protocols. This is consistent with the NAD audit: after coefficient reduction, NAD is supporting, not primary.

**Aging scenarios dominate for strong protocols** — for moderate/aggressive/transplant, the top vulnerability is always age-related (+10yr, +20yr, accelerated aging). The model correctly captures that time is the primary enemy.

## Phase 9: Literature Validation (Feb 19)

**Source:** `artifacts/lit_spider_report.md`

All 26 registered parameters searched against PubMed (keyword-only mode, 363 abstracts fetched, 44 seconds):

| Category | Count | Parameters |
|----------|-------|------------|
| Well-supported | 7 | ros_damage_coupling, ros_relaxation_time, senolytic_clearance, heteroplasmy_cliff, damaged_replication_advantage, tissue_biogenesis_brain, cd38_base_survival |
| Conflicting | 10 | base_replication_rate, apoptosis_rate, exercise_biogenesis, atp_relaxation_time, nad_relaxation_time, cliff_steepness, tissue_ros_cardiac, tissue_biogenesis_muscle, doubling_times, nad_decline, baseline_ros |
| Sparse | 4 | nad_quality_control_boost, senescence_ros_multiplier, yamanaka_repair_rate, ros_per_damaged |
| No data | 3 | rapamycin_mitophagy_boost, tissue_ros_brain, senescence_rate |
| **Major discrepancy** | 2 | **senescence_ros_multiplier, cliff_steepness** |

**Caveat:** Keyword-only extraction has low confidence (no semantic understanding). The "conflicting" assessments often reflect unit mismatches between extracted values and model parameters. LLM-enabled extraction (requires Ollama) would improve accuracy significantly.

**Major discrepancy details:**
- `senescence_ros_multiplier` (current: 2.0) — only 1 paper found, extracted value 0.1 (but keyword extraction, likely not comparable)
- `cliff_steepness` (current: 15.0) — explicitly noted as "simulation calibration, not from book"

## Calibration Status

| Constant | Value | Source | Status |
|----------|-------|--------|--------|
| HETEROPLASMY_CLIFF | 0.50 | Rossignol 2003 (recalibrated for C11) | Review with Cramer |
| NAD_ATP_DEPENDENCE | 0.2 | Modeling assumption (reduced from 0.4) | Review with Cramer |
| NAD_DEFENSE_BOOST | 0.2 | Modeling assumption (reduced from 0.4) | Review with Cramer |
| NATURAL_HEALTH_REF | 0.91 | Iterative calibration (2 iterations) | Stable |
| SLEEP_DISRUPTION_IMPACT | 0.5 | Literature-approximated (Irwin 2016) | Awaiting LEMURS data |
| ALCOHOL_SLEEP_DISRUPTION | 0.4 | Modeling assumption | Awaiting LEMURS data |
| DELETION_REPLICATION_ADVANTAGE | 1.21 | Cramer Appendix 2 "at least 21%" | Verified |
| CD38_BASE_SURVIVAL | 0.4 | Cramer Ch. VI.A.3 p.73 | Well-supported |
| TRANSPLANT_ADDITION_RATE | 0.30 | Cramer Ch. VIII.G (doubled per email C8) | Author-confirmed |

## Test Suite

454 tests across 26 modules, all passing. Growth over validation sprint:

| Date | Tests | Milestone |
|------|-------|-----------|
| Feb 15 (start) | 81 | Pre-falsifier |
| Feb 15 (end) | 85 | Falsifier fixes |
| Feb 15 (Cramer) | 262 | C7-C9 corrections + expansion tests |
| Feb 17 (C11) | 453 | Mutation split + precision medicine |
| Feb 19 (current) | 454 | Sleep/APOE4 fixes |

## Pending External Reviews

1. **John Cramer:** HETEROPLASMY_CLIFF=0.50, NAD coefficients at 0.2, point of no return at het~0.93
2. **Dodds & Danforth (LEMURS):** Sleep→recovery, alcohol→sleep, sleep→inflammation coefficients

## Deferred Analysis (overnight)

- Zimmerman 14-tool suite re-run (n_base=256, ~20 min)
- Full Sobol sensitivity with recalibrated parameters
- Script ready: `bash run_overnight.sh`
