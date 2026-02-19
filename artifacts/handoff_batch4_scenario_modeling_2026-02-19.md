# Handoff Batch 4: Scenario Modeling, Clinical Impact Analysis, and Genotype Comparisons

**Date:** 2026-02-19
**Status:** Received, not yet implemented.

## Scope

This batch specifies:
1. A scenario definition/runner/comparison framework for batch simulation
2. Predefined escalating intervention scenarios (A–D) with projected trajectories
3. Incremental transplant contribution analysis for 63yo and 91yo patients
4. Per-intervention clinical impact breakdowns by patient profile
5. APOE4 vs non-carrier comparison at age 63 with identical history

---

## Part 1: Scenario Modeling Framework

### New Files to Create

#### `scenario_definitions.py`
- `InterventionProfile` dataclass with all intervention parameters (core + lifestyle + supplements + sleep + therapy + intellectual)
- `Scenario` dataclass: name, description, patient_params dict, InterventionProfile, duration_years, output_metrics list
- `get_example_scenarios()` function returning predefined scenarios A–D

#### `scenario_runner.py`
- `run_scenario(scenario, years)` → converts InterventionProfile to dict, calls `run_simulation()`, returns DataFrame with `age` column
- `run_scenarios(scenarios, years)` → batch runner returning `Dict[str, DataFrame]`

#### `scenario_analysis.py`
- `extract_milestones(df, name)` → dementia_age, amyloid_pathology_age, het_below_50_age, atp_above_08_age, final values
- `compare_scenarios(results)` → DataFrame of milestones across all scenarios
- `summary_table(results, ages)` → pivot of metrics at specified ages

#### `scenario_plot.py`
- `plot_trajectories(results, metrics)` → multi-panel line plot with dementia threshold line
- `plot_milestone_comparison(milestone_df)` → horizontal bar chart of milestone ages
- `plot_summary_heatmap(summary_df, metric)` → seaborn heatmap of metric×age×scenario

#### `run_scenario_comparison.py`
- Main script: loads scenarios, runs all, prints milestones + summary, saves 3 PNG plots

### Design Notes
- Uses pandas DataFrames for trajectory storage
- Uses matplotlib + seaborn for plotting (Agg backend, output to `output/`)
- InterventionProfile has defaults of 0.0 for all parameters — scenarios only set non-zero values
- `run_simulation()` must accept all new intervention parameters and return list of state dicts

---

## Part 2: Predefined Scenarios (A–D)

Base patient: 63-year-old female, APOE4 heterozygous, post-menopause, grief_intensity 0.18, grief_duration 10yr, doctoral education, intellectual_engagement 0.9, baseline heteroplasmy 62%, ATP 0.55, memory_index 0.84.

| Scenario | Components | Key Parameters |
|----------|-----------|----------------|
| A | Sleep + alcohol cessation | sleep_intervention=0.8, oura=True, alcohol=0.0 |
| B | A + OTC supplements + keto + probiotics | + NR/DHA/CoQ10/resveratrol/PQQ/ALA/vitD/B/Mg/Zn/Se all 0.7-1.0, diet=keto, probiotic=0.8 |
| C | B + prescription | + rapamycin=0.8, senolytic=0.8 |
| D | C + experimental | + transplant=0.9, yamanaka=0.5 (ATP-gated) |

### Projected Trajectories

#### Heteroplasmy (%)
| Age | A | B | C | D |
|-----|---|---|---|---|
| 63 | 62 | 62 | 62 | 62 |
| 70 | 68 | 55 | 50 | 45 |
| 80 | 75 | 45 | 35 | 28 |
| 90 | 82 | 38 | 28 | 20 |
| 95 | 85 | 40 | 30 | 18 |

#### ATP (MU/day)
| Age | A | B | C | D |
|-----|---|---|---|---|
| 63 | 0.55 | 0.55 | 0.55 | 0.55 |
| 70 | 0.50 | 0.75 | 0.85 | 0.92 |
| 80 | 0.40 | 0.85 | 0.92 | 0.96 |
| 90 | 0.30 | 0.80 | 0.88 | 0.94 |
| 95 | 0.25 | 0.75 | 0.85 | 0.92 |

#### Memory Index
| Age | A | B | C | D |
|-----|---|---|---|---|
| 63 | 0.84 | 0.84 | 0.84 | 0.84 |
| 70 | 0.75 | 0.88 | 0.92 | 0.95 |
| 80 | 0.60 | 0.85 | 0.90 | 0.94 |
| 90 | 0.48 (MCI) | 0.80 | 0.88 | 0.92 |
| 95 | 0.40 (dementia) | 0.75 | 0.85 | 0.90 |

#### Key Milestones
| Milestone | A | B | C | D |
|-----------|---|---|---|---|
| Het < 50% | Never | 72 | 68 | 65 |
| ATP > 0.8 | Never | 68 | 66 | 64 |
| Memory < 0.5 (dementia) | 88 | Never | Never | Never |
| Amyloid > 1.0 (pathological) | 82 | 92 | Never | Never |

---

## Part 3: Incremental Transplant Contribution

### 63yo Female APOE4 (on Scenario C baseline)
| Metric | C (no transplant) at 95 | C + Transplant at 95 | Delta |
|--------|------------------------|---------------------|-------|
| Heteroplasmy | 30% | 22% | -8 pp |
| ATP | 0.85 | 0.90 | +0.05 |
| Memory index | 0.85 | 0.89 | +0.04 |
| Clinical impact | — | — | +3-5 years healthy aging |

### 91yo Non-Carrier (on optimized regimen)
| Metric | Without at 95 | With at 95 | Delta |
|--------|--------------|-----------|-------|
| Heteroplasmy | 60% | 50% | -10 pp |
| ATP | 0.75 | 0.82 | +0.07 |
| Memory index | 0.70 | 0.76 | +0.06 |
| Clinical impact | — | — | +2-4 years independent living |

### 91yo APOE4 Carrier (on optimized regimen)
| Metric | Without at 95 | With at 95 | Delta |
|--------|--------------|-----------|-------|
| Heteroplasmy | 82% | 70% | -12 pp |
| ATP | 0.40 | 0.55 | +0.15 |
| Memory index | 0.45 | 0.58 | +0.13 |
| Clinical impact | — | — | Restores independence from severe impairment |

**Key insight:** Transplant benefit is inversely proportional to starting health — APOE4 91yo derives the largest absolute benefit.

---

## Part 4: Per-Intervention Clinical Impact Tiers

### 63yo Female APOE4 Carrier

#### Tier 1: Immediate & Foundational
1. **Alcohol cessation** — 38% ATP increase within 6mo, 50% inflammation reduction, improved clarity. Life-changing.
2. **Sleep optimization (Oura)** — 28% next-day energy improvement, 35% brain fog reduction. Foundational. 2-4 weeks.
3. **Grief therapy + support** — 55% faster grief resolution, reduced cortisol, better adherence. Essential. 3-6 months.

#### Tier 2: Major Lifestyle & OTC
4. **Keto + IF** — 15% ATP increase in 3mo, mental sharpness. Transformative. 2-8 weeks.
5. **NR 1000mg/day** — 2x NAD+ boost, sustained energy, no afternoon crashes. 4-8 weeks.
6. **DHA 2-3g + CoQ10 300mg** — 20% less joint pain, cognitive endurance, cardiovascular protection. 3-6 months.
7. **Probiotics (Lactobacillus)** — reduced GI distress, 15% mood improvement, NAD+ conversion support. 4-8 weeks.
8. **Bacopa 300mg + L-Theanine 400mg** — 18% stress reduction, improved verbal fluency, better sleep. 4-12 weeks.
9. **Coffee 1-3 cups morning** — 5-10% cognitive boost, trigonelline NAD+ boost. Immediate.

#### Tier 3: Prescription
10. **Rapamycin 0.8** — 30% heteroplasmy reversal acceleration, immune modulation. 6-12 months.
11. **Enhanced senolytics** — 25% reduced joint stiffness, faster recovery. 3-6 months.

#### Tier 4: Experimental
12. **Mitochondrial transplant** — h 30%→22%, +0.05 ATP, +0.04 memory by 95. 3-5 extra years sharpness.
13. **Yamanaka factors** — epigenetic rejuvenation, requires ATP>0.8. Frontier.

### 91yo Non-Carrier
Tiers 1-2 focus on energy preservation and fall prevention. Alcohol reduction (not full cessation). NR 500-750mg. Magnesium for sleep/cramps. Rapamycin cautious at 0.5. Transplant provides 2-4 extra years independence.

### 91yo APOE4 Carrier
**Complete alcohol cessation non-negotiable.** Aggressive sleep intervention. NR 1000mg from very low baseline. APOE4-targeted probiotics. Rapamycin gentle at 0.3. Transplant is transformative — lifts from frailty/dementia range to functional independence (ATP 0.40→0.55, memory 0.45→0.58).

---

## Part 5: APOE4 vs Non-Carrier Comparison at Age 63

Both: identical life history (grief, alcohol, intellectual transformation). Same interventions (Scenario C).

### Baseline Differences at 63
| Parameter | APOE4 | Non-Carrier | Why |
|-----------|-------|-------------|-----|
| Heteroplasmy | 62% | 54% | APOE4 amplifies alcohol/stress damage |
| ATP | 0.55 | 0.73 | Lower NAD+, impaired mitophagy |
| Inflammation | 0.466 | 0.35 | Heightened inflammatory response |
| NAD+ | 0.45 | 0.52 | APOE4 × alcohol synergy depletes NAD+ |
| Mitophagy efficiency | 0.52 | 0.72 | APOE4 impairs mitophagy |
| Amyloid burden | 0.30 | 0.22 | Reduced clearance |
| Memory index | 0.84 | 0.88 | Both high from CR, slight APOE4 deficit |

### Projected Trajectories (Scenario C)

#### Heteroplasmy (%)
| Age | APOE4 | Non-Carrier | Gap |
|-----|-------|-------------|-----|
| 63 | 62 | 54 | 8 |
| 70 | 50 | 42 | 8 |
| 80 | 35 | 28 | 7 |
| 90 | 28 | 22 | 6 |
| 95 | 30 | 20 | 10 |

#### ATP (MU/day)
| Age | APOE4 | Non-Carrier | Gap |
|-----|-------|-------------|-----|
| 63 | 0.55 | 0.73 | 0.18 |
| 70 | 0.85 | 0.92 | 0.07 |
| 80 | 0.92 | 0.96 | 0.04 |
| 90 | 0.88 | 0.94 | 0.06 |
| 95 | 0.85 | 0.93 | 0.08 |

#### Memory Index
| Age | APOE4 | Non-Carrier | Gap |
|-----|-------|-------------|-----|
| 63 | 0.84 | 0.88 | 0.04 |
| 70 | 0.92 | 0.95 | 0.03 |
| 80 | 0.90 | 0.94 | 0.04 |
| 90 | 0.88 | 0.93 | 0.05 |
| 95 | 0.85 | 0.92 | 0.07 |

#### Amyloid Burden
| Age | APOE4 | Non-Carrier | Gap |
|-----|-------|-------------|-----|
| 63 | 0.30 | 0.22 | 0.08 |
| 80 | 0.55 | 0.40 | 0.15 |
| 95 | 0.85 | 0.60 | 0.25 |

### Milestones
| Milestone | APOE4 | Non-Carrier | Difference |
|-----------|-------|-------------|------------|
| Het < 50% | 68 | 66 | 2 years earlier |
| ATP > 0.8 | 68 | 66 | 2 years earlier |
| Amyloid > 1.0 | Never (max 0.85) | Never (max 0.60) | — |
| Memory < 0.5 | Never | Never | — |
| Memory at 95 | 0.85 | 0.92 | 7 points higher |

### The APOE4 "Tax"
Despite identical interventions:
- 7-point lower memory at 95
- 25% higher amyloid burden
- 0.08 lower ATP
- 10pp higher heteroplasmy

But the gap **narrows** mid-trajectory (ATP gap shrinks from 0.18→0.04 at age 80) showing interventions are proportionally more effective for APOE4 carriers. Without interventions, the gap would widen dramatically.

### Clinical Translation
- APOE4 carriers need earlier, stricter adherence for equivalent outcomes
- Non-carriers have a larger safety margin for occasional lapses
- Every intervention is qualitatively similar but quantitatively larger for APOE4
- APOE4 + optimal interventions ≈ non-carrier a decade younger

---

## Implementation Priority

1. `scenario_definitions.py` with InterventionProfile and Scenario dataclasses
2. `scenario_runner.py` with batch simulation
3. `scenario_analysis.py` with milestone extraction and comparison
4. `scenario_plot.py` with trajectory plots, milestone bars, heatmaps
5. `run_scenario_comparison.py` main script
6. Test cases in `tests/test_scenarios.py`

All depend on the expanded `run_simulation()` from batch 3 being implemented first.
