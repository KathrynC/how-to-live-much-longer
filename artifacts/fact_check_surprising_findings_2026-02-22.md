# Fact-Check: Surprising Findings Against Scientific Literature
## 2026-02-22

**Objective**: Validate the 10 surprising findings from the mitochondrial aging simulation (post‑falsification‑fixes) against published scientific evidence.

**Sources**:  
- Compiled literature search (2026‑02‑22) covering 5 falsification findings  
- Cramer book (forthcoming 2026) references in CLAUDE.md  
- General mitochondrial biology knowledge  

---

## 1. Yamanaka reprogramming harms ATP at all ages

**Simulation finding**: Partial reprogramming (OSKM) reduces ATP relative to no treatment across ages 40, 60, 80 (–0.134 to –0.121 ATP).

**Literature check**:
- **Cramer Table 3** (Ch. VIII.A p. 100): Yamanaka factors cost 3–5 MU/day (≈ 30–50 % of baseline ATP).  
- **Ci24, Fo18** (Ch. VII.B p. 95): Reprogramming requires 3–10× energy investment.  
- **Recent studies**: OSKM can rejuvenate epigenetic age but at substantial metabolic cost; net ATP benefit depends on co‑interventions (e.g., NAD+ boost to offset cost).

**Verdict**: **PLAUSIBLE** – The energy cost of reprogramming is well‑documented; whether net ATP gain is possible depends on ancillary support (NAD+, transplant). The simulation’s negative net effect aligns with Cramer’s quantitative estimate.

---

## 2. Exercise reduces heteroplasmy but also reduces ATP

**Simulation finding**: Exercise 1.0 vs 0.0: Δhet = –0.0627 (‑14 %), ΔATP = –0.0086 (‑1.1 %).

**Literature check** (from compiled search):
- **Mancuso et al. (2016)**: Elderly exercisers showed 12 % lower heteroplasmy than sedentary controls.  
- **Vissing et al. (2013)**: Exercise reduced mtDNA deletions by 15 %.  
- **Pinto et al. (2017)**: Moderate exercise increased antioxidant capacity by 50–60 %, outweighing ROS increase, leading to net reduction in oxidative damage.  
- **ATP production**: Most studies report improved mitochondrial respiration and ATP synthesis with exercise (e.g., +20 % respiratory capacity in elderly exercisers).

**Verdict**: **CONTRADICTED** – Literature consistently shows exercise improves mitochondrial function and ATP production. The simulation’s ATP decline suggests the model’s metabolic‑cost parameter (`EXERCISE_METABOLIC_COST=0.02`) may be too high relative to biogenesis/antioxidant benefits.

---

## 3. APOE4 vulnerability extends beyond sleep

**Simulation finding**: APOE4 carriers show lower ATP and higher heteroplasmy than wild‑type across all sleep‑intervention levels.

**Literature check** (from compiled search):
- **Zhang et al. (2017)**: APOE4 carriers had 25 % lower respiratory chain efficiency, 30 % higher ROS.  
- **Chen et al. (2020)**: APOE4 carriers showed 25 % lower ATP production, 40 % higher mitochondrial fragmentation.  
- **Sleep‑specific vulnerability**: APOE4 carriers also have amplified sleep‑related damage (25–40 % greater cognitive decline with poor sleep).

**Verdict**: **CONFIRMED** – APOE4 confers baseline mitochondrial dysfunction independent of sleep. The simulation’s APOE4 amplification (×1.5) aligns with literature‑reported effect sizes.

---

## 4. Sleep interventions have zero effect on ATP

**Simulation finding**: Varying `sleep_intervention` (0.1–0.9) produced no change in final ATP (ΔATP < 1e‑8).

**Literature check** (from compiled search):
- **Zhang et al. (2021)**: Sleep improvement in aged mice increased ATP production by 5–10 %.  
- **Mander et al. (2017)**: One night of total sleep deprivation reduced mitochondrial respiratory capacity by 12 %.  
- **Irwin et al. (2006)**: CBT‑I in elderly improved sleep efficiency by 12 %, with associated metabolic benefits.

**Verdict**: **CONTRADICTED** – Sleep improvement should yield modest ATP gains (5–10 %). The simulation’s neutrality suggests the zero‑point recalibration may have over‑corrected, eliminating the beneficial signal.

---

## 5. Transplant shows strong diminishing returns

**Simulation finding**: Marginal ATP gain saturates after dose 0.5; second‑half of dose range adds only ≈ 0.0015 ATP.

**Literature check** (from compiled search):
- **McCully et al. (2019)**: Dose‑response saturated at ~20 million mitochondria/kg; beyond 50 million, no further benefit.  
- **Emani et al. (2021)**: 15 million mitochondria/kg yielded 40 % improvement; 30 million gave only marginal additional benefit.  
- **Cowan et al. (2020)**: Uptake capacity saturated at ≈ 20 million mitochondria/kg.

**Verdict**: **CONFIRMED** – Clinical studies show clear saturation of transplant benefit at moderate doses. The simulation’s saturation curve is realistic.

---

## 6. NAD supplementation exhibits a threshold effect (CD38 saturation)

**Simulation finding**: Low‑dose NAD (0.25) gain = 0.0102 ATP; high‑dose (1.0) gain = 0.0635 ATP (ratio 6.2×).

**Literature check**:
- **Cramer Ch. VI.A.3 p. 73**: CD38 destroys NMN/NR; only 40 % survives at minimal dose.  
- **Apigenin suppression**: High‑dose regimens often include CD38 inhibitors, raising survival to ~100 %.  
- **Camacho‑Pereira (2016)**: NAD+ decline with age is partially due to CD38 upregulation.

**Verdict**: **PLAUSIBLE** – CD38‑mediated destruction of precursors creates a threshold; supplementation without CD38 inhibition is largely futile. The 6‑fold gain ratio is consistent with CD38 survival increasing from 0.4 to 1.0.

---

## 7. Senolytics provide modest benefit even in extreme patients

**Simulation finding**: In a 90‑year‑old patient with heteroplasmy 0.8, senolytics increased ATP by +0.0093 relative to no treatment.

**Literature check**:
- **Dasatinib+quercetin trials**: Generally safe in elderly; benefit seen in frailty indices, but data in extreme mitochondrial dysfunction are limited.  
- **Senescent‑cell clearance**: Removes inflammatory burden, potentially improving microenvironment for remaining mitochondria.  
- **No literature reports of harm** from senolytics in mitochondrial disease contexts.

**Verdict**: **PLAUSIBLE** – Senolytics are not expected to exacerbate mitochondrial dysfunction; modest benefit via reduced inflammation is biologically plausible. Lack of harm is consistent with clinical safety profile.

---

## 8. Dark‑matter analysis reveals no paradoxical interventions

**Simulation finding**: Random sampling of 70 intervention vectors produced no “paradoxical” outcomes (intervention making patient worse).

**Literature check**:
- **Exercise hormesis**: Moderate doses beneficial, extreme doses harmful (ROS outweighs antioxidants).  
- **Yamanaka energy cost**: Could be net‑harmful without co‑support.  
- **Drug interactions**: Possible antagonistic combinations (e.g., rapamycin + high‑dose exercise in frail elderly).  
- **General principle**: Most interventions in moderate doses are safe; harm emerges at extremes or in specific vulnerabilities.

**Verdict**: **PARTIALLY CONTRADICTED** – Real‑world interventions can be harmful (e.g., overtraining, excessive senolytics in thrombocytopenia). The simulation’s benign parameter space may reflect overly conservative coupling or missing toxicity pathways.

---

## 9. Exercise mitophagy boost channel now active

**Simulation finding**: Added `EXERCISE_MITOPHAGY_BOOST=0.01` to quality‑control term.

**Literature check** (from compiled search):
- **Wang et al. (2018)**: Moderate exercise increased mitophagy markers by 35 %.  
- **Kim et al. (2019)**: Exercise enhanced selective removal of damaged mitochondria by 45 %.  
- **PINK1/Parkin activation**: Exercise increases mitophagy via this pathway (Youle et al. 2016).

**Verdict**: **CONFIRMED** – Exercise stimulates mitophagy; the added channel corrects a previous model omission.

---

## 10. Heteroplasmy reduction does not guarantee ATP improvement

**Simulation finding**: Exercise reduces heteroplasmy (–0.063) yet ATP declines (–0.0086); transplant increases ATP with little het change.

**Literature check**:
- **Cliff factor**: ATP production depends on deletion heteroplasmy via sigmoid cliff (0.50 threshold).  
- **NAD, senescence, membrane potential**: Additional determinants of ATP beyond heteroplasmy.  
- **Clinical correlation**: Heteroplasmy is an imperfect biomarker of function; some patients with high heteroplasmy retain decent ATP via compensatory mechanisms.

**Verdict**: **PLAUSIBLE** – ATP is multi‑factorial; interventions can improve energy output through pathways independent of mtDNA damage clearance (e.g., NAD boost, senolytic clearance, membrane potential support).

---

## Summary Table

| Finding | Literature Support | Notes |
|---------|-------------------|-------|
| 1. Yamanaka harms ATP | **Plausible** (Cramer energy cost) | Net benefit may require co‑interventions |
| 2. Exercise reduces ATP | **Contradicted** | Model metabolic cost likely overstated |
| 3. APOE4 systemic vulnerability | **Confirmed** | Literature shows baseline mitochondrial dysfunction |
| 4. Sleep interventions neutral | **Contradicted** | Should provide 5–10 % ATP gain |
| 5. Transplant saturation | **Confirmed** | Clinical dose‑response shows clear plateau |
| 6. NAD threshold (CD38) | **Plausible** | CD38 destruction creates efficacy threshold |
| 7. Senolytics safe/beneficial | **Plausible** | No evidence of harm; modest benefit plausible |
| 8. No paradoxical interventions | **Partially contradicted** | Real interventions can be harmful at extremes |
| 9. Exercise mitophagy boost | **Confirmed** | Correct model addition |
| 10. Het ≠ ATP correlation | **Plausible** | ATP depends on multiple factors beyond heteroplasmy |

---

## Key Model Calibration Issues

1. **Exercise metabolic cost** (`EXERCISE_METABOLIC_COST=0.02`) appears too high relative to biogenesis/antioxidant benefits. Literature suggests net ATP gain, not loss.

2. **Sleep effect magnitude** may be under‑represented; after zero‑point recalibration, the beneficial signal disappeared entirely. Sleep improvement should yield 5–10 % ATP gain.

3. **Lack of paradoxical interventions** suggests model may be missing toxicity pathways (e.g., ROS‑overwhelm from combined stressors, drug‑drug antagonism).

4. **Yamanaka net harm** is consistent with energy cost but may need co‑intervention coupling (e.g., Yamanaka + NAD boost could be net‑positive).

---

## Recommendations for Model Refinement

1. **Re‑calibrate exercise parameters**: Reduce `EXERCISE_METABOLIC_COST` or increase `EXERCISE_BIOGENESIS_FACTOR`/`EXERCISE_MITOPHAGY_BOOST` to achieve net ATP gain.

2. **Restore sleep benefit**: Adjust `SLEEP_QUALITY_ANCHORS` or coupling coefficients so that sleep intervention yields modest ATP improvement (≈ 5 %).

3. **Introduce toxicity pathways**: Add ROS‑overwhelm or antagonistic interaction terms to allow paradoxical outcomes at extreme doses.

4. **Validate with head‑to‑head literature**: Compare simulated dose‑response curves (transplant, NAD) directly to published quantitative data.

5. **Cross‑check with Cramer book**: Ensure all energy‑cost estimates (Yamanaka, exercise) align with Cramer’s forthcoming quantitative tables.

---

*Generated by analysis of compiled literature search (2026‑02‑22) and project CLAUDE.md references.*  
*Primary source: `artifacts/lit_search/compiled_lit_search_2026-02-22.md`*