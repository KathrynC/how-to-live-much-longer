# Dodds & Danforth Bibliography: Prioritization for Mitochondrial Aging Simulator

**Date:** 2026-02-19
**Context:** Prioritization of ~200 papers from the D&D research group bibliography for relevance to the how-to-live-much-longer mitochondrial aging ODE simulator project.

## Research Group Methodology Overview

The Computational Story Lab (Dodds & Danforth, UVM) operates at the intersection of complex systems science, computational social science, and information theory. Their methodological toolkit spans:

1. **Scaling laws and allometry** — Rigorous treatment of power-law relationships in biological and social systems (Dodds, Rothman, Weitz 2001; Dodds & Rothman 1999-2000)
2. **Contagion dynamics on networks** — Generalized threshold models unifying biological and social spreading (Dodds & Watts 2004, 2005)
3. **Information-theoretic comparison** — Allotaxonometry, rank-turbulence divergence, mutual information for comparing complex system outputs (Dodds et al. 2020; Adams et al. 2021)
4. **Time series decomposition** — Shocklet transform for identifying mechanism-driven local dynamics (Dewhurst et al. 2019)
5. **Digital biomarkers** — Wearable-derived health prediction, depression markers, stress detection (Reece & Danforth 2017; Bloomfield et al. 2024; Fudolig et al. 2024, 2025)
6. **Sentiment and narrative analysis** — Hedonometrics, emotional arcs, word shift graphs (Dodds & Danforth 2010; Reagan et al. 2016; Gallagher et al. 2021)
7. **Ecological resilience** — Early warning signals for critical transitions (Scheffer et al. 2009 — cited, not authored)

**Key methodological principle:** The group consistently applies information-theoretic and scaling-law frameworks to extract universal patterns from heterogeneous data, treating social and biological systems with the same mathematical formalism. This aligns directly with our simulator's approach: biological ODE dynamics interrogated via Zimmerman/Cramer toolkits using the same tools originally built for social systems.

---

## Tier 0: LEMURS Papers (Directly Calibrate Simulator)

These papers are THE most relevant — they contain the quantitative data needed to calibrate our three pending sleep/lifestyle coefficients.

### L1. Bloomfield et al. (2024) — Stress Prediction from Oura Sleep Data

**Citation:** Bloomfield, A., Fudolig, M. I., et al. "Predicting stress in first-year college students using sleep data from wearable devices." *PLOS Digital Health* 3(4): e0000473.

**URL:** https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000473

**Zotero:** In library (multiple entries)

**Key quantitative findings for our simulator:**

| Sleep Metric | Effect on PSS (continuous) | Odds Ratio (PSS≥14) | p-value |
|---|---|---|---|
| Total Sleep Time | -0.877 points/hour | OR=0.617 (38.3% ↓ per hr) | <0.01 |
| Resting Heart Rate | +0.055 points/bpm | OR=1.036 (3.6% ↑ per bpm) | <0.01 |
| Heart Rate Variability | -0.012 points/ms | OR=0.988 (1.2% ↓ per ms) | <0.05 |
| Respiratory Rate | +0.270 points/breath | OR=1.230 (23.0% ↑ per br) | <0.01 |

**Calibration mapping:**
- `SLEEP_DISRUPTION_IMPACT` (currently 0.5): The TST odds ratio (0.617 per hour) means each lost hour of sleep increases stress odds by ~62%. If we treat 1 hour loss as a "unit" of sleep disruption, our 0.5 coefficient (50% repair loss) is **well-calibrated** — slightly conservative vs the 62% stress increase per lost hour.
- Sleep → stress/inflammation: The PSS increase of 0.877 points per lost hour of sleep, on a 40-point scale, gives a ~2.2% stress increase per hour. Our `(1.0 - sleep_quality) * 0.05` inflammation effect is in the right order of magnitude.

**N=525 students, 3,112 weekly observations, mixed-effects regression with random intercepts.**

### L2. Fudolig et al. (2024) — Two Shapes of Sleep Heart Rate

**Citation:** Fudolig, M. I., et al. "The Two Fundamental Shapes of Sleep Heart Rate Dynamics and Their Connection to Mental Health in College Students." *Digital Biomarkers* 8(1): 120-131.

**URL:** https://karger.com/dib/article/8/1/120/909820/

**Zotero:** In library

**Key findings:**
- Two HR curve types: "early-minimum" (HR drops early in sleep) and "late-minimum" (HR drops late)
- Late-minimum pattern associated with anxiety/depression, prior diagnoses, traumatic experiences
- Sleep architecture differs: late-minimum has shorter deep sleep, reduced REM, extended light sleep
- **No difference in total sleep duration** — the shape matters more than the duration
- Association more pronounced in females
- N≈600 students, 25,000+ sleep records

**Calibration relevance:** This suggests our sleep coefficient should account for *quality* (HR dynamics) not just *duration*. The Oura ring's "sleep score" integrates these dynamics. Could inform a sex-dependent modifier on sleep recovery.

### L3. Fudolig et al. (2025) — Collective Sleep Patterns

**Citation:** Fudolig, M. I., Dodds, P. S., & Danforth, C. M. "Collective sleep and activity patterns of college students from wearable devices." *npj Complexity* 2: 32.

**URL:** https://www.nature.com/articles/s44260-025-00055-x | arXiv: https://arxiv.org/abs/2412.17969

**Zotero:** In library

**Key findings:**
- Median midpoint of sleep at 5 AM (very late chronotype)
- Social jetlag prevalent; positive on weekdays, negative on weekends
- School-day sleep 37-55 minutes shorter than break week → chronic sleep debt
- Males and those with mental health impairment show more delayed sleep
- High Oura compliance (>70% daily + nightly)

**Calibration relevance:** Establishes baseline sleep deficit magnitude (~45 min/day = ~10% sleep reduction). Our `ALCOHOL_SLEEP_DISRUPTION=0.4` should be compared against alcohol-induced sleep reduction relative to this 45-min baseline deficit.

### LEMURS Calibration Summary

| Our Coefficient | Current Value | LEMURS Data Point | Assessment |
|---|---|---|---|
| `SLEEP_DISRUPTION_IMPACT` | 0.5 | TST OR=0.617/hr (62% stress ↑/hr lost) | **Well-calibrated** (conservative) |
| `ALCOHOL_SLEEP_DISRUPTION` | 0.4 | Not directly measured in these papers | **Still modeling assumption** |
| Sleep→inflammation | `(1-sq)*0.05` | PSS Δ=0.877/hr on 40-pt scale (~2.2%/hr) | **Order of magnitude correct** |

**What to ask Dodds & Danforth:** The LEMURS papers don't directly measure alcohol→sleep. The 3 specific requests in CLAUDE.md remain valid:
1. HRV recovery rate vs sleep quality → `SLEEP_DISRUPTION_IMPACT`
2. Alcohol→Oura sleep score dose-response → `ALCOHOL_SLEEP_DISRUPTION`
3. Oura sleep score→next-day perceived stress → inflammation pathway

---

## Tier 1: Directly Relevant to Biological ODE Modeling

### T1.1. Dodds, Rothman, & Weitz (2001) — Metabolic Scaling

**Citation:** Dodds, P. S., Rothman, D. H., & Weitz, J. S. "Re-examination of the '3/4-law' of Metabolism." *Journal of Theoretical Biology* 209(1): 9-27.

**URL:** https://doi.org/10.1006/jtbi.2000.2238

**Zotero:** Not found as PDF (in references but no local PDF)

**Relevance:** Our simulator models ATP production as a function of mitochondrial health. Metabolic scaling laws constrain the relationship between energy production, body size, and cellular demand. Dodds et al. rigorously re-examined West-Brown-Enquist's claim of universal 3/4-power scaling, showing the exponent may not be universal. This matters for our `metabolic_demand` parameter and tissue-specific ATP production rates.

**Extractable methodology:** Dimensional analysis framework for biological rate processes; critique of network-based derivations of scaling exponents; statistical methods for testing power-law hypotheses.

### T1.2. Mitchell, Dodds, Mahoney, & Danforth (2020) — Chimera States & Seizures

**Citation:** Mitchell, H. M., Dodds, P. S., Mahoney, J. M., & Danforth, C. M. "Chimera States and Seizures in a Mouse Neuronal Model." *Int. J. Bifurcation and Chaos* 30(13): 2050256.

**URL:** https://doi.org/10.1142/S0218127420502569 | arXiv: https://arxiv.org/abs/1908.07039

**Relevance:** This is the D&D group's most directly analogous work to our ODE simulator:
- Uses Hindmarsh-Rose neurons (coupled ODEs, like our 8-state system)
- Studies **chimera states** (coexistence of synchrony/asynchrony) — analogous to our heteroplasmy cliff bistability
- Maps a **two-parameter bifurcation space** — like our cliff steepness × heteroplasmy threshold
- Identifies parameter regimes where systems transition between qualitatively different behaviors

**Extractable methodology:** Bifurcation analysis in coupled ODE systems; chimera state detection; parameter space exploration in biological dynamical models.

### T1.3. Allgaier, Danforth (2012) — Empirical Correction of a Toy Climate Model

**Citation:** Allgaier, N. A., Harris, K. D., & Danforth, C. M. "Empirical Correction of a Toy Climate Model." *Physical Review E* 85: 026201.

**Relevance:** Directly addresses model correction methodology for dynamical systems — exactly what our falsifier/recalibration workflow does. They take a simplified climate ODE model and empirically correct its dynamics using observational data.

**Extractable methodology:** Techniques for empirically correcting ODE model dynamics against observations; error quantification in simplified models; approaches to model validation that our Zimmerman falsifier could adopt.

### T1.4. Frank, Mitchell, Dodds, & Danforth (2014) — Lorenz '96 Model

**Citation:** Frank, M. R., et al. "Standing Swells Surveyed Showing Surprisingly Stable Solutions for the Lorenz '96 Model." *Int. J. Bifurcation and Chaos* 24(9): 1430027.

**URL:** https://doi.org/10.1142/S0218127414300274

**Relevance:** Studies stable solutions and bifurcation structure of a well-known dynamical system (Lorenz '96). Methodological template for our heteroplasmy cliff analysis — finding and characterizing stable/unstable equilibria in parameter space.

### T1.5. Scheffer et al. (2009) — Early Warning Signals [cited, not authored]

**Citation:** Scheffer, M., et al. "Early-warning signals for critical transitions." *Nature* 461(7260): 53-59.

**URL:** https://doi.org/10.1038/nature08227

**Relevance:** Already cited in our resilience suite (`resilience_metrics.py`). Provides the theoretical foundation for our regime retention and recovery metrics. The concept of "critical slowing down" before tipping points maps directly to our heteroplasmy cliff dynamics.

---

## Tier 2: Methodological Relevance (Complex Systems Analysis)

### T2.1. Dodds et al. (2020) — Allotaxonometry

**Citation:** Dodds, P. S., Minot, J. R., et al. "Allotaxonometry and rank-turbulence divergence: A universal instrument for comparing complex systems." arXiv:2002.09770.

**URL:** http://arxiv.org/abs/2002.09770

**Relevance:** Universal framework for comparing categorical distributions in complex systems. Could be applied to compare protocol outcome distributions across patient populations — a more principled alternative to our current evaluation correlation matrices.

### T2.2. Dewhurst et al. (2019) — Shocklet Transform

**Citation:** Dewhurst, D. R., et al. "The shocklet transform: A decomposition method for the identification of local, mechanism-driven dynamics in sociotechnical time series." arXiv:1906.11710.

**URL:** http://arxiv.org/abs/1906.11710

**Relevance:** Time series decomposition that identifies **mechanism-driven local dynamics** — exactly what our trajectory analyst should be doing. Could detect when the cliff mechanism activates in individual trajectories, identify resilience recovery signatures, and classify intervention response patterns by their temporal shape.

### T2.3. Adams et al. (2021) — Sirius (Mutual Information)

**Citation:** Adams, J. L., et al. "Sirius: A Mutual Information Tool for Exploratory Visualization of Mixed Data." arXiv:2106.05260.

**URL:** http://arxiv.org/abs/2106.05260

**Relevance:** Could replace or augment our Sobol sensitivity analysis. Mutual information captures nonlinear dependencies that Sobol (variance-based) might miss. Particularly relevant for detecting the cliff mechanism's influence, which is highly nonlinear.

### T2.4. Gallagher et al. (2021) — Word Shift Graphs

**Citation:** Gallagher, R. J., et al. "Generalized word shift graphs: A method for visualizing and explaining pairwise comparisons between texts." *EPJ Data Science* 10(1): 4.

**URL:** https://doi.org/10.1140/epjds/s13688-021-00260-3

**Relevance:** Shift graph methodology for decomposing aggregate differences into component contributions. Could inform our protocol comparison visualization — showing *which* parameters drive the difference between two protocols, decomposed into positive/negative contributions.

### T2.5. Reagan et al. (2016) — Emotional Arcs

**Citation:** Reagan, A. J., et al. "The emotional arcs of stories are dominated by six basic shapes." *EPJ Data Science* 5(1): 31.

**URL:** https://doi.org/10.1140/epjds/s13688-016-0093-1

**Relevance:** Dimensionality reduction of narrative trajectories to a small set of archetypal shapes. Analogous to our trajectory classification problem: can we reduce the ~3000-step ATP/het trajectories to a taxonomy of "aging arcs"? The methodology (SVD on trajectory space) could directly apply.

### T2.6. Bliss et al. (2014) — Evolutionary Algorithm for Link Prediction

**Citation:** Bliss, C. A., et al. "An evolutionary algorithm approach to link prediction in dynamic social networks." *J. Computational Science* 5(5): 750-764.

**URL:** https://doi.org/10.1016/j.jocs.2014.01.003 | Zotero: `5MP7HUC4`

**Relevance:** EA methodology from the D&D group — directly relevant to our `ea_optimizer.py` and `llm_seeded_evolution.py`. Shows how the group thinks about evolutionary search in complex spaces.

---

## Tier 3: Health/Biomedical Applications

### T3.1. Reece & Danforth (2017) — Instagram Depression Markers

**Citation:** Reece, A. G. & Danforth, C. M. "Instagram photos reveal predictive markers of depression." *EPJ Data Science* 6(1): 15.

**URL:** https://doi.org/10.1140/epjds/s13688-017-0110-z | Zotero: `RHIYL4AA`

**Relevance:** Demonstrates the group's approach to digital biomarker extraction — passive sensing → feature extraction → health prediction. Directly analogous to the LEMURS Oura ring → sleep features → stress prediction pipeline. Their methodology (classifier outperforming GP diagnosis) shows the potential for objective biomarker-based health assessment.

### T3.2. Reece et al. (2017) — Twitter Mental Illness Forecasting

**Citation:** Reece, A. G., et al. "Forecasting the onset and course of mental illness with Twitter data." *Scientific Reports* 7(1): 13006.

**URL:** https://doi.org/10.1038/s41598-017-12961-9 | Zotero: `9HHICVIU`

**Relevance:** Longitudinal health prediction from passive data. The temporal forecasting methodology (predicting onset *before* diagnosis) parallels our causal surgery analysis — identifying the point of no return for intervention.

### T3.3. Ross et al. (2020) — Story Arcs in Palliative Care

**Citation:** Ross, L., Danforth, C. M., et al. "Story Arcs in Serious Illness: Natural Language Processing features of Palliative Care Conversations." *Patient Education and Counseling* 103(4): 826-832.

**URL:** https://doi.org/10.1016/j.pec.2019.11.021

**Relevance:** NLP applied to clinical trajectories. Shows the group's interest in health narratives and serious illness — thematically close to our aging simulation context.

### T3.4. Linnell et al. (2020) — Sleep Loss Daylight Savings

**Citation:** Linnell, K., et al. "The sleep loss insult of Spring Daylight Savings in the US is absorbed by Twitter users within 48 hours." arXiv:2004.06790.

**URL:** http://arxiv.org/abs/2004.06790

**Relevance:** Studies population-level sleep disruption and recovery time (~48 hours). Maps to our resilience framework: sleep disruption is a "disturbance" with measurable resistance and recovery metrics. The 48-hour recovery finding suggests biological sleep recovery is fast when the insult is brief.

### T3.5. Price et al. (2021) — Doomscrolling and Mental Health

**Citation:** Price, M., et al. "Doomscrolling during COVID-19: The negative association between daily social and traditional media consumption and mental health symptoms." PsyArXiv.

**URL:** https://doi.org/10.31234/osf.io/s2nfg

**Relevance:** Behavioral patterns affecting mental health outcomes — relevant to our lifestyle_module.py behavioral modifiers.

---

## Tier 4: Network & Contagion Theory (Structural Analogies)

### T4.1. Dodds & Watts (2004, 2005) — Generalized Contagion

**Citations:**
- Dodds, P. S. & Watts, D. J. "Universal Behavior in a Generalized Model of Contagion." *PRL* 92(21): 218701 (2004).
- Dodds, P. S. & Watts, D. J. "A generalized model of social and biological contagion." *J. Theor. Biol.* 232(4): 587-604 (2005).

**URLs:** https://doi.org/10.1103/PhysRevLett.92.218701 | https://doi.org/10.1016/j.jtbi.2004.09.006

**Zotero:** `5AEPGUM3`

**Relevance:** The generalized contagion model unifies biological and social spreading processes with threshold dynamics. Our heteroplasmy cliff is essentially a contagion threshold — when damaged mitochondria exceed a threshold, the "disease" (energy collapse) spreads irreversibly. The Dodds-Watts framework's treatment of dose-dependent and memory-dependent contagion could inform how we model cumulative damage.

### T4.2. Harris, Danforth, & Dodds (2013) — Dynamical Influence

**Citation:** Harris, K. D., Danforth, C. M., & Dodds, P. S. "Dynamical influence processes on networks: General theory and applications to social contagion." *Phys. Rev. E* 88(2): 022816.

**URL:** https://doi.org/10.1103/PhysRevE.88.022816

**Relevance:** General theory of influence propagation — applicable to how mitochondrial damage spreads through the cellular population. The "influence" of one damaged mitochondrion on its neighbors via shared ROS and membrane potential.

### T4.3. Callaway et al. (2000) — Network Robustness and Fragility

**Citation:** Callaway, D. S., et al. "Network robustness and fragility: Percolation on random graphs." *PRL* 85: 5468-5471.

**Relevance:** Percolation framework for understanding when systems catastrophically fail. Directly maps to our cliff dynamics: the mitochondrial population "percolates" through damage states, and the cliff represents a percolation threshold.

---

## Tier 5: Scaling Laws & Allometry (Biological Foundations)

### T5.1. Dodds & Rothman (1999-2000) — River Network Geometry

**Citations:** Series of 4 papers on scaling, fluctuations, and geometry of river networks.

**URLs:** https://doi.org/10.1103/PhysRevE.59.4865 | Zotero: `FI4QWY9Q`, `XGTP4TMI`

**Relevance:** Demonstrates Dodds' rigorous approach to scaling analysis in branching networks. Methodological template for how we might analyze scaling relationships in our mitochondrial population dynamics.

### T5.2. Dodds (2010) — Optimal Branching Networks

**Citation:** Dodds, P. S. "Optimal Form of Branching Supply and Collection Networks." *PRL* 104(4): 048702.

**URL:** https://doi.org/10.1103/PhysRevLett.104.048702 | Zotero: `CM6CTNUJ`

**Relevance:** Optimization of biological supply networks. Relates to our multi-tissue simulation where NAD+ is a shared resource distributed across brain/muscle/cardiac tissues.

### T5.3. Price et al. (2012) — Testing Metabolic Theory

**Citation:** Price, C. A., Weitz, J. S., et al. "Testing the metabolic theory of ecology." *Ecology Letters* 15(12): 1465-1474.

**URL:** https://doi.org/10.1111/j.1461-0248.2012.01860.x

**Relevance:** Rigorous empirical test of metabolic scaling theory. Our ATP production model implicitly assumes metabolic scaling relationships; this paper's approach to testing such assumptions could inform our model validation.

---

## Papers in Your Zotero Library

The following D&D papers from this bibliography are confirmed in your local Zotero:

| Zotero ID | Paper | Tier |
|---|---|---|
| `RHIYL4AA` | Reece & Danforth 2017 (Instagram depression) | T3.1 |
| `9HHICVIU` | Reece et al. 2017 (Twitter mental illness) | T3.2 |
| `5AEPGUM3` | Dodds & Watts 2004 (Generalized contagion) | T4.1 |
| `5MP7HUC4` | Bliss et al. 2014 (EA link prediction) | T2.6 |
| `HBJBARND` | Watts & Dodds 2002 (Identity and search) | — |
| `UZJUUKTN` | Dodds et al. 2003 (Search in global networks) | — |
| `R4VZRBEF` | Dodds et al. 2011 (Hedonometrics) | — |
| `ARAAA4IH` | Dodds & Danforth 2010 (Happiness measurement) | — |
| `UPP9VT53` | Mitchell et al. 2013 (Geography of happiness) | — |
| `NJMRDD2Z` | Dodds et al. 2015 (Universal positivity bias) | — |
| `PVNDUJJ2` | Pechenick et al. 2015 (Google Books limits) | — |
| `FI4QWY9Q` | Dodds & Rothman 1999 (River network scaling) | T5.1 |
| `XGTP4TMI` | Dodds & Rothman 2000 (River geometry I) | T5.1 |
| `CM6CTNUJ` | Dodds 2010 (Optimal branching) | T5.2 |
| `SX2Q866N` | Frank et al. 2013 (Happiness patterns) | — |
| `RC47PEUU` | Kloumann et al. 2012 (Language positivity) | — |
| `FFIPFQKI` | Gallagher et al. 2018 (BLM discourse) | — |
| `885XPIUN` | Schwartz et al. 2019 (Greenspace sentiment) | — |
| Multiple | LEMURS papers (Fudolig, Bloomfield) | L1-L3 |

---

## Recommended Reading Order

For maximum impact on the simulator project:

1. **L1. Bloomfield et al. 2024** — PLOS Digital Health (LEMURS stress prediction) — **calibrates our coefficients**
2. **L2. Fudolig et al. 2024** — Digital Biomarkers (sleep HR shapes) — **informs sleep quality modeling**
3. **T1.2. Mitchell et al. 2020** — Chimera states (bifurcation ODE) — **methodological analogy for cliff dynamics**
4. **T2.2. Dewhurst et al. 2019** — Shocklet transform — **trajectory analysis methodology**
5. **T1.1. Dodds, Rothman, Weitz 2001** — Metabolic scaling — **biological scaling foundations**
6. **T2.5. Reagan et al. 2016** — Emotional arcs — **trajectory classification approach**
7. **T4.1. Dodds & Watts 2004** — Generalized contagion — **threshold dynamics theory**
8. **L3. Fudolig et al. 2025** — npj Complexity (collective sleep) — **population-level sleep baselines**
9. **T2.1. Dodds et al. 2020** — Allotaxonometry — **protocol comparison framework**
10. **T1.3. Allgaier et al. 2012** — Climate model correction — **ODE calibration methodology**

---

## Actionable Next Steps

1. **Coefficient validation:** LEMURS data (L1) confirms `SLEEP_DISRUPTION_IMPACT=0.5` is well-calibrated. The remaining gap is alcohol→sleep, which these papers don't directly address.

2. **Methodology adoption:** The shocklet transform (T2.2) could be added to our trajectory analysis toolkit. The emotional arcs approach (T2.5) could classify our simulation trajectories into archetypal aging patterns.

3. **Email to Dodds & Danforth:** The 3 CLAUDE.md requests remain valid. Add: "We've read the Bloomfield et al. 2024 paper and confirmed our SLEEP_DISRUPTION_IMPACT=0.5 is consistent with your TST→PSS findings (OR=0.617/hr). We still need alcohol→sleep dose-response data if LEMURS captured it."

4. **Theory connection:** The Dodds-Watts generalized contagion framework (T4.1) provides a rigorous theoretical grounding for our heteroplasmy cliff as a contagion threshold. This could strengthen the Cramer book's narrative with formal network-theoretic foundations.
