# LEMURS Paper #7: Behavioral Interventions RCT (14-Week, 4-Arm)

**Citation:** Bloomfield, L. S. P. et al. (2025). Effects of behavioral interventions on psychological and biometric measures of well-being in young adults: results of a 14-week, four-arm, randomized, controlled trial. *In Review* (preprint).

---

## Parameters

### Design
- Phase II RCT, 4 arms: Control, Physical Activity, Nature Experiences, Group Therapy
- N = 379 (ITT), 311 (per-protocol), 216 (per-completion)
- 14 weeks, Spring 2023; Week 8 excluded (spring break)
- Prior MH diagnosis: 67.5%

### Self-Report Outcomes (weekly)
| Variable | Instrument | Baseline (Control) |
|---|---|---|
| Depression | DASS-21 | 9.05 (8.60) |
| Anxiety | DASS-21 | 8.05 (7.19) |
| Stress | DASS-21 | 11.75 (8.01) |
| PSS | PSS-10 | 16.86 (7.09) |
| Well-being | WEMWBS | 46.66 (8.82) |

### Biometric Outcomes (Oura Ring, nightly)
| Variable | Baseline (Control) |
|---|---|
| TST | 7.408 hrs (0.807) |
| RHR | 63.224 bpm (7.972) |
| HRV (RMSSD) | 62.403 ms (28.540) |
| ARR | 15.602 breaths/min (1.642) |

---

## Mathematical Relations

### Mixed-Effects Model
```
Outcome_ij = (beta_00 + beta_01*TimeInvariant_i + u_0i) + (beta_10*week + u_1i) + beta_20*TimeVarying_ij + epsilon_ij
```

### Significant Group Effects

| Effect | beta | p |
|---|---|---|
| **Nature -> DASS Stress (main)** | **-1.94** | **0.047** |
| **Nature x Week -> PSS** | **-0.089** | **0.019** |
| **Nature x Week -> WEMWBS** | **+0.104** | **0.045** |
| **Therapy x Week -> DASS Stress** | **-0.102** | **0.029** |
| **Therapy x Week -> ARR** | **-0.012** | **0.009** |
| **Nature -> HRV (main)** | **+9.13** | **0.037** |
| MH history -> Stress | -3.10 | <0.001 |

### Mediation Analysis: Nature -> PSS -> HRV
| Path | beta | p |
|---|---|---|
| Nature -> PSS (a) | -1.507 | 0.047 |
| PSS -> HRV (b) | -0.618 | 0.024 |
| Nature -> HRV (total, c) | +9.13 | 0.037 |
| Nature -> HRV (direct, c') | +9.13 | 0.037 |
| **Result: Partial mediation** | | |

### Mediation Analysis: Nature -> WEMWBS -> HRV
| Path | beta | p |
|---|---|---|
| Nature -> WEMWBS x Week (a) | +0.107 | 0.039 |
| WEMWBS -> HRV (b) | +0.365 | 0.083 (marginal) |

### Time Trends (all groups)
| Outcome | Direction | p |
|---|---|---|
| Depression | Decreasing | 0.026 |
| Anxiety | Decreasing | <0.001 |
| Stress | Decreasing | 0.011 |
| PSS | **Increasing** | 0.002 |
| TST | Decreasing | 0.037 |
| ARR | Increasing | 0.004 |

### Intervention Costs
| Intervention | Annual Cost | Significant Effects |
|---|---|---|
| Nature | $2,940 | Stress, PSS, WEMWBS, HRV |
| Physical Activity | $6,300 | Marginal (anxiety, stress, TST) |
| Group Therapy | $195,000 | Stress slope, ARR |

---

## Complex Systems Structure

### Cross-Scale Mediation
Behavioral intervention (nature) -> Psychological state (stress reduction) -> Physiological regulation (HRV). The psychological scale partially mediates the behavioral-to-physiological connection.

### Causal Pathway
```
Nature Exposure -> PSS (down, beta=-1.507) -> HRV (up, beta=-0.618 per PSS point)
                                                    |
                -> WEMWBS (up, beta=+0.107/week) -> HRV (up, beta=+0.365, marginal)
```

### Cumulative Non-Instantaneous Dynamics
Nature intervention shows NO main effect on PSS/WEMWBS, but the interaction with TIME is significant. Effects accrue over weeks, not as a step function -- consistent with slow adaptation/plasticity rather than immediate perturbation.

### Paradoxical HRV Response
Active intervention groups (nature, therapy) showed attenuated HRV increase relative to control. Authors hypothesize "more emotionally demanding or introspective experiences could initially increase physiological arousal or slow autonomic recovery." This is a competing-processes dynamic.

### Differential Attrition as Selection
Physical Activity lost 71% of participants (90->26). Per-protocol analyses show "more pronounced effects" -- dose-response nonlinearity where engagement intensity matters.

---

## Relevance to Our Work
**HIGH.** This is the interventional complement to Paper #3's observational findings. Key coefficients for our model:

1. **Nature -> Stress mediation -> HRV:** The complete causal pathway is quantified (beta_a = -1.507, beta_b = -0.618)
2. **Nature -> HRV = +9.13 ms RMSSD.** This is the magnitude of physiological benefit from a nature-based intervention at $2,940/year
3. **Therapy -> ARR = -0.012 breaths/min/week** over 14 weeks (respiratory regulation)
4. **PSS increases over time** (beta = +0.077/week) even as depression/anxiety decrease -- stress and distress are dissociable constructs
5. The **66:1 cost ratio** (therapy:nature) with nature producing more significant effects is a powerful argument for nature-based interventions
