# LEMURS Paper #2: Two Fundamental Shapes of Sleep Heart Rate Dynamics

**Citation:** Fudolig, M. I. et al. (2024). The two fundamental shapes of sleep heart rate dynamics and their connection to mental health in college students. *Digital Biomarkers*, 8(1), 120-131.

---

## Parameters

### Biometric (Oura Ring Gen3, 5-min intervals)
| Variable | Cluster 1 (64%) | Cluster 2 (36%) |
|---|---|---|
| Lowest HR time offset (% of sleep) | mean=65, median=66 | mean=35, median=34 |
| Deep sleep % | 31% | 34% |
| REM sleep % | 20% | 22% |
| Light sleep % | 49% | 45% |
| Total sleep duration (hrs) | 7.49 | 7.46 (p=0.18, n.s.) |
| Sleep efficiency % | 87.81 | 88.54 |
| Sleep latency (min) | 12.91 | 8.97 |
| Average HR (bpm) | 63.94 | 62.22 |
| Average HRV (ms, RMSSD) | 63.32 | 67.18 |
| Average respiratory rate | 15.64 | 15.50 |

### Survey / Self-Report
- PSS (0-40), GAD-7 (0-21): collected weekly, NOT significant cluster predictors
- Prior MH diagnosis: 45% of sample; perceived impairment: 35%
- Trauma categories (LEC): 0 (28%), 1 (32%), >=2 (40%)

### Study Parameters
- N = 599 participants, 20,167 sleep periods (after exclusions)
- Mental health subsample: N = 505, 15,073 sleep periods
- 8 weeks (Oct-Dec 2022), Oura Gen3

---

## Mathematical Relations

### Time Series Processing
**Piecewise Aggregate Approximation (PAA):**
```
PAA_j = mean(HR[floor((j-1)*L/n)+1 : floor(j*L/n)])  for j = 1,...,30
```
**Z-score standardization** per sleep period, then **k-means clustering** (k=2, Euclidean distance). 99.96% cluster stability across 900 runs.

### Correlations (Spearman's rho)
| Pair | rho |
|---|---|
| HR nadir timing vs. average HR | 0.1 |
| HR nadir timing vs. lowest HR | 0.03-0.05 |
| Prior MH diagnosis vs. impairment | 0.8 |
| PSS vs. GAD-7 | 0.7 |

### Mixed-Effects Logistic Regression (sleep period cluster ~ individual characteristics)
| Predictor | p-value | Direction |
|---|---|---|
| Perceived impairment | 0.001 | Impairment -> Cluster 1 (later HR nadir) |
| Trauma (>=2 categories) | 0.013 | -> Cluster 1 |
| Male gender | 0.028 | -> Cluster 1 |

### Individual-Level Prediction (frac_1 = fraction of nights in Cluster 1)
| Outcome | p-value |
|---|---|
| frac_1 -> perceived impairment | < 0.001 |
| frac_1 -> prior MH diagnosis | 0.003 |
| frac_1 -> trauma (>=2) | 0.001 |
| frac_1 -> female gender | 0.045 |

### Gender Stratification
- **Females:** Both impairment (p<0.001) and trauma (p=0.008) predict Cluster 1
- **Males:** Neither significant (p=0.622, p=0.309)
- **Non-binary:** Trauma significant (p=0.013), impairment not

---

## Complex Systems Structure

### Central Finding: Shape as Emergent Biomarker
The HR curve *shape* (when the nadir is reached) carries mental health information NOT captured by scalar statistics. Spearman's rho between the shape feature and both mean HR and lowest HR is only 0.03-0.1. **The dynamical trajectory contains information destroyed by summary statistics.**

### Two Robust Attractors
Despite 20,167 continuous-variation sleep periods, k-means consistently finds exactly 2 stable centroids (99.96% stability). These are emergent population-level categories.

### Temporal Resolution Mismatch
Nightly HR curves predict mental health, but weekly PSS/GAD-7 do NOT predict HR curve patterns. The physiological signal operates at a faster timescale than subjective reporting.

### Feedback Loops (inferred)
1. **Mental health -> Altered HR dynamics -> Poor sleep quality -> Worsened mental health** (bidirectional, Mendelian randomization evidence cited)
2. **Anxiety/trauma -> Amygdala hyperactivity -> Altered neurotransmitters -> Delayed HR nadir -> Reduced HRV -> Stress/anxiety** (autonomic mediation)
3. **MH disorders -> Altered habits -> Changed sleep architecture -> Altered HR shape** (behavioral)

### Gender-Modulated Nonlinearity
The HR shape-mental health coupling is NOT additive with gender -- gender acts as a moderator that switches the relationship on/off entirely (significant for females, not for males). This is interaction nonlinearity.

---

## Relevance to Our Work
**HIGH.** This paper demonstrates that sleep HR *dynamics* (not just levels) encode health state information. The two-attractor structure (Cluster 1 = later nadir, less deep/REM, higher HR, lower HRV = mental health burden) directly parallels our heteroplasmy cliff concept. The finding that scalar summaries destroy information supports our approach of modeling full ODE trajectories rather than endpoint values. The gender-dependent coupling is a model for how individual differences modulate cliff proximity.
