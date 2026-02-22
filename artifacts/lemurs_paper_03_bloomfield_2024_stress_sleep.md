# LEMURS Paper #3: Predicting Stress from Wearable Sleep Data

**Citation:** Bloomfield, L. S. P. et al. (2024). Predicting stress in first-year college students using sleep data from wearable devices. *PLOS Digital Health*, 3(4), e0000473.

---

## Parameters

### Outcome: Perceived Stress Scale (PSS-10)
- Mean: 15.85 (SD 7.33), range 0-40
- Binary threshold: PSS >= 14 (moderate-to-high stress)
- 64.14% of weekly observations >= 14

### Sleep Variables (Oura Ring, nightly, aggregated weekly)
| Variable | Mean (SD) |
|---|---|
| Total Sleep Time (TST) | 7.41 hrs (0.83) |
| Resting Heart Rate (RHR) | 62.75 bpm (8.64) |
| Heart Rate Variability (HRV) | 68.24 ms RMSSD (33.09) |
| Average Respiratory Rate (ARR) | 15.56 breaths/min (1.61) |
| Deep sleep (N3) | 32.18% (9.63) |
| REM sleep | 20.39% (7.07) |
| Light sleep (N1+N2) | 47.43% (8.71) |
| Sleep efficiency | 87.13% (4.46) |
| Sleep latency | 11.79 min (6.61) |
| Skin temperature deviation | 0.14 C (0.20) |

### Demographics
- N = 525, 3,112 user-week observations
- Female: 66%, Male: 27%, Trans/NB: 6%
- White: 88%

---

## Mathematical Relations

### Core Model: Mixed-Effects Linear Regression
```
PSS_ij = beta_0 + beta_1*(sleep_measure) + beta_2*(week) + beta_3*(gender) + u_j + epsilon_ij
```

### KEY COEFFICIENTS: Sleep -> Stress (Continuous PSS)

| Predictor | beta (SE) | p | Interpretation |
|---|---|---|---|
| **TST (hrs)** | **-0.877 (0.138)** | **<0.001** | Each hour of sleep = 0.877 less stress |
| **RHR (bpm)** | **+0.055 (0.021)** | **0.009** | Each bpm = 0.055 more stress |
| **HRV (ms)** | **-0.012 (0.006)** | **0.035** | Each ms HRV = 0.012 less stress |
| **ARR (breaths/min)** | **+0.270 (0.131)** | **0.040** | Each breath/min = 0.270 more stress |
| Gender (nonmale) | +2.956 (0.571) | <0.001 | Nonmale = 2.956 higher stress |
| Week of semester | -0.346 (0.047) | <0.001 | Stress decreases over semester |

### KEY COEFFICIENTS: Sleep -> Stress (Binary PSS >= 14, Odds Ratios)

| Predictor | OR | p | Interpretation |
|---|---|---|---|
| **TST (hrs)** | **0.617** | **<0.001** | **38.3% reduction in stress odds per hour** |
| **RHR (bpm)** | **1.036** | **0.010** | **3.6% increase per bpm** |
| **HRV (ms)** | **0.988** | **0.035** | **1.2% reduction per ms** |
| **ARR (breaths/min)** | **1.230** | **0.010** | **23.0% increase per breath/min** |
| Gender (nonmale) | 4.953 | <0.001 | ~5x odds for nonmale |

### Within-Person Deviation Effects (stronger than absolute levels)
| Predictor | beta for Delta PSS | p |
|---|---|---|
| TST deviation | -1.312 | <0.001 |
| RHR deviation | +0.188 | <0.001 |
| HRV deviation | -0.048 | <0.001 |

**Critical finding:** Deviations from personal mean are 2.2x stronger predictors than absolute values. The system is sensitive to **perturbations from homeostasis**.

### Variance Decomposition (ICC)
| Variable | Between-subject % | Within-subject % | Type |
|---|---|---|---|
| TST | Moderate | **Substantial** | State variable |
| RHR | **>80%** | <20% | Trait variable |
| HRV | **>80%** | <20% | Trait variable |
| ARR | **~93%** | ~7% | Trait variable |

---

## Complex Systems Structure

### State vs. Trait Architecture
TST acts as a **fast state variable** (responsive to weekly perturbations), while RHR/HRV/ARR act as **slow trait variables** (setting baseline susceptibility). This two-tier structure is characteristic of complex systems where fast variables fluctuate on slow variables.

### Feedback Loops
1. **Stress-Sleep bidirectional:** Decreased TST -> Increased PSS -> Disrupted sleep -> Further stress (explicitly cited, only sleep->stress direction measured)
2. **Autonomic mediation:** Stress -> Decreased parasympathetic -> Increased RHR, Decreased HRV, Increased ARR -> Disrupted sleep architecture -> Further autonomic dysregulation

### Nonlinear Features
- PSS >= 14 threshold creates a sigmoidal probability response (logistic model)
- Deviation effects stronger than level effects = sensitivity to perturbation from homeostasis
- Thanksgiving break perturbation: system-level shift across multiple coupled variables simultaneously

---

## Relevance to Our Work
**HIGHEST.** This is the paper with the actual coupling coefficients we need for the sleep-inflammation pathway in our ODE. The key numbers:

- **Sleep -> Stress:** -0.877 PSS points per hour of sleep (continuous), OR=0.617 (38.3% risk reduction per hour)
- **HRV -> Stress:** -0.012 PSS points per ms HRV
- **RHR -> Stress:** +0.055 PSS points per bpm
- **ARR -> Stress:** +0.270 PSS points per breath/min

The state-vs-trait decomposition (TST = fast state, cardiorespiratory = slow trait) maps directly onto our ODE architecture. The finding that deviations from homeostasis are stronger predictors than absolute levels supports our cliff-dynamics framework.
