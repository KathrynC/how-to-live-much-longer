# LEMURS Paper #4: Anxiety Trajectories in First-Year College Students

**Citation:** Bloomfield, L. S. P. et al. (2024). Events and behaviors associated with symptoms of generalized anxiety disorder in first-year college students. *JAACAP Open*.

---

## Parameters

### Primary Outcome: GAD-7 (Generalized Anxiety Disorder Questionnaire)
- 7 items, 4-point Likert (0-3), range 0-21
- Threshold: >= 10 for moderate anxiety (sensitivity 89%, specificity 82%)
- Cronbach alpha = 0.90
- Mean: 6.49 (SD 5.33)
- 51% of participants scored >= 10 at least once; 9% at ALL timepoints

### Personality Traits (TIPI, Ten Item Personality Inventory)
| Trait | Range | Role |
|---|---|---|
| Extraversion | 1-7 | Protective |
| Emotional Stability | 1-7 | **Most powerful predictor across all models** |
| Agreeableness | 1-7 | Risk factor for occurrence |

### Other Predictors
- Prior anxiety diagnosis: 39% (n=210)
- Trauma (LEC): 0 exposures (37%), 1 (28%), >=2 (35%)
- Academic stressor (test/project): binary, weekly
- Gender: Female 66%, Male 28%, NB/Trans 6%

### Study Parameters
- N = 539 (89% retention), 3,318 weekly survey responses
- 7 weekly surveys, Oct 21 - Dec 12, 2022

---

## Mathematical Relations

### Model 1: Occurrence (GEE, binomial, log link)
| Predictor | AOR (95% CI) | p |
|---|---|---|
| **Emotional Stability** | **0.58 (0.50, 0.67)** | **<0.001** |
| **Extraversion** | **0.83 (0.75, 0.92)** | **0.001** |
| **Agreeableness** | **1.20 (1.03, 1.39)** | **0.020** |
| **Anxiety Diagnosis** | **2.10 (1.51, 2.93)** | **<0.001** |
| **LEC >= 2** | **1.80 (1.28, 2.52)** | **0.001** |
| **Academic Stressor** | **1.68 (1.42, 1.98)** | **<0.001** |

### Model 2: Persistence (zero-one inflated beta regression)
- Emotional Stability: IRR = 0.72 (p=0.022)
- Anxiety Diagnosis: IRR = 1.64 (p=0.231 in combined model)
- Academic Stressor: IRR = 2.09 (p=0.002)

### Model 3a: Development GAD+ (week-to-week transition below -> above threshold)
- 20% (562/2772) of transitions were GAD+
- Emotional Stability: OR = 0.84 (p<0.001)
- Academic Stressor: OR = 1.58 (p<0.001)

### Model 3b: Reduction GAD- (week-to-week transition above -> below threshold)
- 20% (560/2783) of transitions were GAD-
- Emotional Stability: OR = 0.82 (p<0.001)
- Academic Stressor: OR = 0.69 (p<0.001) -- stressor PREVENTS recovery
- Week of Study: OR = 0.92 (p=0.001) -- recovery rate declines over time

### Empirical Transition Matrix (Markov-like)
| Transition | Range across weeks |
|---|---|
| P(below -> below) | 0.83 - 0.95 |
| P(below -> above) | 0.05 - 0.17 (mean ~0.12) |
| P(above -> below) | 0.44 - 0.82 (mean ~0.65) |
| P(above -> above) | 0.18 - 0.56 |

**Asymmetric dynamics:** Recovery from GAD state is much more probable than development into it in any given week, but recovery rate DECLINES over the semester (hysteresis).

---

## Complex Systems Structure

### Bimodal Attractor Structure
The zero-one inflated beta distribution reveals three subpopulations:
1. **Resilient** (never cross threshold): ~49%
2. **Chronic** (always above threshold): ~9%
3. **Fluctuating** (transition between states): ~42%

### Multiplicative Risk Compounding
On the log-odds scale, risk factors combine multiplicatively. A student with ALL risk factors (female, prior diagnosis, LEC>=2, test/project, low emotional stability):
```
Combined OR ~ 2.10 * 1.80 * 1.68 * (0.58^-3) * (0.83^-3) ~ 57
```

### Hysteresis
Week-of-study predictor shows declining GAD+ (OR=0.96) BUT also declining GAD- (OR=0.92). Recovery declines faster than development -> the system settles into its current state over time.

### Emotional Stability as Universal Predictor
Significant in ALL 5 model specifications with the largest effect sizes. It is the only variable predicting occurrence, persistence (both measures), development, AND reduction.

---

## Relevance to Our Work
**Low-moderate.** No biometric/wearable data used (pure self-report). However, three elements are useful:
1. The Markov transition matrix could calibrate a discrete-state stress model
2. The multiplicative risk compounding parallels our multi-factor cliff-approach model
3. The hysteresis finding (system settles into current state over time) is relevant to our modeling of intervention timing -- earlier intervention matters because the system becomes increasingly resistant to state change
