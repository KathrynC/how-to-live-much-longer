# LEMURS Paper #1: Trial Design (Protocol Paper)

**Citation:** Price, M. et al. (2023). A large clinical trial to improve well-being during the transition to college using wearables: The Lived Experiences Measured Using Rings Study. *Contemporary Clinical Trials*, 133, 107338.

**Paper type:** Protocol paper (pre-data). No empirical results.

---

## Parameters

### Experimental Design
- N = 600, 4 arms (1:1:1:1), 150/arm, blocks of 8
- Arms: Group Therapy, Physical Activity, Nature Experiences, Weekly Assessment (control)
- Duration: 14 weeks (spring semester), follow-up at 6 and 12 months
- Power: 0.80 for d = 0.1 standard units; expected attrition 20%

### Primary Outcomes (weekly self-report)
| Instrument | Construct | Items |
|---|---|---|
| WEMWBS | Subjective well-being | 14 items, Likert |
| DASS-21 | Depression, Anxiety, Stress | 7 items each |
| PSS-10 | Perceived stress | 10 items, 0-40 |

### Secondary Outcomes
- Sleep Quality Index (1 item, 11-point scale)
- MOSSS social support (19 items, 4 dimensions)
- Time Outside Survey (custom)
- BSTAD substance use

### Passive Biometric Variables (Oura Ring Gen3)
| Variable | Sensor | Resolution |
|---|---|---|
| Heart rate | PPG | Continuous |
| Respiration | PPG-derived | Continuous |
| Body temperature | NTC sensor | Continuous |
| Movement | Accelerometer | Continuous |
| Sleep score | Multi-sensor derived | 0-100, nightly |
| Physical activity score | Derived | 0-100, daily |

### Environmental Variables
- NatureScore (NatureDose app): 0-100 per minute, aggregated weekly
- Nature exposure duration: minutes/week

### Intervention Costs
| Intervention | Annual Cost |
|---|---|
| Group Therapy (Mood Lifters) | $195,000 |
| Physical Activity | $6,300 |
| Nature Experiences | $2,940 |
| Cost ratio (therapy : nature) | 66.3 : 1 |

---

## Mathematical Relations

### Planned MLM (two-level hierarchical linear model)

**Level 1 (within-person):**
```
Y_it = beta_0i + beta_1i * Time_t + e_it
```

**Level 2 (between-person):**
```
beta_0i = gamma_00 + gamma_01 * Treatment_i + gamma_02 * Covariate_i + u_0i
beta_1i = gamma_10 + gamma_11 * Treatment_i + gamma_12 * Covariate_i + u_1i
```

Primary parameter of interest: gamma_11 (treatment effect on slope).

### Planned Mechanism Analysis
For each intervention, a separate MLM tests whether the intervention-specific mechanism mediates the outcome:
- Nature arm: NatureScore -> Outcomes
- Exercise arm: Oura activity score -> Outcomes
- Therapy arm: Mood Lifters points -> Outcomes

### Planned Algorithmic Prediction
```
f: {HR(t), Temp(t), Respiration(t), Movement(t)} -> {Depression, Anxiety, Stress, Well-being}
```
Data-driven dimensionality reduction from high-frequency physiological time series to weekly psychological state.

---

## Complex Systems Structure

### Multi-Scale Architecture
| Scale | Variables | Resolution |
|---|---|---|
| Physiological | HR, respiration, temperature, movement | Seconds (continuous) |
| Sleep | Sleep quality/score, recovery | Nightly |
| Behavioral | Activity score, rest score | Daily |
| Environmental | NatureScore, time outside | Per exposure event |
| Psychological | DASS-21, PSS, WEMWBS | Weekly |
| Social | MOSSS social support | Weeks 1, 7, 14 |
| Developmental | Treatment effects, slopes | 14-week trajectory |

### Feedback Loops (inferred from design)
1. **Behavioral activation -> Well-being -> Engagement** (positive feedback via Mood Lifters points system)
2. **Sleep -> Stress -> Sleep** (negative/vicious cycle; both measured at high resolution)
3. **Nature exposure -> Positive affect -> Seeking nature** (dose-response: 120 min/week threshold cited)
4. **Physical activity -> Sleep -> Recovery -> Physical activity** (positive feedback via Oura ring capture)

### Key Design Insight
The multi-component intervention design explicitly acknowledges emergent effects: "nature experiences and group therapy arms are multi-component interventions... It may be unclear if a specific activity was effective relative to the whole treatment package."

---

## Relevance to Our Work
**Moderate.** This is the protocol paper -- no data, no coefficients. But it defines the measurement architecture and statistical framework that all subsequent LEMURS papers build on. The planned algorithmic prediction mapping (continuous physiology -> weekly psychology) is directly relevant to our sleep-inflammation coupling needs. The 66:1 cost ratio (therapy vs nature) is striking for intervention design.
