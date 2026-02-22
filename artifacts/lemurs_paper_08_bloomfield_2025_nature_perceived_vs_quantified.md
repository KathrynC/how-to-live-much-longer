# LEMURS Paper #8: Not All Nature Exposure Is Equal

**Citation:** Bloomfield, L. S. P. et al. (2025). Not all nature exposure is equal: Perceived nature engagement predicts mental health better than quantified nature. *In Review* (preprint).

---

## Parameters

### Study Design
- N = 548, Fall 2023 semester, 15 weeks, bi-weekly surveys
- Two independent measures of nature exposure:
  - **Perceived (HIN):** Self-reported hours in nature per week (M = 6.06 hrs/week)
  - **Quantified:** GPS-tracked NatureDose minutes per week (M = 159.53 min/week)

### Mental Health Outcomes
| Variable | Instrument | Mean (SD) |
|---|---|---|
| Depression | DASS-21 | 8.42 (8.05) |
| Anxiety | DASS-21 | 7.53 (7.33) |
| Stress | DASS-21 | 11.27 (7.73) |
| PSS | PSS-10 | 16.84 (6.79) |
| Loneliness | UCLA (1 item) | 4.73 (1.69) |
| Positive Affect | PANAS+ | 30.28 (7.23) |
| Negative Affect | PANAS- | 20.82 (7.06) |
| Sleep Quality | -- | 5.79 (1.34) |

### NatureScore
- Integrates 30+ remotely sensed variables (vegetation, water, parks, air/noise/light pollution, road density, land classification)
- Range: 1 (nature deficit) to 100 (nature rich)
- Burlington census tracts: 22 to 99.9

---

## Mathematical Relations

### Mixed-Effects Models (LMM, random intercepts for participant)

#### Perceived Nature (HIN) -> Mental Health
| Outcome | beta | p | Direction |
|---|---|---|---|
| Depression | **-0.066** | **<0.001** | Protective |
| Loneliness | **-0.0046** | **<0.001** | Protective |
| Positive Affect | **+0.118** | **<0.001** | Beneficial |
| Stress | significant | sig | Protective |
| PSS | significant | sig | Protective |
| Negative Affect | significant | sig | Protective |
| Anxiety | n.s. | -- | -- |
| Sleep Quality | n.s. | -- | -- |

#### Quantified Nature (GPS) -> Mental Health (PARADOXICAL)
| Outcome | beta | p | Direction |
|---|---|---|---|
| Depression | **+0.032** | **0.003** | **Harmful (paradoxical)** |
| Stress (log) | +0.00014 | <0.001 | Harmful |
| Anxiety (log) | +0.00016 | <0.001 | Harmful |

#### On-Campus Nature -> Mental Health
- Stress: beta = +0.00018, p < 0.001 (on-campus nature INCREASES stress)

#### Combined Model (both predictors simultaneously)
```
Depression = beta_0 + (-0.069)*HIN + (+0.001)*QuantifiedNature + u_i + epsilon
```
Both remain significant; NO interaction. Perceived nature is robust.

---

## Complex Systems Structure

### Central Finding: Measurement Dissociation
GPS exposure and perceived exposure predict mental health in **opposite directions**:
- Perceived (HIN): consistently protective
- Quantified (GPS): paradoxically harmful

This is the paper's key contribution: **dose ≠ bioavailability**. Physical presence in nature without psychological engagement has no benefit or is actually associated with worse outcomes (reverse causality: distressed people seek nature as coping).

### Context as Mode Switch
On-campus vs. off-campus reverses the sign of quantified exposure effects:
- On-campus: nature exposure -> MORE stress (task-oriented transit)
- Off-campus: modest negative association with stress (discretionary, immersive)

### Attention Restoration Theory as Implicit ODE
```
dDAC/dt = -alpha*(academic_demands + digital_demands) + beta*nature_engagement_quality
```
Where DAC = Directed Attention Capacity, and nature_engagement_quality depends on perception/attention, NOT mere GPS proximity.

### Feedback Loops (inferred)
1. **Engagement -> Positive affect -> Motivation to seek nature -> More engagement** (positive)
2. **Distress -> Seek nature -> GPS records exposure -> But without engagement -> Distress not reduced** (paradoxical positive correlation)
3. **On-campus transit -> GPS records "nature" -> But task-oriented -> Stress maintained** (context-dependent)

### Competing Attractors
Digital attention vs. nature attention compete for finite attentional budget. A student physically in a park but on their phone gets GPS-recorded exposure but NO attentional restoration.

---

## Relevance to Our Work
**Moderate.** The key insight is methodological: any sleep/nature intervention in our model must distinguish between *exposure* and *engagement*. Physical presence is insufficient; cognitive engagement is the active ingredient. For our ODE, this means the "nature intervention" parameter should model attention/engagement quality, not just duration. The Attention Restoration Theory equation (dDAC/dt) is a candidate for incorporation.

The paradoxical finding also cautions against naive "more nature = better" assumptions in intervention protocols.
