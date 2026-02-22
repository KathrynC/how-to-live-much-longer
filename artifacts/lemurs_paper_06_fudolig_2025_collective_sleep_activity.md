# LEMURS Paper #6: Collective Sleep and Activity Patterns from Wearables

**Citation:** Fudolig, M. I. et al. (2025). Collective sleep and activity patterns of college students from wearable devices. *npj Complexity*, 2:32.

---

## Parameters

### Biometric (Oura Gen3, 5-min intervals)
| Variable | Definition | Units |
|---|---|---|
| Sleep period start/end | Longest nightly sleep period | Clock time |
| Midpoint of sleep (MSF/MSW) | (start + end) / 2 | Hours from 12 AM |
| Active calories | Daily expenditure | kcal (log-transformed) |
| Activity level | MET-based intensity | 1=rest, 2=inactive, 3=low, 4=medium, 5=high |
| Ring wear compliance | % of days with >=12h non-rest wear | 77% median |

### Derived Variables
- **Social jetlag:** SJL = MSF - MSW (free day midpoint minus school day midpoint)
- **Chronotype:** MSF during Thanksgiving break (median ~5 AM = late-night)

### Demographics
- N = 582 (sleep + activity data), 566 after filtering
- Female 65%, Male 28%, Non-binary 6%
- Mental health impairment (anxiety/depression): 35%
- 97% aged 18-19

---

## Mathematical Relations

### Chronotype Prediction (Linear Regression)
| Predictor of MSF | Coefficient | p |
|---|---|---|
| Male gender | +0.472 hrs (+28.3 min later) | <0.001 |
| Mental health impairment | +0.273 hrs (+16.4 min later) | 0.014 |
| Sleep period start | +0.776 | <0.001 |
| Sleep period end | +0.841 | <0.001 |
| Sleep duration | -0.09 | 0.084 (n.s.) |

**Key finding:** Chronotype is about TIMING (start/end), NOT duration.

### Social Jetlag (Mixed-Effects, random: participant + week)
| Predictor | Coefficient | p |
|---|---|---|
| Weekday (vs weekend) | +0.924 hrs (+55.4 min earlier) | <0.001 |
| Saturday (vs Monday) | -1.000 hrs (60 min later) | <0.001 |
| Sunday (vs Monday) | -0.983 hrs | <0.001 |
| Gender, impairment | NOT significant | >0.27 |

### Sleep Debt (School weeks vs. Thanksgiving break)
| Day | Duration Difference (min) |
|---|---|
| Monday | -37.92 min |
| Tuesday | -37.50 |
| Wednesday | -37.02 |
| Thursday | -42.55 |
| Friday | -54.68 |

Students lose **37-55 minutes of sleep per night** during school vs. break.

### Active Calories (Mixed-Effects, log scale)
| Predictor | Coefficient | p | Interpretation |
|---|---|---|---|
| Male | +0.105 | <0.001 | +11.1% active calories |
| Impairment | -0.028 | 0.043 | -2.8% active calories |
| Social jetlag (hrs) | +0.023 | <0.001 | +2.3% per hour |
| Bedtime duration (hrs) | -0.021 | <0.001 | -2.1% per hour of sleep |
| Weekday | +0.051 | <0.001 | +5.1% vs weekend |
| School in session | +0.137 | 0.023 | +13.7% vs break |

---

## Complex Systems Structure

### Emergent Collective Rhythms
Population-level activity time series shows sharp synchronized spikes at class transition times -- emergent from superposition of heterogeneous individual schedules under shared institutional constraints. "Not guaranteed given the heterogeneity in individual timetables."

### Phase Transition: School vs. Break
Qualitative shift when institutional constraints are removed during Thanksgiving:
- Activity spikes disappear entirely
- Sleep timing shifts 40-54 minutes
- Active calories drop 13.7%
- Transition is sharp (single day boundary), not gradual

### Nonlinear Social Jetlag
Social jetlag is super-linear for late chronotypes -- students with later MSF experience disproportionately more misalignment. "Becomes more positive with increasing MSF."

### Asymmetric Day-of-Week Dynamics
Rapid delay on weekends (bedtime shifts ~40 min later on Friday), potentially slower re-entrainment on Monday. Asymmetric response dynamics.

### Feedback Loops (inferred)
1. **School schedule -> Sleep debt -> Later weekend sleep -> Larger social jetlag -> More Monday misalignment** (reinforcing)
2. **Mental health impairment -> Later chronotype -> More social jetlag -> Activity reduction** (inferred cascade)
3. **Campus walking between classes -> Activity spikes -> Walking absent on breaks -> Activity drops** (institutional coupling)

---

## Relevance to Our Work
**HIGH.** Key quantitative findings for our sleep-aging model:

1. **Chronic sleep debt of 37-55 min/night** during school weeks -- this is the magnitude of real-world sleep loss we should model
2. **Social jetlag = MSF - MSW** provides a quantitative circadian disruption metric
3. **Mental health impairment shifts chronotype +16.4 min** and reduces activity by 2.8% -- these are coupling coefficients for mental health -> sleep -> activity cascade
4. **Activity and sleep are inversely coupled:** each hour of sleep = 2.1% less active calories; each hour of social jetlag = 2.3% more active calories (counterintuitive -- misaligned students are more active because they're forced out of bed by school)
5. The **phase transition** between school and break states is a clean natural experiment showing how environmental constraints dominate individual biology
