# LEMURS Paper #5: Crowdsourcing Goal-Specific Personalized Wellness Practices

**Citation:** Hidalgo, J. E. et al. (2024). Meeting people where they are: Crowdsourcing goal-specific personalized wellness practices. *PLOS Digital Health*, 3(11), e0000650.

---

## Parameters

### Study Design
- N = 992 U.S. adults (Prolific, 2023), cross-sectional survey
- Age: M = 45.77 (SD 15.92), range 19-76
- 5 wellness dimensions: Sleep, Physical, Emotional, Productivity, Social
- 25 curated practices per dimension; participants select 3 of 5 dimensions, then 3-5 practices each

### Personality (BFI-2-XS, 15 items)
| Trait | Mean | SD | Alpha |
|---|---|---|---|
| Extraversion | 2.75 | 0.94 | 0.60 |
| Agreeableness | 3.92 | 0.80 | 0.58 |
| Conscientiousness | 3.73 | 0.98 | 0.70 |
| Negative Emotionality | 2.65 | 1.15 | 0.79 |
| Open-Mindedness | 3.92 | 0.84 | 0.65 |

### Self-Regulation (SSRQ, 17 items, alpha = 0.93)
- Goal Setting: M = 22.35 (SD 4.98)
- Perseverance: M = 10.34 (SD 2.76)
- Decision Making: M = 16.39 (SD 5.17)

### Health Status
- Mental health diagnosis: 43%
- Physical health diagnosis: 50%
- Depression: 23%
- Cardiovascular disease: 5%

### Dimension Selection Rates
| Dimension | Selected |
|---|---|
| Physical Wellness | 67% |
| Emotional Wellness | 66% |
| Sleep | 65% |
| Productivity | 61% |
| Social Wellness | 41% |

---

## Mathematical Relations

### Chi-Squared Tests
| Comparison | chi-sq | p |
|---|---|---|
| Physical Health Dx x Physical Wellness selection | 27.78 | <0.001 |
| Mental Health Dx x Emotional Wellness selection | 0.909 | 0.340 (n.s.) |
| Wellness App Use x Sleep selection | 7.24 | 0.007 |

### Rank Biased Overlap (RBO) Scores
RBO quantifies similarity between rank-ordered practice lists (0 = disjoint, 1 = identical).
- Overall range: 0.62 - 0.99
- Physical & Productivity practices: most similar across demographics (RBO > 0.80)
- Social & Emotional practices: most dissimilar across demographics
- Depression vs no MH condition (Emotional): RBO = 0.71
- Anxiety vs no MH condition (Social): RBO = 0.62

### Top Sleep Practices (n=649)
| Practice | % Selected | Ideal Days/Week |
|---|---|---|
| Sleep in dark, quiet environment | 55.2% | 6.5 (1.5) |
| Consistent sleep/wake times | 54.2% | 5.9 (1.7) |
| Limit caffeine | 42.5% | 5.8 (2.1) |

---

## Complex Systems Structure

### Core-Periphery Practice Structure
Top 3-4 practices per dimension are robust across demographic splits (RBO > 0.60). Divergence occurs in positions 5-10 -- stable core surrounded by demographically sensitive periphery.

### Health Status -> Dimension Selection Gap
Physical health diagnosis: 57% vs 74% select Physical Wellness (chi-sq = 27.78). Creates a potential "poverty trap" -- those who most need practices are least likely to self-identify expertise.

### Condition-Dependent Practice Attractors
- Depression shifts emotional practices from self-directed to externally-supported
- Anxiety shifts social practices from expansive/exploratory to protective/boundary-setting
- Meditation app use reinforces mindfulness-based sleep practices (self-reinforcing loop)

### Social Wellness as Outlier
Selected by only 41% (vs 61-67% for others), lowest RBO scores across demographic splits. Possibly a metastable state more sensitive to individual/contextual variation.

---

## Relevance to Our Work
**Low.** Purely descriptive survey data. No wearables, no longitudinal dynamics, no regression models. However, the finding that 65% of people select Sleep as a priority wellness dimension, and that the top practices (dark/quiet environment, consistent schedule, limit caffeine) align with sleep hygiene recommendations, validates the sleep intervention pathway in our model. The "poverty trap" finding (those with health conditions are less likely to engage with the relevant wellness dimension) has implications for intervention design.
