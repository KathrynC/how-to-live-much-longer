# LEMURS Paper #9: Spatial Patterns and Temporal Trends of Nature Exposure

**Citation:** Bloomfield, L. S. P. et al. (2025). Spatial and temporal patterns of nature exposure among college students: Insights from a randomized controlled trial using GPS-enabled mobile technology. *In Review* (preprint).

---

## Parameters

### Study Design
- N = 315, 4 treatment groups, 15 weeks (Jan 21 - May 12, 2023)
- GPS-tracked NatureDose via NatureDose app
- NatureScore: 1-100 (30+ remote sensing variables, ML model)
- Spatial resolution: census block level

### NatureDose by Group and Location (weekly minutes)
| Group | Ski Mean | Ski % Participants | Lake Mean | Lake % Participants |
|---|---|---|---|---|
| Control | 57.8 | 48.3% | 157.6 | 80.5% |
| Physical Activity | 44.9 | 61.6% | 209.8 | 69.9% |
| **Nature (NBI)** | 63.1 | 53.2% | 21.7 | **87.3%** |
| Therapy | 177.2 | 60.0% | 41.1 | 75.0% |

### Census Block Counts by Group
| Group | Blocks with >=5% participation |
|---|---|
| Control | 232 |
| Physical Activity | 236 |
| **Nature (NBI)** | **298** (+28% vs control) |
| Therapy | 256 |

---

## Mathematical Relations

### Wilcoxon Rank Sum Tests (non-parametric)
| Comparison | W | p |
|---|---|---|
| NBI vs Control (census block count) | 2804 | 0.092 |
| NBI vs Physical Activity | 2234 | 0.055 |

### Lake Area Comparisons
- Nature vs Control: p < 0.05 (NBI more broadly engaged)
- Control vs Therapy: p < 0.05

### Heavy-Tailed Distributions
Large mean-median divergence indicates power-law-like distribution:
- Therapy ski: mean = 177.2, median = 13.7
- Physical Activity lake: mean = 209.8, median = 3.8

---

## Complex Systems Structure

### Spring Break Phase Transition
Week 8 (Spring Break) produces an abrupt geographic dispersion:
- Pre-break: concentrated in Vermont
- During break: dispersed across contiguous US (home zip codes, travel)
- Post-break: re-concentration but with expanded range

### Seasonal Regime Shifts
Three temporal regimes:
1. **Winter (Weeks 1-7):** Ski-mountain dominated off-campus exposure
2. **Transition (Weeks 7-9):** Spring break disruption
3. **Spring (Weeks 10-15):** Lakeshore-dominated exposure

### Emergent Spatial Clustering
Individual GPS behaviors aggregate into coherent geographic clusters (Burlington, ski areas, lakeshores) not prescribed by intervention design. Clusters emerge from interaction of: environmental affordances, seasonal conditions, institutional structure, individual behavior.

### Intervention-Induced Geographic Diversification
NBI group explored 298 census blocks vs. 232-256 for other groups. The intervention didn't prescribe locations but produced broader geographic reach as an emergent group-level property.

### Awareness-Exploration Positive Feedback (inferred)
```
NBI participation -> awareness of nature benefits -> motivation to seek -> exploration of new locations
    -> broader geographic exposure -> reinforced engagement -> back to awareness
```

### Network Structure (implicit bipartite graph)
- Nodes: {participants} x {census blocks}
- Edges: participant recorded NatureDose in block
- NBI group has denser, broader network
- Seasonal transitions rewire the network (ski blocks lose edges, lake blocks gain)

---

## Relevance to Our Work
**Low.** Primarily a spatial/geographic analysis with no health outcome data and no regression models linking nature exposure to physiological or psychological measures. The main value is methodological: it demonstrates that nature-based interventions produce broader geographic exploration as an emergent property, and that spatial exposure patterns show phase-transition-like seasonal regime shifts and heavy-tailed distributions. These patterns could inform a more sophisticated nature-intervention model but provide no coefficients for our ODE.
