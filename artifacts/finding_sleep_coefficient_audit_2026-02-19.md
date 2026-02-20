# Finding: Sleep Coefficient Audit — UVM LEMURS Traceability

**Date:** 2026-02-19
**Source:** Audit of sleep-related constants and their provenance

## Summary

Three sleep-related coefficients in the precision medicine expansion lack traceable
citations to published research. The section header references "UVM LEMURS" but the
actual LEMURS study does not provide the coefficient values used.

## Coefficients Under Audit

| Constant | Value | Location | Cited Source | Actual Source |
|----------|-------|----------|-------------|---------------|
| `SLEEP_DISRUPTION_IMPACT` | 0.7 | constants.py:667 | "UVM LEMURS" (section header) | **Unverified** — not from LEMURS |
| `ALCOHOL_SLEEP_DISRUPTION` | 0.4 | constants.py:713 | None | **Modeling assumption** |
| Inflammation effect | 0.05 | parameter_resolver.py:169 | None | **Modeling assumption** |

## What is LEMURS?

**LEMURS** (Lived Experience Measured Using Rings Study) is a longitudinal study at
UVM's Vermont Complex Systems Center (Computational Story Lab), funded by MassMutual.
Led by Chris Danforth and Laura Bloomfield. ~600 first-year students wear Oura rings
to collect temperature, heart rate, breathing rate, and nightly sleep duration.

**Key finding (PLOS Digital Health, 2024):** Sleep measures from Oura rings are
predictive of perceived stress levels — first study to find changes in perceived
stress reflected in sleep data.

**What LEMURS does NOT provide:**
- No data on mitochondrial intervention efficacy
- No coefficients for sleep→inflammation coupling
- No quantification of alcohol's impact on sleep quality
- No data on sleep quality's effect on NAD+ or mitophagy

## Dead Import

`SLEEP_DISRUPTION_IMPACT = 0.7` is imported by `parameter_resolver.py` (line 33)
but **never used in any calculation**. Only `ALCOHOL_SLEEP_DISRUPTION` is used
(line 167). The constant has no effect on the simulation.

## How Sleep Currently Affects the Model

In `parameter_resolver.py` (Step 5, lines 164-169):
```python
sleep_quality = self._intervention_exp.get('sleep_intervention', 0.5)
alcohol_t = float(np.interp(t, self._time_points, self._alcohol_trajectory))
sleep_quality = max(0.0, sleep_quality - alcohol_t * ALCOHOL_SLEEP_DISRUPTION)
patient['inflammation_level'] += (1.0 - sleep_quality) * 0.05
```

**Actual effects:**
1. Sleep quality starts at `sleep_intervention` value (0-1, default 0.5)
2. Alcohol reduces sleep quality by `alcohol_t * 0.4`
3. Poor sleep increases inflammation by `(1.0 - sleep_quality) * 0.05`

**Missing effects** (specified in handoff batch 3, not implemented):
- Sleep→ROS production
- Sleep→repair efficiency
- Sleep→histone acetylation
- Sleep→synaptic strength / memory consolidation

## Zotero Library

The Zotero library contains one sleep-related paper from UVM:
- **Linnell et al. 2020** — "The sleep loss insult of Spring Daylight Savings" —
  same Computational Story Lab, but about Twitter activity patterns, not
  intervention efficacy.

No LEMURS publications are in the Zotero library.

## Provenance

The user confirmed these coefficients were discussed with ChatGPT while waiting for
Claude credits. The values were likely generated in that conversation and are not
traceable to any specific LEMURS data or published finding.

## Recommendations

1. **Flag `SLEEP_DISRUPTION_IMPACT = 0.7` as unused** — either implement it or remove
2. **Add LEMURS citation** — The published PLOS Digital Health paper should be added
   to the Zotero library if we want to cite LEMURS for any sleep-related work
3. **Validate `ALCOHOL_SLEEP_DISRUPTION = 0.4`** — Cross-reference against
   Ebrahim et al. (2013) "Alcohol and sleep" or Colrain et al. (2014) for
   evidence-based coefficient
4. **Validate inflammation effect (0.05)** — Cross-reference against Irwin (2015)
   "Why Sleep Is Important for Health" for sleep→inflammation coupling strength
5. **Consider implementing the missing sleep pathways** — The handoff batch 3 spec
   describes sleep→ROS and sleep→repair pathways that would make sleep a more
   meaningful intervention parameter

## Risk Assessment

| Issue | Severity | Impact |
|-------|----------|--------|
| Dead import (SLEEP_DISRUPTION_IMPACT) | Low | No effect on simulation |
| Unverified alcohol-sleep coefficient | Medium | Affects inflammation in expanded scenarios only |
| Missing sleep→ROS pathway | Medium | Sleep intervention is weaker than specified |
| False attribution to LEMURS | High | Provenance integrity |

The core Cramer ODE is NOT affected — these constants are only consumed by the
precision medicine expansion layer (parameter_resolver.py).
