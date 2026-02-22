# Pattern Language with SystemViz + Lakoff Semantics

This directory contains the integration of Peter Stoyko's SystemViz visual vocabulary with George Lakoff's cognitive semantics (conceptual metaphor theory) for the mitochondrial aging project's protocol pattern language.

## Overview

- **SystemViz**: Structural vocabulary for describing systems (drivers, signals, states, boundaries, relations, domains)
- **Lakoff semantics**: Cognitive grounding of metaphors (grounded vs linking features, Idealized Cognitive Models, metaphor violation detection)
- **Integration**: Dual annotation of protocol patterns with both structural (SystemViz) and cognitive (Lakoff) perspectives

## Files

### Core Files
- `protocol_pattern_language.v1.json` - 7-stage DAG for protocol curation pipeline + 4 experimental Stoyko-based patterns
- `systemviz_lexicon.json` - SystemViz vocabulary (6 categories, 56 terms) from Stoyko's Codex-Text 1-1
- `systemviz_mapping.json` - Mapping of mitochondrial simulator components to SystemViz terms
- `systemviz_integration.py` - Bridge between pattern language and SystemViz lexicon

### Lakoff Integration
- `lakoff_integration.py` - Main implementation: feature layer classification, grounding criteria, ICMs, archetypes, metaphor auditor
- `lakoff_archetypes.json` - Default archetype definitions (conservative, aggressive, transplant_focused, metabolic_optimizer)
- `lakoff_systemviz_crosswalk.json` - Crosswalk mapping between SystemViz terms and Lakoff archetypes
- `lakoff_protocol_classification_test.json` - Test results on protocol dictionary

### Test Files
- `test_lakoff_on_protocols.py` - Test script for classifying protocol records
- `protocol_rewrite_rules.v1.json` - Rewrite rules for protocol classification

## Usage

### Basic Lakoff Archetype Classification

```python
from patterns.lakoff_integration import (
    create_default_archetypes, MetaphorAuditor, 
    extract_features_from_analytics
)

# Load archetypes
library = create_default_archetypes()
auditor = MetaphorAuditor(library)

# Extract features from analytics dict (from analytics.compute_all())
features = extract_features_from_analytics(analytics_dict)

# Audit all archetypes
results = auditor.audit(features)

# Find best match
best_arch = max(results.items(), key=lambda x: x[1]["similarity"])[0]
```

### SystemViz-Lakoff Bridge

```python
from patterns.lakoff_integration import LakoffSystemVizBridge

bridge = LakoffSystemVizBridge()

# Dual annotation of a pattern
dual = bridge.annotate_with_dual_vocabulary("analytics_profile")
print(f"Pattern: {dual['lakoff']['pattern_name']}")
print(f"Grounded ratio: {dual['lakoff']['grounded_ratio']:.2f}")
print(f"Relevant archetypes: {[a['archetype'] for a in dual['lakoff']['relevant_archetypes']]}")

# Crosswalk report
report = bridge.generate_crosswalk_report()
```

### Testing on Protocol Dictionary

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer/patterns
python3 test_lakoff_on_protocols.py
```

## Lakoff Archetypes

Four default intervention strategy archetypes:

1. **Conservative** - Low-risk, maintenance-focused with minimal energy cost
   - Grounding: ATP final > 0.7, total dose < 1.5, benefit-cost ratio > 2.0
   - ICM: Assumes patient relatively healthy (het < 0.3)

2. **Aggressive** - High-intensity aiming for damage reversal
   - Grounding: Total dose > 3.0, delta het < -0.1, crisis delay > 5 years
   - ICM: Assumes sufficient energy reserves (ATP > 0.5) and not too close to cliff

3. **Transplant-focused** - Protocol centered on mtDNA transplant as primary rejuvenation
   - Grounding: Transplant rate > 0.5, deletion het final < 0.4, ATP benefit > 0.1
   - ICM: Assumes deletion heteroplasmy > 0.1 and NAD not declining

4. **Metabolic optimizer** - Optimizes metabolic flexibility via combined interventions
   - Grounding: Exercise > 0.5, NAD supplement > 0.5, ROS-het correlation < -0.3
   - ICM: Assumes metabolic flexibility exists (ROS amplitude < 0.3, ATP CV < 0.2)

## Feature Layer Classification

- **Grounded features** (61.5%): Direct biological measurements (ATP, heteroplasmy, ROS amplitude, NAD slope, etc.)
- **Linking features** (38.5%): Abstract concepts (benefit-cost ratio, correlations, slopes, ratios, etc.)

Following Lakoff Maxim 7: "ground first, link second" - metaphorical labels must be grounded in observable features before cross-domain linking.

## SystemViz Categories → Lakoff Layers

| SystemViz Category | Lakoff Layer | Rationale |
|-------------------|--------------|-----------|
| signal, state | grounded | Signals and states are directly observable/monitorable |
| driver, boundary, relation, domain | linking | Drivers, boundaries, relations, domains are conceptual abstractions |

## Next Steps

1. **Refine archetype grounding criteria** to use more grounded features (per Lakoff Maxim 7)
2. **Expand archetype set** based on protocol dictionary analysis
3. **Integrate with protocol classifier** to provide Lakoff-based metaphor audit
4. **Add visualization** for metaphor violation detection
5. **Apply to pattern language validation** - ensure pattern metaphors are cognitively grounded

## References

- Stoyko, P. (2025). *SystemViz Codex-Text 1-1* (CC-BY 4.0)
- Lakoff, G. & Johnson, M. (1980). *Metaphors We Live By*
- Lakoff, G. (1987). *Women, Fire, and Dangerous Things*
- Cramer, J.G. (forthcoming 2026). *How to Live Much Longer: The Mitochondrial DNA Connection*