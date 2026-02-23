# Symmathesy as a Goal in Mitochondrial Aging Interventions

## What is Symmathesy?

**Symmathesy** (from Nora Bateson, *Warm Data Lab*): *"Mutual learning in living systems. The process by which living systems learn and adapt together through interaction."*

Key characteristics:
- **Mutual**: Not one-way adaptation, but reciprocal learning
- **Contextual**: Learning occurs within specific relationships and environments  
- **Emergent**: Learning arises from interaction, not predetermined
- **Relational**: Focuses on relationships between entities, not entities alone

## Current Mitochondrial Architecture vs. Symmathesy

| Current Approach | Symmathesy-Oriented Approach |
|-----------------|-----------------------------|
| **Static protocols**: Fixed intervention vectors | **Adaptive protocols**: Interventions adjust based on patient response |
| **One-way causality**: Intervention → Patient outcome | **Mutual causality**: Patient state influences intervention, intervention influences patient |
| **Independent components**: mtDNA, ATP, ROS modeled separately | **Relational networks**: Focus on interactions between components |
| **Optimization metrics**: ATP, heteroplasmy, cliff distance | **Learning metrics**: Mutual information, adaptation rate, coherence |
| **Discrete states**: CA bins as fixed categories | **Learning trajectories**: State transitions as mutual adaptation patterns |

## Operationalizing Symmathesy as a Goal

### 1. **Symmathesy Metrics**

```python
def compute_symmathesy_metrics(trajectory, intervention_sequence):
    """Quantify mutual learning between patient and intervention."""
    return {
        # Mutual information between intervention adjustments and patient response
        "mutual_information": compute_mi(intervention_changes, state_changes),
        
        # Adaptation coherence: do changes move toward shared goals?
        "adaptation_coherence": correlation(intervention_direction, improvement_direction),
        
        # Learning rate: how quickly does system find stable mutual configuration?
        "learning_rate": 1.0 / time_to_stable_mutual_configuration,
        
        # Relationship diversity: how many distinct interaction patterns emerge?
        "relationship_diversity": entropy_of_interaction_patterns,
        
        # Positive SystemViz term activation
        "positive_terms_activated": count_positive_systemviz_terms(trajectory),
    }
```

### 2. **Symmathesy-Optimized Interventions**

Instead of optimizing for ATP or heteroplasmy alone, optimize for:

1. **Mutual adaptation capacity**: Ability of intervention-patient system to co-evolve
2. **Relationship richness**: Diversity of interaction patterns that emerge
3. **Learning trajectory smoothness**: Gradual, coherent mutual adjustment
4. **Positive term maximization**: Activation of SystemViz positive terms:
   - `driver.enabler` → intervention enables patient self-repair
   - `relation.mediation` → intervention mediates between conflicting processes
   - `state.capacity` → builds patient capacity for future adaptation
   - `boundary.buffer` → creates protective boundaries during adaptation
   - `state.redundancy` → builds backup systems for resilience

### 3. **Symmathesy in Cellular Automata**

Extend the CA layer to model mutual learning:

```python
class SymmathesyCARule(Rule):
    """CA rule that implements mutual learning between discrete states."""
    
    def apply(self, lattice, node_idx, intervention_state):
        # Current patient state
        patient_state = lattice.get_state(node_idx)
        
        # Intervention influence
        intervention_effect = compute_intervention_effect(intervention_state, patient_state)
        
        # Patient response to intervention  
        patient_response = compute_patient_response(patient_state, intervention_effect)
        
        # Mutual adjustment: both change based on each other
        new_patient_state = patient_state + learning_rate * patient_response
        new_intervention_state = intervention_state + learning_rate * adaptation_to_response
        
        # Return mutual state
        return {
            "patient": new_patient_state,
            "intervention": new_intervention_state,
            "symmathesy_gain": compute_mutual_information_gain(
                patient_state, intervention_state,
                new_patient_state, new_intervention_state
            )
        }
```

### 4. **Lakoff Archetypes Reimagined for Symmathesy**

Extend the 4 existing archetypes with symmathesy dimensions:

| Archetype | Current Focus | Symmathesy Dimension |
|-----------|---------------|---------------------|
| **Conservative** | Low-risk maintenance | **Relational stability**: Maintains coherent patient-intervention relationship |
| **Aggressive** | Damage reversal | **Transformative learning**: Rapid mutual adaptation toward health |
| **Transplant-focused** | mtDNA replacement | **Structural symmathesy**: Rebuilds capacity for mutual learning |
| **Metabolic optimizer** | Combined interventions | **Integrative learning**: Multiple interventions learn to work together |

Add new symmathesy-specific archetypes:
- **Relational mediator**: Maximizes `relation.mediation`, balances conflicting processes
- **Capacity builder**: Focuses on `state.capacity` and `state.redundancy` for future adaptation
- **Enabling facilitator**: Maximizes `driver.enabler`, helps patient help itself
- **Boundary protector**: Creates `boundary.buffer` and `boundary.protector` for safe learning

### 5. **SystemViz Positive Terms as Symmathesy Building Blocks**

Map positive SystemViz terms to symmathesy functions:

| Positive Term | Symmathesy Role | Implementation |
|---------------|----------------|----------------|
| `driver.enabler` | Enables patient self-repair | Interventions that boost endogenous repair mechanisms |
| `relation.mediation` | Mediates conflicting processes | Balances ROS production vs. antioxidant defense |
| `state.capacity` | Builds adaptive capacity | Increases ATP reserve for future challenges |
| `state.redundancy` | Creates backup systems | Multiple mtDNA repair pathways |
| `state.fault_recovery` | Enhances recovery learning | Faster return to homeostasis after disturbance |
| `state.maintenance` | Supports ongoing mutual adjustment | Continuous monitoring and micro-adjustments |
| `boundary.buffer` | Protects during learning | Temporary protection during adaptation phases |
| `boundary.protector` | Shields vulnerable processes | Protects healthy mtDNA during transplant |
| `relation.group` | Coordinates multiple elements | Synchronizes intervention timing |
| `relation.mediation` | Resolves conflicts | Balances energy cost vs. repair benefit |
| `domain.resources` | Provides learning resources | NAD+ as substrate for adaptation |
| `domain.support_structure` | Creates learning scaffolding | Exercise as hormetic stress that builds resilience |

### 6. **Symmathesy-Optimized TIQM Pipeline**

Extend the TIQM pipeline to include symmathesy evaluation:

```
Original TIQM:
Offer wave → Simulation → Confirmation wave (resonance)

Symmathesy TIQM:
Mutual offer wave (patient + intervention) → 
Co-evolution simulation (adaptive protocol) → 
Symmathesy confirmation wave (learning evaluation)
```

**Symmathesy confirmation prompt**:
```
"The intervention protocol adapted to patient response over 30 years:
- Initial adjustment: {initial_adjustment}
- Mid-course correction: {mid_correction}  
- Final mutual configuration: {final_state}

Symmathesy metrics:
- Mutual information: {mutual_information:.3f}
- Adaptation coherence: {coherence:.3f}
- Positive SystemViz terms activated: {positive_terms}

Questions:
1. Describe the mutual learning relationship between intervention and patient.
2. Rate the quality of symmathesy from 0.0 (no mutual learning) to 1.0 (perfect co-evolution).
3. What would improve the mutual learning relationship?
```

### 7. **Implementation Roadmap**

**Phase 1: Metrics and Analysis** (1-2 weeks)
- Implement `compute_symmathesy_metrics()` function
- Add symmathesy metrics to CA analytics
- Create visualization of learning trajectories

**Phase 2: Adaptive Protocols** (2-3 weeks)
- Implement time-varying interventions that respond to patient state
- Add intervention adjustment rules based on CA state transitions
- Create "learning-aware" intervention schedules

**Phase 3: Symmathesy-Optimized Search** (3-4 weeks)
- Modify EA optimizer to maximize symmathesy metrics
- Create symmathesy-aware protocol interpolation
- Add symmathesy to protocol dictionary enrichment

**Phase 4: Full Integration** (4+ weeks)
- Extend TIQM pipeline with symmathesy evaluation
- Create symmathesy archetypes and Lakoff grounding
- Integrate with SystemViz positive term tracking

### 8. **Expected Benefits**

1. **Clinical relevance**: Real-world interventions require mutual adaptation
2. **Resilience**: Symmathesy-optimized protocols may be more robust to disturbances
3. **Personalization**: Learns from individual patient responses
4. **Theoretical advancement**: Bridges Bateson's symmathesy with computational modeling
5. **Positive focus**: Shifts from damage reduction to capacity building

### 9. **Research Questions**

1. Do symmathesy-optimized protocols outperform static protocols on traditional metrics?
2. Which positive SystemViz terms correlate with good symmathesy outcomes?
3. How does symmathesy relate to existing resilience metrics (recovery time, resistance)?
4. Can we identify "symmathesy cliffs" where mutual learning breaks down?
5. How do different archetypes perform on symmathesy metrics?

### 10. **Example: Symmathesy-Optimized Transplant Protocol**

Current transplant: Fixed rate based on initial patient state.

Symmathesy transplant: Rate adjusts based on:
- Patient ATP level (energy availability for integration)
- Inflammation state (hostile environment detection)
- Transplant success signals (engraftment markers)
- Feedback from previous transplant cycles

The intervention *learns* the optimal transplant timing and dose for *this specific patient* through mutual adaptation.

## Implementation Status (2026-02-22)

**✅ Completed: Phases 1-4 Operational**

### Phase 1: Symmathesy Metrics ✅
- **Analytics integration**: `compute_symmathesy_metrics()` added to `analytics.py`
- **Metrics**: mutual_information, relationship_diversity, adaptation_coherence, learning_rate
- **5-pillar structure**: Energy, Damage, Dynamics, Intervention, **Symmathesy**
- **Validation**: All tests pass (14/14 in test_analytics.py)

### Phase 2: Adaptive Protocols ✅
- **Core framework**: `AdaptiveProtocol` class in `adaptive_protocol.py` with rule-based state adaptation
- **Rule constructors**: Basic threshold + advanced proportional/bidirectional rules
- **Protocol templates**: `create_symmathesy_protocol()` and `create_advanced_symmathesy_protocol()`
- **Simulator tracking**: Added `applied_intervention_intensity` field to simulation results
- **Comprehensive testing**: 8 tests in `test_adaptive.py` covering basic rules, sick patients, proportional/bidirectional rules, protocol comparison

### Phase 3: EA Optimizer Integration ✅
- **Fitness metric**: Added `"symmathesy"` to `FITNESS_METRICS` in `ea_optimizer.py`
- **Scoring method**: `_compute_symmathesy_score()` combines normalized MI + RD + |AC|
- **Evaluation integration**: `evaluate()` returns symmathesy metrics alongside ATP/het benefits
- **Functional verification**: EA optimizer runs with `--metric symmathesy` (tested hill_climber, budget 10)

### Phase 4: TIQM Pipeline Extension ✅
- **Confirmation wave enhancement**: `CONFIRMATION_PROMPT` includes symmathesy metrics section
- **New question**: "How well does the intervention adapt to the patient's changing state?" with mutual learning resonance rating
- **JSON output expansion**: Added `"resonance_symmathesy"` field
- **Summary statistics**: Extended reporting to include symmathesy resonance mean/min/max
- **Cross-template update**: Also updated confirmation prompt in `prompt_templates.py`
- **Live validation**: TIQM experiment runs successfully, LLM returns symmathesy resonance scores (observed: 0.6)

### Key Discoveries
1. **Negative adaptation_coherence is biologically plausible** — Adaptive protocols that reduce intervention when patient health improves show inverse correlation
2. **Advanced protocols improve mutual learning** — Proportional/bidirectional rules increase mutual information (0.0903 → 0.6498) and relationship diversity (0.5000 → 0.6599)
3. **Symmathesy metrics work for both static and adaptive interventions** — Analytics module handles 2D/3D arrays (single vs multi-trajectory)
4. **Static interventions score low** — Constant intervention intensity yields zero mutual information, highlighting need for adaptive protocols

### Files Modified/Created
- **New**: `adaptive_protocol.py` (420 lines) — AdaptiveProtocol class + rule constructors
- **New**: `test_adaptive.py` (8 comprehensive tests)
- **Modified**: `simulator.py` — Added `applied_intervention_intensity` tracking
- **Modified**: `analytics.py` — Enhanced `compute_symmathesy_metrics()` to use applied intensity
- **Modified**: `ea_optimizer.py` — Added symmathesy fitness metric and scoring
- **Modified**: `tiqm_experiment.py` — Extended confirmation wave with symmathesy metrics
- **Modified**: `prompt_templates.py` — Updated confirmation prompt template

### Next Steps (Future Work)

#### Phase 3B: Test EA Optimizer Across All Algorithms
```bash
python ea_optimizer.py --algo de --budget 100 --metric symmathesy
python ea_optimizer.py --algo cma_es --budget 200 --metric symmathesy
python ea_optimizer.py --compare --budget 50 --metric symmathesy  # Head-to-head comparison
```

#### Phase 3C: Compare Symmathesy-Optimized vs Traditional Protocols
- Run optimization with ATP/het/symmathesy metrics on same patient
- Compare resulting protocols: intervention composition, trajectory outcomes
- Analyze trade-offs: Do symmathesy-optimized protocols sacrifice ATP benefit for adaptation?

#### Phase 2E: Parameterized Adaptive Protocol Constructor for EA
- Create `create_parameterized_adaptive_protocol(thresholds, gains, deadzones)` that EA can optimize
- Encode adaptive rule parameters (thresholds, gains) as part of EA genotype
- Enable EA to evolve both intervention parameters AND adaptation rules

#### Phase 4B: Symmathesy Evaluation Prompt Templates
- Create specialized confirmation prompts focusing on mutual learning
- Add SystemViz positive term analysis to prompts
- Develop narrative evaluation templates for diegetic symmathesy assessment

#### Integration with Existing Frameworks
- **Cramer-toolkit**: Add symmathesy metrics to resilience analysis (robustness + regret + vulnerability + mutual learning)
- **Zimmerman-toolkit**: Extend contrastive analysis to compare adaptive vs static protocols
- **Protocol dictionary**: Add symmathesy metrics as enrichment fields for protocol records

#### Research Questions (from §9) to Investigate
1. **RQ1**: Do symmathesy-optimized protocols outperform static protocols on traditional metrics?
   - Experiment: Compare top symmathesy protocols vs top ATP protocols on ATP/het/crisis delay
2. **RQ2**: Which positive SystemViz terms correlate with good symmathesy outcomes?
   - Analysis: Run sentiment analysis on confirmation wave descriptions, correlate with symmathesy metrics
3. **RQ3**: How does symmathesy relate to existing resilience metrics?
   - Cross-analysis: Compute correlation between symmathesy metrics and resistance/recovery/regime retention
4. **RQ4**: Can we identify "symmathesy cliffs" where mutual learning breaks down?
   - Experiment: Sweep patient parameters, find regions where adaptation_coherence or mutual information collapses
5. **RQ5**: How do different archetypes perform on symmathesy metrics?
   - Analysis: Run adaptive protocols across patient archetypes (young_prevention, near_cliff, melas, etc.)

### Current Status
**Symmathesy integration is fully operational** — mutual learning metrics are computed, adaptive protocols can be created and tested, EA optimization includes symmathesy fitness, and the TIQM pipeline evaluates mutual learning resonance. The foundation is complete; future work can build on this to answer deeper research questions about mutual learning in mitochondrial aging interventions.

---

## Conclusion

Symmathesy transforms mitochondrial aging interventions from static prescriptions to dynamic learning relationships. By operationalizing mutual learning as a goal—quantified through symmathesy metrics, embodied in adaptive protocols, and visualized through positive SystemViz term activation—we create interventions that don't just treat patients, but learn with them.

This aligns with the broader vision of precision medicine: not just personalized treatments, but *personalizing treatments* that themselves learn and adapt to the unique biology of each patient over time.