# Handoff Batch 3: Unified Specification — Full System Expansion

**Date:** 2026-02-19
**Status:** Received, not yet implemented. This is the comprehensive specification superseding/consolidating batches 1 and 2 with significant new content.

## Scope

Complete expansion of the mitochondrial aging simulator into a precision medicine platform. Integrates: wearable sleep optimization, grief psychobiology, polygenic risk (APOE4/FOXO3/CD38), sex as biological variable, lifestyle (diet/alcohol/coffee), nutraceuticals, intellectual engagement, MEF2-mediated neuroprotection, ion channel regulation, epigenetic modifications, synaptic plasticity, cognitive reserve, and Alzheimer's pathology (amyloid-beta/tau).

## New Content Beyond Batches 1 & 2

- **Alzheimer's pathology**: amyloid_burden and tau_pathology as ODE state variables with clearance/production dynamics
- **Cognitive reserve**: ODE-driven CR accumulation with activity-type-dependent growth rates
- **Ion channel regulation**: KCNQ, Kv4, Nav expression driven by MEF2, composite neuronal excitability
- **Coffee effects**: trigonelline NAD+ boost, chlorogenic acid anti-inflammatory, caffeine mitochondrial boost via p27; preparation method and timing modifiers; APOE4 and female-specific benefit multipliers
- **Full MEF2 pathway ODE**: with APOE4-dependent induction multipliers
- **Neuronal excitability**: derived from ion channel expression ratio (Na/K)
- **Resilience scoring**: weighted composite of MEF2 + synaptic gain + CR that buffers against amyloid/tau toxicity
- **Complete simulation flow**: full `run_simulation()` with all 18+ state variables integrated
- **Utility functions**: resilience scoring, cognitive trajectory estimation, scenario comparison
- **Validation scenarios**: 5 specific literature targets (Horner 2020, Ivanich 2025, O'Shea 2024, bereavement cohorts, Nature 2024 coffee/trigonelline)

## Complete State Variable Inventory (18 ODE + 2 derived)

### Existing (8 ODE states)
H, D, P, ATP, ROS, NAD, S, Psi (healthy mtDNA, deletion mtDNA, point mtDNA, ATP, ROS, NAD+, senescent cells, membrane potential)

### New ODE States (10)
| Variable | Symbol | Range | Baseline | Source |
|----------|--------|-------|----------|--------|
| grief_intensity | G | 0–1 | from patient history | Batch 1 |
| gut_health | M | 0–1 | 0.5 (0.3 for APOE4) | Batch 1 |
| MEF2_activity | MEF2 | 0–1 | 0.2 | Batch 3 (expanded) |
| histone_acetylation | HA | 0–1 | 0.2 | Batch 2 |
| synaptic_strength | SS | 0–2 | 1.0 | Batch 2 |
| cognitive_reserve | CR | 0–1 | 0.5 (college-educated) | Batch 3 new |
| amyloid_burden | Ab | 0–2 | age-dependent | Batch 3 new |
| tau_pathology | tau | 0–2 | age-dependent | Batch 3 new |
| KCNQ_expr | — | 1–2 | 1.0 | Batch 3 new |
| Kv4_expr | — | 1–1.8 | 1.0 | Batch 3 new |

Note: Nav_expr (0.6–1.0, baseline 1.0) is also tracked but downregulated by MEF2.

### Derived Variables (not ODE states)
| Variable | Range | Computed From |
|----------|-------|---------------|
| neuronal_excitability | 0–2 | Na_contrib / K_contrib from ion channels |
| memory_index | 0–1 | SS, MEF2, CR, amyloid, tau, resilience |

Also: sleep_quality (0–1) is updated from Oura data or sleep intervention, not an ODE.

## Complete Patient Parameter Additions

### Genetic (3 params)
- `apoe_genotype`: categorical 0/1/2
- `foxo3_protective`: boolean 0/1
- `cd38_risk`: boolean 0/1

### Sex/Reproductive (3 params)
- `sex`: M/F
- `menopause_status`: pre/peri/post
- `estrogen_therapy`: boolean 0/1

### Psychosocial (3 params)
- `grief_duration`: 0–20 years
- `social_support`: 0–1
- `love_presence`: 0–1

### Intellectual (5 params)
- `intellectual_engagement`: 0–1
- `education_level`: high_school/bachelors/masters/doctoral
- `occupational_complexity`: 0–1
- `social_intellectual_engagement`: 0–1
- `cognitive_activity_years`: 0–50

## Complete Intervention Parameter Additions

### Sleep (2 params)
- `sleep_intervention`: 0–1
- `oura_integration`: boolean

### Dietary (6 params)
- `diet_type`: standard/mediterranean/keto
- `fasting_regimen`: 0–1
- `alcohol_intake`: 0–1
- `coffee_intake`: 0–5 cups/day
- `coffee_type`: filtered/unfiltered/instant
- `coffee_timing`: morning_only/afternoon_allowed/evening

### Probiotics (2 params)
- `probiotic_intensity`: 0–1
- `probiotic_blend`: standard/apoe4_targeted/max

### Supplements (11 params)
- nr_dose, dha_dose, coq10_dose, resveratrol_dose, pqq_dose, ala_dose, vitamin_d_dose, b_complex_dose, magnesium_dose, zinc_dose, selenium_dose (all 0–1)

### Intellectual/Therapy (3 params)
- `therapy_intensity`: 0–1
- `support_group_frequency`: 0–1
- `intellectual_engagement_intervention`: 0–1

**Total new intervention params: 24. Total new patient params: 14.**

## Complete Constants Inventory

### Grief Module
```
GRIEF_ROS_FACTOR = 0.3
GRIEF_NAD_DECAY = 0.15
GRIEF_SENESCENCE_FACTOR = 0.1
SLEEP_DISRUPTION_IMPACT = 0.7
SOCIAL_SUPPORT_BUFFER = 0.5
COPING_DECAY_RATE = 0.3
LOVE_BUFFER_FACTOR = 0.2
GRIEF_REDUCTION_FROM_MEF2 = 0.1
```

### Genetic Multipliers (dict)
```
GENOTYPE_MULTIPLIERS = {
    'apoe4_het': {mitophagy: 0.65, inflammation: 1.2, vulnerability: 1.3, grief_sensitivity: 1.3, alcohol_sensitivity: 1.3, mef2_induction: 1.2, amyloid_clearance: 0.7},
    'apoe4_hom': {mitophagy: 0.45, inflammation: 1.4, vulnerability: 1.6, grief_sensitivity: 1.5, alcohol_sensitivity: 1.5, mef2_induction: 1.3, amyloid_clearance: 0.5},
    'foxo3_protective': {mitophagy: 1.3, inflammation: 0.9, vulnerability: 0.9},
    'cd38_risk': {nad_efficiency: 0.7, baseline_nad: 0.8}
}
```

### Sex-Specific
```
FEMALE_APOE4_INFLAMMATION_BOOST = 1.1
MENOPAUSE_HETEROPLASMY_ACCELERATION = 1.05
ESTROGEN_PROTECTION_LOSS_FACTOR = 1.0
```

### Alcohol
```
ALCOHOL_INFLAMMATION_FACTOR = 0.25
ALCOHOL_NAD_FACTOR = 0.15
ALCOHOL_APOE4_SYNERGY = 1.3
ALCOHOL_SLEEP_DISRUPTION = 0.4
```

### Coffee (NEW — Nature Metabolism 2024)
```
COFFEE_TRIGONELLINE_NAD_EFFECT = 0.05
COFFEE_CHLOROGENIC_ACID_ANTI_INFLAMMATORY = 0.05
COFFEE_CAFFEINE_MITOCHONDRIAL_BOOST = 0.03
COFFEE_APOE4_BENEFIT_MULTIPLIER = 1.2
COFFEE_FEMALE_BENEFIT_MULTIPLIER = 1.3
COFFEE_MAX_BENEFICIAL_CUPS = 3
COFFEE_SLEEP_DISRUPTION_THRESHOLD_HOURS = 12
COFFEE_PREPARATION_MULTIPLIERS = {'filtered': 1.0, 'unfiltered': 0.8, 'instant': 0.5}
```

### Diet
```
KETONE_ATP_FACTOR = 0.1
IF_MITOPHAGY_FACTOR = 0.2
KETO_FEMALE_APOE4_MULTIPLIER = 1.3
```

### Probiotics
```
PROBIOTIC_GROWTH_RATE = 0.1
GUT_DECAY_RATE = 0.02
MAX_GUT_HEALTH = 1.0
MIN_NAD_CONVERSION_EFFICIENCY = 0.7
MAX_NAD_CONVERSION_EFFICIENCY = 1.0
```

### Supplements (22 constants — max_effect + half_max for each)
```
MAX_NR_EFFECT=2.0, NR_HALF_MAX=0.5, MAX_DHA_EFFECT=0.2, DHA_HALF_MAX=0.5,
MAX_COQ10_EFFECT=0.15, COQ10_HALF_MAX=0.5, MAX_RESVERATROL_EFFECT=0.15,
RESVERATROL_HALF_MAX=0.5, MAX_PQQ_EFFECT=0.15, PQQ_HALF_MAX=0.5,
MAX_ALA_EFFECT=0.12, ALA_HALF_MAX=0.5, MAX_VITAMIN_D_EFFECT=0.10,
VITAMIN_D_HALF_MAX=0.5, MAX_B_COMPLEX_EFFECT=0.15, B_COMPLEX_HALF_MAX=0.5,
MAX_MAGNESIUM_EFFECT=0.10, MAGNESIUM_HALF_MAX=0.5, MAX_ZINC_EFFECT=0.08,
ZINC_HALF_MAX=0.5, MAX_SELENIUM_EFFECT=0.08, SELENIUM_HALF_MAX=0.5
```

### MEF2 Pathway (Barker et al. 2021, Science Translational Medicine)
```
MEF2_INDUCTION_RATE = 0.15
MEF2_DECAY_RATE = 0.08
EXCITABILITY_SUPPRESSION_FACTOR = 0.7
MEF2_RESILIENCE_BOOST = 0.4
MEF2_MEMORY_BOOST = 0.2
```

### Ion Channels
```
KCNQ: induction=0.3, decay=0.2, max=2.0, baseline=1.0
Kv4: induction=0.25, decay=0.2, max=1.8, baseline=1.0
Nav: induction=-0.2 (downregulated), decay=0.2, min=0.6, baseline=1.0
```

### Epigenetics
```
HA_INDUCTION_RATE = 0.2
HA_DECAY_RATE = 0.05
PLASTICITY_FACTOR_BASE = 0.5
PLASTICITY_FACTOR_HA_MAX = 1.0
```

### Synaptic Plasticity
```
LEARNING_RATE_BASE = 0.3
SYNAPTIC_DECAY_RATE = 0.1
MAX_SYNAPTIC_STRENGTH = 2.0
SYNAPSES_TO_MEMORY = 0.3
BASELINE_MEMORY = 0.5
```

### Cognitive Reserve (Nature 2025)
```
CR_ND_HAZARD_RATIO = 0.82
CR_ATTRIBUTABLE_RISK_FRACTION = 0.16
CR_HIPPOCAMPAL_BETA_LEFT = -0.085
CR_HIPPOCAMPAL_BETA_RIGHT = -0.097
CR_TOTAL_BRAIN_BETA = -0.161
CR_WMH_BETA = 0.045
COGNITIVE_ACTIVITY_RR_PER_POINT = 0.95
CR_APOE4_MULTIPLIER = 1.2
CR_FEMALE_MULTIPLIER = 1.1
CR_GROWTH_RATE_BY_ACTIVITY = {
    'collaborative_novel': 0.10, 'solitary_novel': 0.08,
    'collaborative_routine': 0.05, 'solitary_routine': 0.03
}
```

### Amyloid & Tau
```
AMYLOID_PRODUCTION_BASE = 0.05
AMYLOID_PRODUCTION_AGE_FACTOR = 0.001
AMYLOID_CLEARANCE_BASE = 0.12
AMYLOID_CLEARANCE_APOE4_FACTOR = 0.7
AMYLOID_INFLAMMATION_SYNERGY = 0.2
TAU_SEEDING_RATE = 0.1
TAU_SEEDING_FACTOR = 0.5
TAU_INFLAMMATION_FACTOR = 0.1
TAU_CLEARANCE_BASE = 0.05
AMYLOID_TOXICITY = 0.3
TAU_TOXICITY = 0.5
RESILIENCE_WEIGHTS = {'MEF2': 0.3, 'synaptic_gain': 0.3, 'CR': 0.4}
```

**Total new constants: ~90+**

## ODE System — All New Equations

### 1. Grief (decays via therapy + support + MEF2 feedback)
```python
dG/dt = -(base_decay + therapy*COPING_DECAY_RATE + support*0.2 + MEF2*GRIEF_REDUCTION_FROM_MEF2) * G
```

### 2. MEF2 (induced by intellectual engagement, APOE4 multiplier)
```python
dMEF2/dt = engagement * MEF2_INDUCTION_RATE * apoe_mult * (1-MEF2) - MEF2 * MEF2_DECAY_RATE * (1 - engagement*0.5)
```

### 3. Ion Channels (3 equations, MEF2-driven)
```python
dKCNQ/dt = MEF2 * 0.3 * (2.0 - KCNQ) - (KCNQ - 1.0) * 0.2
dKv4/dt  = MEF2 * 0.25 * (1.8 - Kv4) - (Kv4 - 1.0) * 0.2
dNav/dt  = MEF2 * (-0.2) * (Nav - 0.6) - (Nav - 1.0) * 0.2
```

### 4. Histone Acetylation
```python
dHA/dt = MEF2 * 0.2 * (1-HA) - HA * 0.05
```

### 5. Synaptic Strength
```python
plasticity = 0.5 + 1.0*HA
dSS/dt = 0.3 * engaged * plasticity * (1 - SS/2.0) - 0.1*(SS - 1)
```

### 6. Cognitive Reserve
```python
dCR/dt = engagement * growth_rate_by_type * (1-CR) + (education_boost + occ_complexity*0.1) * 0.05
```

### 7. Amyloid-beta
```python
dAb/dt = (base_production + age_factor*(age-63) - clearance_base*apoe_factor*Ab) * (1 + inflammation*0.2)
```

### 8. Tau
```python
dtau/dt = seeding_rate*Ab*0.5 + inflammation*0.1 - clearance_base*mitophagy*tau
```

### 9. Gut Health
```python
dM/dt = (probiotic*0.1 + diet_effect)*(1-M) - M*0.02 - alcohol*0.125
```

### Derived: Neuronal Excitability
```python
NE = (Nav/1.0) / ((KCNQ/1.0)*(Kv4/1.0))
```

### Derived: Memory Index
```python
MI = 0.5 + (SS-1)*0.3 + MEF2*0.2 + CR*0.2 - effective_pathology
effective_pathology = (Ab*0.3 + tau*0.5) * (1 - resilience)
resilience = min(1.0, 0.3*MEF2 + 0.3*max(0,SS-1) + 0.4*CR)
```

## Coupling to Existing ODE (Modifications)

### In dR/dt (ROS):
```python
R_production *= (1 + grief_intensity * 0.3 * genotype_sensitivity)
```

### In dN/dt (NAD+):
```python
N_degradation += grief_intensity * 0.15 * genotype_sensitivity
# Also: NAD+ conversion modulated by gut_health (0.7 + 0.3*M)
# Also: coffee trigonelline boost, NR supplement boost (Hill function)
```

### In dS/dt (Senescence):
```python
S_accumulation += grief_intensity * 0.1 * genotype_sensitivity
```

### Sleep quality effects on existing variables:
```python
# Poor sleep increases ROS, reduces repair efficiency
```

### Alcohol effects on existing variables:
```python
inflammation *= (1 + alcohol * 0.25 * apoe_sensitivity)
nad_efficiency *= (1 - alcohol * 0.15 * apoe_sensitivity)
```

## New Modules to Create

1. `grief_module.py` — grief ODE, coping dynamics
2. `genetics_module.py` — genotype multiplier lookup, gene×environment interactions
3. `lifestyle_module.py` — diet/alcohol/coffee/fasting effects
4. `supplement_module.py` — Hill-function dose-response for 11 supplements
5. `probiotic_module.py` — gut health ODE, NAD+ conversion efficiency
6. `neuroplasticity_module.py` — MEF2, ion channels, HA, synaptic strength, CR, memory index
7. `alzheimer_module.py` — amyloid and tau ODEs, resilience scoring
8. `oura_integration.py` — wearable data processing, grief detection via HRV drop
9. `personalized_optimizer.py` — multi-objective search across expanded parameter space

## Oura Integration Details

- Maps sleep_score → sleep_quality (0-1)
- HRV 7-day average as inflammation proxy (inverse relationship)
- Resting heart rate as stress marker
- Grief detection: sustained 20% HRV drop over 14 days triggers intervention boost
- Closed-loop feedback: adjusts therapy_intensity and sleep_intervention automatically

## Validation Scenarios (5 targets)

1. **Horner et al. 2020**: APOE4 male, ketogenic diet, cognitive improvement
2. **Ivanich et al. 2025**: Keto effects on female APOE4 microbiome
3. **O'Shea et al. 2024**: APOE4 × lifestyle interactions
4. **Bereavement cohorts**: grief module matches epidemiological outcomes
5. **Nature Metabolism 2024**: coffee trigonelline NAD+ boosting

## Implementation Notes

- Existing RK4 integrator should be extended (spec shows Euler for clarity, but use RK4)
- All new state variables need NumpyEncoder support for JSON serialization
- Maintain Zimmerman protocol compliance: `run()` + `param_spec()` with expanded bounds
- Coffee effects have genotype×sex interactions (APOE4 benefit 1.2x, female benefit 1.3x)
- Cognitive reserve has NO decay — it's cumulative (unlike other state variables)
- APOE4 genotype multiplier lookup needs graceful handling for non-carrier case (apoe_genotype=0)
- Ion channel equations have asymmetric dynamics: KCNQ/Kv4 upregulated, Nav downregulated by MEF2
- Amyloid production is age-dependent (increases after 63), clearance is APOE4-dependent

## Visualization Additions for visualize.py

- Multi-panel trajectory: memory index, amyloid, tau, MEF2 over time
- Heatmaps: genotype × intervention interactions
- Sensitivity tornado plots
- Cognitive reserve accumulation curves
- Grief resolution trajectories with/without interventions

## Relationship to Batches 1 & 2

This batch is the **authoritative specification**. Where it overlaps with batches 1 and 2, this batch takes precedence (it includes additional constants like coffee, ion channels, cognitive reserve, and Alzheimer's pathology not in the earlier batches). Batches 1 and 2 remain useful for the narrative context and design rationale they provide.
