# Handoff Batch 1: Expansion Specification

**Date:** 2026-02-19
**Status:** Received, not yet implemented. Additional batches incoming.

## Scope

Major expansion of the mitochondrial aging simulator from 12D (6 intervention + 6 patient) to ~36+ dimensions across 7 new modules. All additions couple into the existing RK4 ODE system via multiplicative modifiers and new state variables.

## New Parameter Dimensions

| Module | Params | Type | Key additions |
|--------|--------|------|---------------|
| Wearable/sleep | 3 | patient + intervention | `sleep_quality`, `sleep_intervention`, `oura_hrv_trend` |
| Grief/psychosocial | 6 | state + patient | `grief_intensity` (state var), `grief_duration`, `sleep_disruption` (state var), `social_support`, `coping_mechanisms`, `love_presence` |
| Genetics/sex | 6 | categorical/boolean | `apoe_genotype` (0/1/2), `foxo3_protective`, `cd38_risk`, `sex` (M/F), `menopause_status`, `estrogen_therapy` |
| Lifestyle | 4 | intervention + categorical | `alcohol_intake`, `diet_type` (standard/mediterranean/keto), `fasting_regimen`, `probiotic_intensity` |
| Supplements | 11+ | intervention | NR, DHA, CoQ10, resveratrol, PQQ, ALA, vitamin D, B complex, magnesium, zinc, selenium |

## ODE Coupling Summary

- **Grief** couples into dR/dt (ROS production), dN/dt (NAD+ degradation), dS/dt (senescence accumulation); has its own decay ODE governed by coping/support/time
- **Genetics** applied as multiplicative modifiers: APOE4 reduces mitophagy efficiency (0.65 het, 0.45 hom) and amplifies inflammation; FOXO3 boosts mitophagy 1.3x; CD38 reduces NAD+ efficiency to 0.7
- **Sex** modifies baselines post-menopause (inflammation +10%, heteroplasmy +5%); APOE4 negates estrogen benefit
- **Alcohol x APOE4** synergy: inflammation and NAD+ damage amplified 1.3x for carriers
- **Supplements** use Hill-function dose-response: `max_effect * dose / (dose + half_max)`
- **Gut health** new state variable (0-1) driven by probiotics, affects NAD+ conversion efficiency (0.7-1.0 range)
- **Love/connectedness** buffers inflammation via `grief_impact_reduction = 1 - love_presence * 0.2`

## New State Variables

Two additions to the ODE system beyond the current 7:
1. `grief_intensity` — decays via coping/support/time
2. `gut_health` — driven by probiotic intensity, decays naturally

## New Modules to Create

1. `grief_module.py` — grief state equations, parameter mapping
2. `genetics_module.py` — genotype multipliers, gene x environment interactions
3. `lifestyle_module.py` — diet, alcohol, fasting effects
4. `supplement_module.py` — dose-response curves for 11+ supplements
5. `probiotic_module.py` — gut health dynamics
6. `oura_integration.py` — wearable data processing, grief detection via HRV
7. `personalized_optimizer.py` — multi-objective search across expanded parameter space

## Existing Files Requiring Modification

- `constants.py` — 50+ new constants (grief, genetics, sex, lifestyle, supplements)
- `ode_system.py` — coupling terms for all new parameters, two new state variables
- `intervention_grids.py` — extend intervention space for supplements, diet, fasting
- `patient_params.py` — genetics, sex, grief, wearable parameters
- `simulate.py` — integrate modules, allow dynamic wearable updates
- `analyze.py` — grief impact metrics, genotype risk scores

## Literature References Cited

- Ivanich et al. 2025 (sex-specific APOE4 effects, keto x female APOE4)
- Anttila et al. 2004 (alcohol x APOE4 synergy)
- Downer et al. 2014 (alcohol x APOE4)
- Horner et al. 2020 (grief scenario validation)

## Key Design Decisions

- Genotype modifiers applied **multiplicatively** (not additively)
- Supplements use **Hill-function** dose-response with diminishing returns
- Grief has its own **ODE** (not static parameter) — it evolves over simulation time
- Gut health is a **bounded state variable** (clamped 0-1)
- Wearable integration supports both simulated and real Oura data
- All new modules maintain the existing Zimmerman protocol (`run()` + `param_spec()`)

## Notes

- This is batch 1 of multiple handoff documents. Do not begin implementation until all batches received.
- Current model on `main` branch has 12D input, 7 state variables, ~262 tests.
- Expansion will need careful cliff-mapping: APOE4 homozygous + grief + alcohol is a candidate new cliff region.
