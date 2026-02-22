# Mitochondrial Semantic Cellular Automaton — Design Document

**Date:** 2026-02-21
**Status:** Approved
**Approach:** Direct port of LEMURS CA architecture (Approach A)

## Goal

Build a Semantic Cellular Automaton for the mitochondrial aging simulator that discretizes the 8D continuous ODE state into clinically meaningful bins and simulates state transitions using tiered rules. Provides an interpretable complement to the continuous ODE — the same role the LEMURS CA plays for the LEMURS ODE.

## Architecture

6-file mirror of `~/lemurs-simulator/ca_*.py`:

```
ca_schema.py              <- 8-variable bin schema, discretize/exemplar
ca_rules.py               <- 32 tiered rules (6 tiers + cross-tier compounds)
ca_simulator.py           <- single-cell stepper + 4-tissue population grid
ca_analytics.py           <- rule/cascade/attractor/fidelity/epoch metrics
ca_stochastic.py          <- stochastic rule engine + ensemble simulation
ca_zimmerman_bridge.py    <- 3 Zimmerman protocol adapters
ca_visualize.py           <- trajectory heatmap, rule timeline, tissue grid, fidelity
tests/test_ca.py          <- comprehensive test suite
```

Modified: `constants.py` (age_epoch helper), `CLAUDE.md` (CA documentation).

## Bin Schema (ca_schema.py)

8 state variables discretized into 3-4 bins each. Thresholds from `constants.py` biological values.

| Variable | Index | Thresholds | Labels | Centers | Unit | Source |
|----------|-------|-----------|--------|---------|------|--------|
| N_healthy | 0 | [0.3, 0.7] | depleted, reduced, adequate | [0.15, 0.5, 0.85] | normalized | C2 homeostasis |
| N_deletion | 1 | [0.1, 0.3, 0.5] | minimal, growing, approaching_cliff, past_cliff | [0.05, 0.2, 0.4, 0.7] | het fraction | HETEROPLASMY_CLIFF=0.50 |
| ATP | 2 | [0.2, 0.5, 0.8] | collapsed, crisis, compromised, healthy | [0.1, 0.35, 0.65, 0.9] | MU/day | ATP_CRISIS_FRACTION=0.5 |
| ROS | 3 | [0.1, 0.25] | basal, elevated, pathological | [0.05, 0.175, 0.4] | normalized | BASELINE_ROS=0.1 |
| NAD | 4 | [0.3, 0.7] | depleted, declining, robust | [0.15, 0.5, 0.85] | normalized | NAD_DECLINE_RATE |
| Senescent | 5 | [0.1, 0.4] | minimal, emerging, severe | [0.05, 0.25, 0.6] | fraction | SENESCENCE_RATE |
| Psi (ΔΨ) | 6 | [0.3, 0.7] | collapsed, impaired, intact | [0.15, 0.5, 0.85] | normalized | MITOPHAGY_ATP_MIDPOINT |
| N_point | 7 | [0.1, 0.3] | low, moderate, high | [0.05, 0.2, 0.5] | het fraction | POINT_ERROR_RATE |

Total discrete state space: 3 × 4 × 4 × 3 × 3 × 3 × 3 × 3 = 11,664 possible states.

Functions: `discretize_state()`, `continuous_exemplar()`, `bin_index()`, `bin_count()`.

## Rule Table (ca_rules.py)

32 rules in 6 tiers + cross-tier compounds. Same JSON-serializable dict format as LEMURS:
`{tier, name, inputs, context, outputs, confidence, citation}`.

### Tier 1: Energy-Damage Coupling (4 rules)
- deletion_expansion_young: N_del growing+ & young → N_del +1, N_healthy -1 (0.75)
- deletion_expansion_old: N_del growing+ & old → N_del +1, N_healthy -1 (0.90)
- cliff_atp_collapse: N_del past_cliff → ATP -2 (0.95)
- cliff_approaching_warning: N_del approaching_cliff & ATP compromised+ → ATP -1 (0.80)

### Tier 2: ROS-Damage Vicious Cycle (4 rules)
- ros_from_deletions: N_del growing+ → ROS +1 (0.85)
- ros_from_points: N_point moderate+ → ROS +1 (0.70)
- ros_drives_points: ROS elevated+ → N_point +1 (0.75)
- ros_membrane_damage: ROS pathological → Psi -1 (0.85)

### Tier 3: Mitophagy Quality Control (4 rules)
- mitophagy_clears_deletions: Psi collapsed/impaired & ATP compromised+ & rapamycin high → N_del -1 (0.80)
- mitophagy_atp_gated: N_del growing+ & ATP collapsed → no clearance (0.90)
- mitophagy_weak_on_points: N_point moderate+ & rapamycin high → N_point -1 (0.55)
- rapamycin_membrane_benefit: rapamycin high → Psi +1 (0.75)

### Tier 4: Senescence & SASP (4 rules)
- ros_drives_senescence: ROS pathological → Sen +1 (0.80)
- senescent_energy_drain: Sen emerging+ → ATP -1 (0.85)
- senescent_ros_amplification: Sen severe → ROS +1 (0.80)
- senolytics_clear: Sen emerging+ & senolytic high → Sen -1 (0.85)

### Tier 5: NAD+ & Supplementation (4 rules)
- nad_age_decline: NAD robust & transition/old → NAD -1 (0.85)
- nad_supplement_restores: NAD depleted/declining & nad_supplement high → NAD +1 (0.75)
- nad_low_dose_cd38_blocked: NAD depleted & nad_supplement low → no effect (0.80)
- nad_boosts_defense: NAD robust → ROS -1 (0.70)

### Tier 6: Interventions & Transplant (6 rules)
- transplant_adds_healthy: transplant high → N_healthy +1, N_del -1 (0.85)
- transplant_het_penalty: N_del past_cliff & transplant high → reduced effect (0.90)
- exercise_biogenesis: exercise moderate+ → N_healthy +1 (0.75)
- exercise_hormesis: exercise moderate → ROS transient +1, Psi +1 (0.70)
- yamanaka_repairs: ATP compromised+ & yamanaka high → N_del -1, N_point -1 (0.65)
- yamanaka_energy_cost: yamanaka high → ATP -1 (0.90)

### Cross-Tier Compounds (6 rules)
- **point_of_no_return** (ABSORBING): N_del past_cliff & ATP collapsed & ROS pathological & Sen severe → FREEZE (0.95)
- vicious_cycle_lock: ROS pathological & Psi collapsed & ATP crisis/collapsed → N_del +1, ROS +1 (0.90)
- transplant_rescue: N_del approaching_cliff & ATP compromised & transplant high → N_del -1, ATP +1 (0.80)
- cocktail_synergy: NAD robust & Sen minimal & rapamycin high & exercise moderate+ → N_healthy +1, Psi +1 (0.75)
- age_transition_acceleration: old → N_del +1, NAD -1, Sen +1 (0.85)
- young_homeostasis: N_healthy adequate & ATP healthy & young → hold state (0.80)

Conflict resolution: deterministic = highest confidence wins; stochastic = probabilistic gate then confidence-weighted sampling. Point of no return is deterministic even in stochastic mode.

## Simulator (ca_simulator.py)

### Time stepping
- 30 years at quarterly resolution: 120 timesteps, dt = 0.25 years
- Age-based context (replaces LEMURS calendar):
  - young: age < 50
  - transition: 50 ≤ age < 70 (includes near_transition flag at ±5yr from 65)
  - old: age ≥ 70

### Context building
`_build_context(step, patient, intervention, prev_state, curr_state)`:
- Age epoch: young/transition/old (from baseline_age + step * 0.25)
- Intervention levels: rapamycin (none/low/high), nad_supplement (none/low/high), senolytic (none/high), exercise (none/moderate/high), yamanaka (none/high), transplant (none/high)
- Threshold: intervention > 0.5 → high, > 0.2 → low/moderate, else none
- Patient: genetic_vulnerability, metabolic_demand, inflammation_level, tissue_type
- Derived: near_transition, cliff_proximity (how close N_del is to 0.50)

### Single-cell mode
`run_single_cell(patient, intervention, sim_years=30, dt=0.25)`:
- Initialize from ODE: `initial_state(patient)` → discretize
- 120 steps, return trajectory + rule_log + final_state

### Multi-tissue population (4-cell grid)
`run_tissue_grid(patient, intervention, sim_years=30, dt=0.25, tissue_coupling=0.5)`:
- 4 cells: brain (demand=2.0, ros_sens=1.5, biogenesis=0.3), muscle (1.5, 0.8, 1.5), cardiac (1.8, 1.2, 0.5), skin (0.5, 0.5, 1.0)
- Each cell has tissue-specific context modifying rule applicability
- Inter-tissue coupling (3 channels):
  1. **Systemic inflammation**: Sen severe in any tissue → ROS +1 in all (SASP is blood-borne)
  2. **Circulating NAD**: NAD level equilibrates across tissues (systemic supplementation)
  3. **Senolytic clearance**: Senolytics clear senescent cells in all tissues equally
- Transplant is tissue-LOCAL (not shared)

## Analytics (ca_analytics.py)

5 sections (same structure as LEMURS):
1. **rule_stats**: total_firings, unique_rules, top_10, mean_rules_per_step
2. **cascade_stats**: multi-tier chain reactions, max cascade length
3. **attractor_stats**: 4 attractors (healthy_aging, slow_decline, cliff_approaching, point_of_no_return)
4. **fidelity_stats**: CA vs ODE bin agreement per variable per timestep
5. **epoch_diagnostic**: pre/post age-65 transition state comparison (analog of spring break diagnostic)

Attractor classification:
- **Point of no return**: N_del past_cliff & ATP collapsed & ROS pathological & Sen severe
- **Cliff approaching**: N_del approaching_cliff or past_cliff (not all 4 conditions)
- **Slow decline**: ATP compromised or N_del growing
- **Healthy aging**: none of the above

Population analytics: per-tissue attractor classification, tissue divergence metrics.

## Stochastic Engine (ca_stochastic.py)

- `apply_rules_stochastic()`: each rule fires with P = confidence; conflicts resolved by confidence-weighted sampling
- Point of no return is deterministic even in stochastic mode (absorbing)
- `run_single_cell_stochastic(n_trials=100)`: Monte Carlo ensemble
- `compute_ensemble_analytics()`: attractor probabilities, cliff-crossing probability, time-to-crisis distribution

## Zimmerman Bridge (ca_zimmerman_bridge.py)

3 adapters:
- `MitoCASimulator`: single-cell, 12D param_spec
- `MitoTissueSimulator`: 4-tissue grid, 12D + tissue_coupling = 13D
- `MitoCAEnsembleSimulator`: stochastic ensemble, 12D, returns distributional metrics

All flatten CA analytics to scalar dict for Zimmerman compatibility.

## Visualization (ca_visualize.py)

- `plot_ca_trajectory()`: 8-var × 120-step heatmap (bin indices as colors)
- `plot_rule_timeline()`: rule firings by tier across time
- `plot_ca_fidelity()`: CA vs ODE agreement bars per variable
- `plot_tissue_grid()`: 4-panel tissue comparison (attractor state per tissue)
- `plot_cliff_approach()`: N_deletion trajectory through bin boundaries toward cliff

## Design Rationale

**Why direct LEMURS port (not generalized framework):**
~60% of LEMURS CA code is domain-specific (calendar, social coupling, burnout detection). Generalizing prematurely would create a thin framework or complex plugin architecture. Building both CAs first reveals the actual shared patterns for a future refactor.

**Why 4-tissue grid (not NxN population):**
The mito simulator models a single patient's cells, not a population of patients. Inter-tissue coupling is systemic (blood-borne SASP, NAD, senolytics) not social. A 4-cell grid (one per tissue type from TISSUE_PROFILES) captures this naturally.

**Why quarterly timesteps:**
The mito ODE runs 30 years at dt=0.01yr (3000 steps). The CA at quarterly resolution (120 steps) provides interpretable granularity while keeping rule tables manageable. Major biological transitions (age epochs, cliff approach) evolve over years, not days.

**Why this absorbing state:**
The "point of no return" (N_del past_cliff + ATP collapsed + ROS pathological + Sen severe) maps to the bistable trap in the ODE: past the cliff, low ATP impairs mitophagy, damaged mitos persist and produce more ROS, senescent cells drain remaining energy. All repair pathways fail simultaneously. This parallels LEMURS burnout cascade (all 4 conditions met → state freezes).
