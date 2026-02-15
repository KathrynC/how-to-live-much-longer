# What We Know About the Mitochondrial Aging Parameter Space

## Status: Partial Characterization (2026-02-15)

Based on causal_surgery.json (192 sims) and perturbation_probing.json (8x24 probes).
Discovery tools (sobol, interaction_mapper, reachable_set, competing_evaluators,
temporal_optimizer, multi_tissue_sim) are written but not yet run.

---

## Space Structure

**12D parameter space**: 6 intervention doses (discrete grid, 6^6 = 46,656 points)
+ 6 patient characteristics (continuous). One critical nonlinearity: the heteroplasmy
cliff at het=0.70 (sigmoid steepness 15.0) creates bistability.

## Key Findings

### 1. The Space is Patient-Stratified, Not Uniformly Rough

The landscape is a patchwork — not a single topology:

- **Healthy patients (het < 0.3)**: 0/128 sims reach crisis regardless of intervention.
  The ODE has a healthy attractor. Interventions don't matter much. Boring plateau.
- **Moderate patients (het 0.3-0.6)**: Interventions produce graded, monotonic effects.
  Hill-climbing works. This is where optimization is meaningful.
- **Near-cliff patients (het > 0.6)**: 68.8% of interventions fail. Bistability
  dominates. Timing is irrelevant (r=0.007 between switch-year and outcome).

### 2. Intervention Potency Hierarchy

From perturbation sensitivity (dATP/dparam, averaged across 8 probe vectors):

| Intervention | Sensitivity | Classification |
|---|---|---|
| Transplant | 0.76 | **Primary driver** — Cramer's recommended path |
| Yamanaka | 0.64 | Strong but costly (3-5 MU ATP) |
| Rapamycin | 0.63 | Strong — mTOR/mitophagy axis |
| NAD supplement | 0.38 | Moderate — gated by CD38 degradation |
| Exercise | 0.08 | Near-inert in current ODE |
| Senolytics | 0.04 | Near-inert in current ODE |

17.4x range between strongest and weakest. Strongly anisotropic.

### 3. Dead Zones

~19.8% of sampled space produces ATP < 0.2 (cellular energy crisis), but this is
entirely concentrated in near-cliff patients. The dead zone is not randomly distributed
— it's geographically localized in patient space.

### 4. Robustness Paradox

Successful protocols (high ATP) are 7.4x more robust to parameter perturbation than
failing protocols. Healthy cells with transplanted mtDNA are buffered by redundancy;
sick cells with high ROS are sensitive to any perturbation tipping them toward crisis.

### 5. No Point of No Return (Within the Window)

For near-cliff patients, the correlation between intervention start time and final
ATP is r=0.007 — essentially zero. The system is already committed. Earlier treatment
trends slightly better but not significantly. The cliff's bistability has already
captured the trajectory by the time any intervention begins.

### 6. Two Universal Inerts

Baseline age and baseline NAD level affect 0% of probes (zero sensitivity). This is
suspicious for age — suggests AGE_TRANSITION dynamics at 65 may not propagate strongly
through perturbation mode. Worth investigating with Sobol analysis.

## Landscape Roughness

**CV(ATP) = 0.503** — moderate roughness. For comparison, the Evolutionary Robotics
project's fitness landscape has sign flip rates of 0.58-0.72 between adjacent samples
(effectively CV >> 1.0). The mito landscape is navigable; the ER landscape is fractal
chaos.

The fundamental difference: ODE dynamics are smooth (polynomials, sigmoids, products).
PyBullet contact dynamics are discontinuous (foot-ground collisions create cascading
phase shifts). Smoothness enables gradient structure; contact discontinuities destroy it.

## What's Still Unknown

Pending the 5 discovery tools (~7,800 sims, ~19 min runtime):

1. **Parameter interactions** (Sobol ST-S1): Are effects additive or coupled?
2. **Synergy/antagonism pairs** (interaction_mapper): Which interventions enhance or
   cancel each other?
3. **Reachable outcome space** (reachable_set): What (het, ATP) outcomes are achievable?
   Where is the Pareto frontier?
4. **Transaction rarity** (competing_evaluators): How rare are protocols that satisfy
   ALL clinical criteria simultaneously?
5. **Timing importance** (temporal_optimizer): Does phased dosing outperform constant?
6. **Cross-tissue coupling** (multi_tissue_sim): Does the cardiac cascade create
   systemic fragility?

---

*Generated 2026-02-15. See also: cross_project_gnarliness_comparison.md for the full
Rucker/Wolfram analysis comparing this landscape to the ER atlas.*
