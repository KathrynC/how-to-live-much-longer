# Falsifier Report: ODE Model Critical Review

**Date:** 2026-02-15
**Agent:** falsifier (Claude Opus 4.6)
**Target:** simulator.py derivatives() function

## Status: FIXES IN PROGRESS

**ACTION REQUIRED: Report these findings to John Cramer.**

The falsifier agent performed a rigorous adversarial review of the 7-state
ODE model and identified 4 critical bugs, 5 major issues, and 6 minor issues.
The critical bugs collectively mean the model's central claim — that
heteroplasmy crossing the 70% cliff causes irreversible ATP collapse — is
imposed by the sigmoid readout, not emergent from the dynamics.

## Critical Issues

### C1. Cliff is cosmetic, not dynamical
The cliff factor (`_cliff_factor(het)`) only enters the ATP health_score.
It does NOT feed back into mtDNA replication, damage rate, or mitophagy.
Changing CLIFF_STEEPNESS from 5 to 50 changes ATP but leaves heteroplasmy
trajectories identical. Biologically, energy collapse should halt
replication and trigger apoptosis.

### C2. N_healthy grows unbounded
Transplant adds a constant 0.2 copies/yr with no upper cap. At max
transplant rate, N_h reaches 3.96 after 30 years. Total mtDNA copy number
is tightly regulated in biology (~100-10,000 per cell).

### C3. NAD supplementation increases heteroplasmy (inverted sign)
NAD drives both healthy and damaged replication, but the damaged term
benefits more (no homeostatic brake on damaged copies). The model predicts
NMN/NR supplementation accelerates aging — contradicting Yoshino et al.
2018, Mills et al. 2016, and the therapeutic intent.

### C4. Universal attractor prevents cliff crossing
All initial conditions converge toward het≈0.55-0.68 regardless of starting
point. A patient at 90% het spontaneously recovers to 68% without
intervention. There is no bistability, no irreversible collapse.

## Major Issues

- M1: Yamanaka repair continues at ATP=0 (no energy gating)
- M2: Damaged mtDNA replicates SLOWER than healthy (the 0.5 factor on
  line 123 inverts the "survival of the smallest" mechanism)
- M3: ROS-damage vicious cycle loop gain ≈ 0.01 (effectively absent)
- M4: N_damaged also exceeds 1.0 (no total copy number conservation)
- M5: Exercise increases heteroplasmy — hormesis adaptation not modeled

## Minor Issues

- m1: Initial ATP uses multiplicative formula, dynamics use additive (79% transient)
- m2: Membrane potential is a slave variable, not independent
- m3: health_score double-counts cliff through psi
- m4: Rapamycin effect over-calibrated by 2-4x at high doses
- m5: Yamanaka cost makes intensities above 0.2 catastrophic
- m6: Age-dependent deletion accumulation is negligible (0.01 scaling)

## Root Cause

No conservation law for total mtDNA copy number, and the cliff does not
feed back into the dynamics that should make it catastrophic. The system
needs:
1. Total copy number constraint (N_h + N_d regulated to ~1.0)
2. ATP-gated replication (no energy → no replication)
3. Damaged replication advantage actually favoring damaged copies
4. Stronger ROS-damage coupling (loop gain approaching 1.0 near cliff)

## Resolution

Rewrite of derivatives() function preserving the same 7-state interface
and all downstream code (analytics, cliff_mapping, visualize, tiqm_experiment).
