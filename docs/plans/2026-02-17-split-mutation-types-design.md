# Design: Split Mutation Types (C11 — Cramer Email 2026-02-17)

## Problem

The current ODE model treats all mtDNA damage as a single pool (`N_damaged`) with a
flat 5% exponential replication advantage. John Cramer's email (2026-02-17) corrects
this based on current molecular biology (Appendix 2, pp.152-155; Va23 data):

1. **Point mutations** grow **linearly** with no replication advantage (same-length
   mtDNA). Sources: ~67% Pol gamma replication errors + ~33% ROS-induced transition
   mutations. Functionally mild -- most don't eliminate ETC subunit genes.

2. **Deletion mutations** grow **exponentially** with size-dependent replication
   advantage ("at least 21% faster" for >3kbp deletions, Va23). These drive the
   heteroplasmy cliff because large deletions remove ETC subunit genes needed for
   oxidative phosphorylation.

3. **ROS is a minor contributor** to mtDNA mutations (~33% of point mutations only),
   not the main damage driver. Pol gamma replication errors during mtDNA replication
   are primary. The "ROS-damage vicious cycle" exists but is weaker than the 1980s
   Free Radical Theory assumed.

## Approach: Append N_point at End (Minimal Blast Radius)

Rather than inserting a new state variable at index 1-2 (which shifts all subsequent
indices and touches 43+ files), we:

- **Rename** `N_damaged` (index 1) to `N_deletion` -- semantically correct since the
  old `N_damaged` was already modeling exponential/deletion dynamics
- **Append** `N_point` at index 7 (new) with linear growth dynamics
- Existing code that reads `state[1]` now gets deletion damage, which is what drives
  the cliff -- **more biologically accurate**, not less

### State Vector (8D)

| Index | Variable | Change |
|-------|----------|--------|
| 0 | N_healthy | unchanged |
| 1 | N_deletion | **renamed** from N_damaged |
| 2 | ATP | unchanged |
| 3 | ROS | unchanged |
| 4 | NAD | unchanged |
| 5 | Senescent_fraction | unchanged |
| 6 | Membrane_potential | unchanged |
| 7 | N_point | **NEW** |

### Dynamics Changes

#### dN_point/dt (NEW equation)

```
point_from_replication = base_replication_rate * POINT_ERROR_RATE * n_h * nad * energy
point_from_ros         = ROS_POINT_FRACTION * ros * gen_vuln * n_h   # ~33% of old damage_rate
mitophagy_point        = mitophagy_rate * POINT_MITOPHAGY_SELECTIVITY * n_pt
                         # low selectivity: point-mutated mitos often have normal delta-psi
```

Point mutations replicate at the SAME rate as healthy copies (no advantage):
```
replication_point = base_replication_rate * n_pt * nad * energy * copy_pressure
```

No replication advantage means linear growth (replication ~= mitophagy at steady state).

#### dN_deletion/dt (REVISED from dN_damaged/dt)

```
deletion_from_slippage   = del_rate * SLIPPAGE_FRACTION * n_h * energy
                           # Pol gamma replication slippage during mtDNA copying
deletion_from_dsb        = del_rate * DSB_FRACTION * n_h * energy
                           # Double-strand break repair errors
replication_del          = base_replication_rate * DELETION_REPLICATION_ADVANTAGE
                           * n_del * nad * energy * copy_pressure
                           # Exponential growth: shorter rings replicate faster
mitophagy_del            = mitophagy_rate * n_del
                           # High selectivity: deletions cause low delta-psi
```

Key change: **deletions do NOT arise from ROS**. They arise from replication errors.

#### ROS coupling (WEAKENED)

Old:
```python
damage_rate = 0.15 * ros * gen_vuln * n_h * ros_sensitivity  # ALL to N_damaged
```

New:
```python
ros_point_rate = ROS_POINT_COEFF * ros * gen_vuln * n_h * ros_sensitivity
# ~33% of the old damage_rate, feeds ONLY into N_point
# No ros_deletion_rate -- ROS does not cause deletions
```

The ROS-damage "vicious cycle" still exists but is attenuated:
damaged mitos -> more ROS -> some additional point mutations -> but NOT more deletions

#### Cliff factor

The cliff is driven by **deletion heteroplasmy**, not total heteroplasmy:

```python
deletion_het = n_del / (n_h + n_pt + n_del)
cliff = _cliff_factor(deletion_het)  # sigmoid at 0.70
```

Point mutations contribute to total copy count but not to the cliff.

#### Total heteroplasmy (for reporting)

```python
total_het = (n_pt + n_del) / (n_h + n_pt + n_del)
```

Reported alongside deletion_het for analytics.

### Copy Number Regulation

Total copy count = N_h + N_pt + N_del (was N_h + N_d).
Copy number pressure unchanged in principle but now applies across 3 pools:
```python
total = n_h + n_pt + n_del
copy_number_pressure = max(1.0 - total, -0.5)
```

### Initial State

The baseline_heteroplasmy patient parameter now represents TOTAL damage.
The point:deletion ratio is derived from age:

```python
# Deletion fraction increases with age (deletions accumulate exponentially)
deletion_fraction = 0.4 + 0.4 * min(max(age - 20, 0) / 70.0, 1.0)
# age 20: 40% deletion, 60% point
# age 55: ~60% deletion, 40% point
# age 90: 80% deletion, 20% point

n_del_0 = het0 * deletion_fraction
n_pt_0  = het0 * (1.0 - deletion_fraction)
n_h_0   = 1.0 - het0  # same as before
```

This keeps the 12D parameter space unchanged.

### New Constants

```python
# Point mutation dynamics
POINT_ERROR_RATE = 0.001           # Pol gamma error rate per replication cycle
ROS_POINT_FRACTION = 0.33          # fraction of ROS damage -> point mutations
ROS_POINT_COEFF = 0.05             # ~33% of old 0.15 damage_rate coefficient
POINT_MITOPHAGY_SELECTIVITY = 0.3  # low: point-mutated mitos look normal to PINK1

# Deletion dynamics
SLIPPAGE_FRACTION = 0.7            # fraction of de novo deletions from Pol gamma slippage
DSB_FRACTION = 0.3                 # fraction from double-strand break misrepair
DELETION_REPLICATION_ADVANTAGE = 1.21  # raised from 1.05; book says >= 1.21

# Deletion fraction by age (for initial state split)
DELETION_FRACTION_YOUNG = 0.4      # age 20: 40% of damage is deletions
DELETION_FRACTION_OLD = 0.8        # age 90: 80% of damage is deletions
```

### Downstream Impact (Minimized)

**Must change (8 files):**
| File | Change |
|------|--------|
| constants.py | STATE_NAMES[1] rename, add index 7, new constants |
| simulator.py | derivatives() split, initial_state() split, stochastic noise |
| disturbances.py | Damage transfer in IonizingRadiation/ChemotherapyBurst |
| analytics.py | Add deletion_het vs total_het metrics |
| tests/test_simulator.py | New state variable assertions |
| tests/test_resilience.py | Update impulse assertions |
| tests/test_grief_bridge.py | Minor: verify N_point doesn't break grief bridge |
| CLAUDE.md + README.md | Documentation updates |

**No change needed (key files preserved):**
- grief_bridge.py: SNS->ROS is still state[3], inflammation->params unchanged
- grief_mito_simulator.py: Works through simulate_with_disturbances, auto-adapts
- grief_mito_viz.py: Reads heteroplasmy from result dict, not raw state indices
- visualize.py: Uses STATE_NAMES dict for labels (auto-updates)
- resilience_viz.py: Parametric index lookups via STATE_NAMES
- zimmerman_bridge.py: Reads result["heteroplasmy"], not raw state

### Calibration Targets

After the refactor, verify these biological properties:

1. **Natural aging (no intervention, age 70):** het should still reach ~0.85-0.95 at
   30 years. Most of the growth is deletions.
2. **Young healthy (age 30, low het):** Point mutations accumulate slowly, deletions
   barely visible. Total het < 0.15 at 30 years.
3. **Cliff behavior:** Still occurs at ~70% deletion heteroplasmy. Point mutations
   alone should never trigger the cliff.
4. **Transplant effectiveness:** Should still outperform NAD (fixes deletions directly).
5. **ROS coupling:** Weaker than before. Antioxidants help less (they only reduce the
   minor ROS->point pathway). This is biologically correct per Cramer.

### Risk: Backwards Compatibility of `result["heteroplasmy"]`

The `heteroplasmy` array in simulation results currently means total damage fraction.
After the refactor, we need to decide: does `result["heteroplasmy"]` mean:
- (a) Deletion heteroplasmy (what drives the cliff) -- breaking change but more useful
- (b) Total heteroplasmy (backwards compatible)

**Decision: (b) Total heteroplasmy** for backwards compatibility. Add
`result["deletion_heteroplasmy"]` for the cliff-driving metric. The cliff factor
internally uses deletion het, but the reported "heteroplasmy" stays total.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
