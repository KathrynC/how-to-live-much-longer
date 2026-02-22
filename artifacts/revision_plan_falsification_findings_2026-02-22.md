# Revision Plan: Falsification Findings 1-4

**Date:** 2026-02-22
**Status:** Ready for implementation
**Scope:** Findings 1-4 from the lit search falsification cycle (F5 transplant blocked — contradicts Cramer C8)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Correct 4 parameter/architecture bugs identified by the 3-model LLM literature search consensus, validated against Cramer compliance.

**Architecture:** Constants tweaks + 2 formula rewrites in sleep_trajectory.py + 1 new exercise channel in simulator.py. No new modules, no ODE expansion. All changes are backward-compatible (existing tests pass with updated values or are updated to match).

**Tech Stack:** Python 3.11+, numpy 1.26.4, pytest

---

## Evidence Base

Each finding below was:
1. Identified by simulation experiments (2026-02-22 audit)
2. Literature-searched by 3 local Ollama LLMs (qwen3-coder:30b, deepseek-r1:8b, gpt-oss:20b)
3. Cross-model consensus analyzed (agreement matrices, citation verification)
4. Cramer compliance checked (all 4 findings are SAFE — outside Cramer's book scope)

Full evidence: `artifacts/lit_search/consensus_finding_{1-5}.md`, `artifacts/lit_search/cramer_compliance_check.md`

---

## Task 1: Fix Exercise Parameters (Finding 1)

**Problem:** Exercise is harmful at every dose and age. `EXERCISE_BIOGENESIS_FACTOR=0.03` exactly cancels `EXERCISE_METABOLIC_COST=0.03`, letting the quadratic ROS term (`exercise * 0.03` in `ros_eq`) dominate. All 3 models unanimously reject this — moderate exercise is net beneficial.

**Consensus correction:**
- Biogenesis: 20-40% increase at maximum exercise → `0.08` (midpoint, conservative)
- Metabolic cost: slight reduction to break the exact cancellation → `0.02`
- New channel: exercise→mitophagy enhancement (PGC-1alpha upregulates PINK1/Parkin) → `0.01`

**Files:**
- Modify: `~/how-to-live-much-longer/constants.py`
- Modify: `~/how-to-live-much-longer/simulator.py`
- Modify: `~/how-to-live-much-longer/tests/` (update any tests that hardcode old values)

**Changes:**

### constants.py

Add after the sleep trajectory constants block (~line 771), before the genetic multipliers:

```python
# ── Exercise→Mito coupling constants ─────────────────────────────────────
# Exercise biogenesis factor: fraction of healthy mtDNA copies added per
# year at maximum exercise level (exercise_level=1.0). Literature consensus
# (Holloszy 1967, Hood 2001): chronic exercise increases mitochondrial
# content 20-40% via PGC-1alpha. Factor of 0.08 at full exercise, gated
# by energy_available and copy_number_pressure, yields ~8-15% net increase
# in healthy copies over 5 years of moderate exercise.
# Provenance: [B] Cross-model consensus from 3 LLMs, grounded in Holloszy
# 1967 (landmark) and Hood 2001 (review). Previous value 0.03 created
# exact cancellation with metabolic cost.
EXERCISE_BIOGENESIS_FACTOR = 0.08   # [B] was 0.03 — see finding_1 consensus

# Exercise metabolic cost: additional ATP consumption at maximum exercise.
# Reduced from 0.03 to 0.02 to break the exact cancellation and reflect
# that biogenesis benefit outweighs metabolic cost at moderate levels.
# Provenance: [C] Modeling calibration. The 0.03 created artificial symmetry.
EXERCISE_METABOLIC_COST = 0.02      # [C] was 0.03

# Exercise→mitophagy enhancement: moderate exercise upregulates PINK1/Parkin
# quality control pathway via PGC-1alpha and AMPK signaling. This accelerates
# clearance of damaged mitochondria, complementing biogenesis of new healthy
# copies. Small additive boost to the base mitophagy rate.
# Provenance: [C] Cross-model consensus direction, magnitude is calibrated.
# Laker et al. 2017 (Autophagy): exercise induces mitophagy in skeletal muscle.
EXERCISE_MITOPHAGY_BOOST = 0.01     # [C] NEW channel
```

### simulator.py — exercise biogenesis (line 769)

Replace the hardcoded `0.03` with the new constant:

```python
# Before:
exercise_biogenesis = (exercise * 0.03 * energy_available
                       * max(copy_number_pressure, 0.0)
                       * tissue_mods["biogenesis_rate"])

# After:
exercise_biogenesis = (exercise * EXERCISE_BIOGENESIS_FACTOR * energy_available
                       * max(copy_number_pressure, 0.0)
                       * tissue_mods["biogenesis_rate"])
```

### simulator.py — exercise metabolic cost (line 923)

Replace the hardcoded `0.03`:

```python
# Before:
exercise_cost = exercise * 0.03

# After:
exercise_cost = exercise * EXERCISE_METABOLIC_COST
```

### simulator.py — exercise ROS (line 974)

The exercise ROS term (`exercise * 0.03`) stays at 0.03 — this is the hormetic signal, not a cost parameter. It is balanced by `defense_factor += exercise * 0.2` (line 968). Leave unchanged.

### simulator.py — exercise mitophagy boost (near line 769)

Add exercise→mitophagy channel. Find where `_current_mitophagy_rate` is used (should be near the mitophagy terms). Add after the existing mitophagy computation:

```python
# Exercise-enhanced mitophagy: moderate exercise upregulates PINK1/Parkin
# quality control via AMPK signaling, enhancing clearance of damaged mitos.
exercise_mitophagy = (exercise * EXERCISE_MITOPHAGY_BOOST
                      * n_del * mitophagy_efficiency * energy_available)
```

Then add `exercise_mitophagy` as a sink in the `dn_del` equation and as a corresponding source of cleared copies in `dn_h` (or just a sink in `dn_del` — mitophagy destroys, doesn't convert).

Add `- exercise_mitophagy` to the `dn_del` sum.

### simulator.py — imports

Add `EXERCISE_BIOGENESIS_FACTOR, EXERCISE_METABOLIC_COST, EXERCISE_MITOPHAGY_BOOST` to the imports from constants.

**Tests:**
- Verify exercise_level=0.5 at age 70 now produces HIGHER final ATP than exercise_level=0.0 (was lower before)
- Verify monotonic benefit: exercise 0.0 < 0.25 < 0.5 in ATP at age 70 (may plateau or reverse at very high doses due to ROS)
- Verify exercise_level=1.0 is still net beneficial (not harmful)
- Verify backward compat: simulation with exercise_level=0.0 produces same result as before (no exercise → no change)

---

## Task 2: Boost Sleep Coupling Coefficients (Finding 2)

**Problem:** Sleep effect is 26x weaker than NAD supplementation (+0.0024 vs +0.0633 ATP). Cross-model consensus: the 3 new sleep channels (ROS, NAD drain, membrane) are slightly under-powered.

**Consensus correction:**
- `SLEEP_ROS_COEFF`: 0.04 → 0.06 (literature: 20-40% oxidative stress increase from sleep deprivation)
- `SLEEP_NAD_DRAIN_COEFF`: 0.02 → 0.03 (literature: PARP activation + circadian NAMPT disruption)
- Other 3 channels (inflammation, repair, membrane) keep current values

**Files:**
- Modify: `~/how-to-live-much-longer/constants.py` (2 constant changes)
- Modify: `~/how-to-live-much-longer/tests/` (update any tests that hardcode old values)

**Changes:**

### constants.py (line 749)

```python
# Before:
SLEEP_ROS_COEFF = 0.04     # [B] per unit sleep deficit → ROS boost factor

# After:
SLEEP_ROS_COEFF = 0.06     # [B] per unit sleep deficit → ROS boost factor
                            # Increased from 0.04: Everson et al. 2005 (Sleep)
                            # reports 20-40% oxidative stress increase from chronic
                            # sleep restriction. Cross-model consensus (2026-02-22).
```

### constants.py (line 757)

```python
# Before:
SLEEP_NAD_DRAIN_COEFF = 0.02   # [C] per unit sleep deficit → NAD drain rate

# After:
SLEEP_NAD_DRAIN_COEFF = 0.03   # [C] per unit sleep deficit → NAD drain rate
                                # Increased from 0.02: PARP activation from
                                # oxidative DNA damage + NAMPT circadian disruption.
                                # Ramsey et al. 2009 (Science): CLOCK/BMAL1→NAMPT.
                                # Cross-model consensus (2026-02-22).
```

**No code changes needed** — `sleep_trajectory.py` and `simulator.py` already read these constants. The values propagate automatically.

**Tests:**
- Verify sleep_intervention=0.9 vs 0.1 ATP delta is now larger than before
- The 26x gap should narrow to ~10-18x (still smaller than NAD, which is correct — NAD is a direct supplement, sleep is indirect)

---

## Task 3: Fix APOE4 Sleep Vulnerability Direction (Finding 3)

**Problem:** APOE4 carriers show LESS sleep vulnerability than wild-type in the model, but literature unanimously says they should show MORE. The bug is in `sleep_trajectory.py` line 133:

```python
sleep_repair_factor = 1.0 - (SLEEP_REPAIR_COEFF / mitophagy_eff) * deficit
```

Dividing by smaller `mitophagy_eff` (0.65 for APOE4-het) makes the penalty larger, which is correct for the repair channel. But the NET effect is reversed because APOE4 carriers have less to lose (their baseline mitophagy is already impaired, so the marginal effect of poor sleep is smaller in absolute ATP terms).

**Consensus correction:** APOE4 should AMPLIFY sleep vulnerability across ALL channels, not just repair. The fix has two parts:

### Part A: Fix repair factor formula

The division-based formula creates the "floor effect" paradox. Replace with multiplicative amplification:

```python
# Before (sleep_trajectory.py line 133):
sleep_repair_factor = 1.0 - (SLEEP_REPAIR_COEFF / mitophagy_eff) * deficit

# After:
# APOE4 amplifies sleep repair penalty (lower mitophagy_eff → larger penalty)
apoe4_amplification = 2.0 - mitophagy_eff  # WT(1.0)→1.0x, het(0.65)→1.35x, hom(0.45)→1.55x
sleep_repair_factor = 1.0 - SLEEP_REPAIR_COEFF * deficit * apoe4_amplification
sleep_repair_factor = np.clip(sleep_repair_factor, 0.0, 1.0)
```

### Part B: Add APOE4 amplification to inflammation and ROS channels

APOE4 carriers have impaired glymphatic clearance and increased neuroinflammation from poor sleep (Shokri-Kojori et al. 2018, PNAS). Add genotype gating to channels 1 and 3:

**Files:**
- Modify: `~/how-to-live-much-longer/sleep_trajectory.py`
- Modify: `~/how-to-live-much-longer/constants.py` (add APOE4 sleep amplification constant)
- Modify: `~/how-to-live-much-longer/tests/test_sleep_trajectory.py`

**Changes:**

### constants.py

Add after `SLEEP_AGE_SENSITIVITY_MAX` (line 771):

```python
# APOE4→sleep vulnerability amplification
# [C] APOE4 carriers have impaired glymphatic clearance (Shokri-Kojori et al.
# 2018, PNAS) and increased amyloid accumulation from poor sleep. The
# amplification factor scales with mitophagy_efficiency deviation from 1.0.
# Formula: amplification = 2.0 - mitophagy_eff
#   WT (1.0) → 1.0x (no amplification)
#   Het (0.65) → 1.35x
#   Hom (0.45) → 1.55x
# This ensures APOE4 carriers are MORE vulnerable to poor sleep, not less.
# Cross-model consensus (2026-02-22): all 3 models unanimous on direction.
APOE4_SLEEP_AMPLIFICATION_ENABLED = True  # [C] flag for backward compat testing
```

### sleep_trajectory.py — compute() method

Replace the Channel 1, 2, and 3 computations:

```python
# APOE4 amplification factor (Finding 3 correction)
# WT (mitophagy_eff=1.0) → 1.0x, APOE4-het (0.65) → 1.35x, hom (0.45) → 1.55x
apoe4_amp = 2.0 - mitophagy_eff

# Channel 1: Inflammation (age-modulated, APOE4-amplified)
age_infl_coeff = SLEEP_INFLAMMATION_COEFF + max(age - 40, 0) * SLEEP_INFLAMMATION_AGE_GAIN
inflammation_delta = deficit_infl * age_infl_coeff * sensitivity * apoe4_amp

# Channel 2: Repair factor (APOE4-amplified)
sleep_repair_factor = 1.0 - SLEEP_REPAIR_COEFF * deficit * apoe4_amp
sleep_repair_factor = np.clip(sleep_repair_factor, 0.0, 1.0)

# Channel 3: ROS boost (APOE4-amplified)
ros_boost = deficit * SLEEP_ROS_COEFF * sensitivity * apoe4_amp
```

Channels 4 (NAD drain) and 5 (membrane) are left unchanged — the APOE4 connection to those pathways is weaker in the literature.

**Tests:**
- `test_apoe4_more_vulnerable`: APOE4-het (mitophagy_eff=0.65) with poor sleep produces LARGER inflammation_delta, ros_boost, and LOWER sleep_repair_factor than WT (mitophagy_eff=1.0) at the same deficit
- `test_apoe4_amplification_at_wt`: WT (mitophagy_eff=1.0) produces amplification=1.0 (no change from current behavior for non-carriers)
- `test_apoe4_hom_worse_than_het`: APOE4-hom (0.45) is worse than het (0.65)
- Update existing `test_genotype_gating` to reflect new formula

---

## Task 4: Fix Sleep Deficit Zero-Point (Finding 4)

**Problem:** The parameter resolver degrades outcomes vs raw defaults because `deficit = 1.0 - quality` penalizes age-typical sleep. A 70-year-old with normal sleep (quality≈0.78) gets deficit≈0.22, imposing sleep penalties that don't exist in the raw ODE (which models a typical aging human). The resolver should model DEVIATIONS from age-typical, not absolute distance from perfection.

**Consensus correction:** Restructure so that the epidemiological baseline for any age IS the neutral state. Benefits for sleep better than age-typical, penalties for sleep worse.

**Also:** The current age anchors overestimate decline. At age 70, the model gives quality≈0.675 but literature consensus is 0.78-0.82. Adjust anchors.

**Files:**
- Modify: `~/how-to-live-much-longer/constants.py` (adjust age anchors)
- Modify: `~/how-to-live-much-longer/sleep_trajectory.py` (fix deficit formula)
- Modify: `~/how-to-live-much-longer/tests/test_sleep_trajectory.py`

**Changes:**

### constants.py — sleep age anchors (lines 722-723)

```python
# Before:
SLEEP_AGE_ANCHORS = [20.0, 40.0, 60.0, 80.0]
SLEEP_QUALITY_ANCHORS = [0.95, 0.88, 0.75, 0.60]

# After:
# Recalibrated to match epidemiological consensus (Ohayon 2004, Mander 2017).
# Sleep efficiency: ~90% at 20, ~86% at 40, ~82% at 60, ~75% at 80.
# Normalized to [0,1] scale. Previous values (0.95, 0.88, 0.75, 0.60)
# overestimated decline — gave 0.675 at age 70 vs literature ~0.78.
SLEEP_AGE_ANCHORS = [20.0, 40.0, 60.0, 80.0]
SLEEP_QUALITY_ANCHORS = [0.95, 0.90, 0.82, 0.72]
```

This gives quality≈0.77 at age 70, matching the 78-82% literature consensus.

### sleep_trajectory.py — deficit computation (line 109)

The key architectural change: deficit is measured from age-baseline, not from 1.0.

```python
# Before (line 108-109):
# 4. Sleep deficit (0 = perfect, 1 = no sleep benefit)
deficit = 1.0 - quality

# After:
# 4. Sleep deficit relative to age baseline (0 = age-typical, positive = worse,
#    negative = better than age-typical). This ensures the resolver is neutral
#    when sleep_intervention=0.5 at any age — age-typical sleep is already
#    modeled by the raw ODE's natural aging trajectory.
#    Finding 4 correction (2026-02-22): deficit from age-baseline, not from 1.0.
deficit = max(baseline_q - quality, 0.0)  # penalty only for worse-than-baseline

# Benefit: sleep BETTER than age-baseline provides a small positive effect.
# This is the recovery from intervention — only active when quality > baseline_q.
benefit = max(quality - baseline_q, 0.0)
```

Then update the 5 channel computations to use deficit for penalties and benefit for bonuses:

```python
# Channel 1: Inflammation (penalty for worse-than-baseline, benefit for better)
age_infl_coeff = SLEEP_INFLAMMATION_COEFF + max(age - 40, 0) * SLEEP_INFLAMMATION_AGE_GAIN
inflammation_delta = (deficit - 0.5 * benefit) * age_infl_coeff * sensitivity * apoe4_amp

# Channel 2: Repair factor (penalty for worse, bonus for better)
sleep_repair_factor = 1.0 - SLEEP_REPAIR_COEFF * deficit * apoe4_amp + 0.3 * benefit
sleep_repair_factor = np.clip(sleep_repair_factor, 0.0, 1.0)

# Channel 3: ROS boost (penalty only — good sleep doesn't reduce ROS below baseline)
ros_boost = deficit * SLEEP_ROS_COEFF * sensitivity * apoe4_amp

# Channel 4: NAD drain (penalty only)
nad_drain = deficit * SLEEP_NAD_DRAIN_COEFF * sensitivity

# Channel 5: Membrane penalty (penalty only)
membrane_penalty = deficit * SLEEP_MEMBRANE_COEFF * sensitivity
```

### sleep_trajectory.py — LEMURS deficit (line 120)

Also fix the inflammation-specific deficit:

```python
# Before:
deficit_infl = 1.0 - quality_for_inflammation

# After:
deficit_infl = max(baseline_q - quality_for_inflammation, 0.0)
benefit_infl = max(quality_for_inflammation - baseline_q, 0.0)
```

And update Channel 1 to use `deficit_infl` and `benefit_infl` instead of `deficit` and `benefit`.

**Tests:**
- `test_neutral_at_default_intervention`: sleep_intervention=0.5 with no alcohol at any age → all 5 channels ≈ 0 (neutral)
- `test_good_sleep_benefits`: sleep_intervention=0.9 → inflammation_delta < 0, sleep_repair_factor > 1.0 (before clip)
- `test_poor_sleep_penalties`: sleep_intervention=0.1 → positive inflammation, ROS, NAD drain; low repair
- `test_no_regression_at_age_20`: young patient with default sleep → near-zero effects
- `test_no_regression_at_age_80`: old patient with default sleep → near-zero effects (age-typical is neutral)
- `test_age_anchors_at_70`: verify interpolated quality ≈ 0.77 (not old 0.675)
- Update all existing sleep trajectory tests that assumed `deficit = 1.0 - quality`

---

## Task 5: Update Tests and Integration Verification

**Files:**
- Modify: `~/how-to-live-much-longer/tests/test_sleep_trajectory.py`
- Modify: `~/how-to-live-much-longer/tests/test_parameter_resolver.py`
- Modify: `~/how-to-live-much-longer/tests/test_simulator.py`
- Modify: `~/how-to-live-much-longer/tests/test_resolver_integration.py`
- Modify: `~/how-to-live-much-longer/tests/test_integration_scenarios.py`

**New integration tests:**

1. **Exercise dose-response**: Sweep exercise 0→1 at age 70, verify ATP benefit is positive and roughly monotonic up to moderate levels

2. **Sleep neutrality**: Run resolver with sleep_intervention=0.5, alcohol=0 at ages 50, 70, 80 — verify resolver output ≈ raw ODE output (within 2% ATP tolerance)

3. **APOE4 × sleep interaction**: Compare WT vs APOE4-het at age 70 with poor sleep (intervention=0.1) — verify APOE4 patient has WORSE outcomes (lower ATP, higher het)

4. **Sleep coefficient boost**: Verify sleep_intervention=0.9 vs 0.1 gap is measurably larger than before the coefficient changes

5. **Full regression**: Run `pytest tests/ -v` and verify all tests pass

**Verification commands:**

```bash
cd ~/how-to-live-much-longer

# 1. Full test suite
pytest tests/ -v

# 2. Exercise dose-response
python -c "
from simulator import simulate
for ex in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = simulate(intervention={'exercise_level': ex})
    print(f'exercise={ex:.2f}: ATP={r[\"states\"][-1,2]:.4f} het={r[\"heteroplasmy\"][-1]:.4f}')
"

# 3. Sleep neutrality check
python -c "
from simulator import simulate
from parameter_resolver import ParameterResolver

r_raw = simulate(patient={'baseline_age': 70.0})
resolver = ParameterResolver(
    patient_expanded={'baseline_age': 70.0, 'apoe_genotype': 0, 'sex': 'M'},
    intervention_expanded={'sleep_intervention': 0.5, 'alcohol_intake': 0.0},
)
r_res = simulate(patient={'baseline_age': 70.0}, resolver=resolver)
print(f'Raw ODE:  ATP={r_raw[\"states\"][-1,2]:.4f}')
print(f'Resolver: ATP={r_res[\"states\"][-1,2]:.4f}')
print(f'Delta:    {abs(r_raw[\"states\"][-1,2] - r_res[\"states\"][-1,2]):.4f}')
"

# 4. APOE4 vulnerability direction
python -c "
from simulator import simulate
from parameter_resolver import ParameterResolver

for geno, label in [(0, 'WT'), (1, 'APOE4-het')]:
    resolver = ParameterResolver(
        patient_expanded={'baseline_age': 70.0, 'apoe_genotype': geno, 'sex': 'M'},
        intervention_expanded={'sleep_intervention': 0.1, 'alcohol_intake': 0.0},
    )
    r = simulate(patient={'baseline_age': 70.0}, resolver=resolver)
    print(f'{label}: ATP={r[\"states\"][-1,2]:.4f} het={r[\"heteroplasmy\"][-1]:.4f}')
"

# 5. Scenario comparison (existing script)
python run_scenario_comparison.py
```

---

## Summary of All Changes

| Finding | File | Change | Old → New |
|---------|------|--------|-----------|
| F1 | constants.py | `EXERCISE_BIOGENESIS_FACTOR` | 0.03 → 0.08 |
| F1 | constants.py | `EXERCISE_METABOLIC_COST` | 0.03 → 0.02 |
| F1 | constants.py | `EXERCISE_MITOPHAGY_BOOST` | NEW: 0.01 |
| F1 | simulator.py | Biogenesis uses constant | hardcoded → constant |
| F1 | simulator.py | Metabolic cost uses constant | hardcoded → constant |
| F1 | simulator.py | Exercise mitophagy channel | NEW |
| F2 | constants.py | `SLEEP_ROS_COEFF` | 0.04 → 0.06 |
| F2 | constants.py | `SLEEP_NAD_DRAIN_COEFF` | 0.02 → 0.03 |
| F3 | sleep_trajectory.py | Repair factor formula | division → multiplicative amplification |
| F3 | sleep_trajectory.py | APOE4 amplifies Ch 1,2,3 | NEW: `apoe4_amp = 2.0 - mitophagy_eff` |
| F3 | constants.py | `APOE4_SLEEP_AMPLIFICATION_ENABLED` | NEW: True |
| F4 | constants.py | `SLEEP_QUALITY_ANCHORS` | [0.95,0.88,0.75,0.60] → [0.95,0.90,0.82,0.72] |
| F4 | sleep_trajectory.py | Deficit zero-point | `1.0 - quality` → `baseline_q - quality` |
| F4 | sleep_trajectory.py | Benefit channel | NEW: better-than-baseline rewards |

**Estimated test impact:** ~10-15 existing tests may need updated expected values. ~15-20 new tests added. Total should go from ~631 to ~650.

---

## NOT Changed (Finding 5 — Blocked)

| Finding | Parameter | Proposed Change | Status |
|---------|-----------|----------------|--------|
| F5 | Transplant saturation | Reduce rate, add diminishing returns | **BLOCKED: Contradicts Cramer C8** |

Cramer explicitly directed transplant_rate increase from 0.15 → 0.30 in email C8. The transplant saturation concern should be presented to Cramer as a question, not implemented as a unilateral change. The observation that 10% dose captures 62% of benefit is real and interesting — it suggests Cramer's model of transplant as "primary rejuvenation" operates in a regime where even small doses are powerful, which actually SUPPORTS his thesis.
