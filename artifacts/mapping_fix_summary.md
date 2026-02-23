# Copy‑Count vs Fraction Mapping Fix Summary

**Date:** 2026-02-22  
**Validation script:** `validate_fixed_mapping.py`  
**Tuned rules:** `final_tuned_rules.json` (45 rules)

## Changes Made

1. **Fixed `discretize_state()` in `ca_schema.py`** – Now computes deletion heteroplasmy fraction (`N_deletion / total`) and point‑mutation fraction (`N_point / total`) instead of raw copy counts. Other variables use raw values (consistent with schema units).

2. **Updated intervention dictionaries** in validation scripts to include all six intervention parameters (using `DEFAULT_INTERVENTION` as base).

## Validation Results (5 normal patients × 4 interventions = 20 runs)

| Metric | Value |
|---|---|
| **Average overall bin agreement** | **0.801** |
| **Average overall continuous RMSE** | **0.154** |
| **Average continuous RMSE per variable** (see JSON) | N_healthy: 0.235, N_deletion: 0.075, ATP: 0.179, ROS: 0.060, NAD: 0.117, Senescent_fraction: 0.107, Membrane_potential: 0.119, N_point: 0.070 |

## Key Discrepancies (center vs ODE distribution mean)

| Variable & Bin | Center | ODE Mean | Diff | Note |
|---|---|---|---|---|
| N_healthy adequate | 0.85 | 0.774 | +0.076 | Center slightly high |
| N_deletion growing | 0.20 | 0.158 | +0.042 | Center slightly high |
| ATP compromised | 0.65 | 0.778 | –0.128 | ATP rarely enters lower bins |
| ATP crisis | 0.35 | 0.794 | –0.444 | (same) |
| ATP collapsed | 0.10 | 0.843 | –0.743 | (same) |
| Membrane_potential intact | 0.85 | 0.653 | +0.197 | Center too high |
| N_point moderate | 0.20 | 0.139 | +0.061 | Center slightly high |

**Note:** ATP “collapsed”, “crisis”, “compromised” bins are **never reached** by normal patients in the ODE (ATP stays >0.74). This indicates that the ATP thresholds (0.2, 0.5, 0.79) may be too low relative to ODE dynamics, or that the normal patient set does not include pathological cases.

## Next Steps (Choose One)

1. **Threshold & center refinement** – Use ODE fractions from edge‑case patients (cliff‑boundary, max‑stress) to recalibrate ATP thresholds and adjust centers for N_healthy, Membrane_potential, N_point.

2. **Expand validation to edge patients** – Rerun validation with `sample_patients_edge.json` (82 edge cases) to capture ATP collapse and cliff‑crossing events, then refine thresholds.

3. **Accept current mapping as sufficient** – The copy‑count vs fraction mapping error is fixed; remaining discrepancies reflect ODE‑vs‑CA dynamics that may be addressed later.

## Files Modified

- `/Users/gardenofcomputation/how-to-live-much-longer/ca_schema.py` – `discretize_state()` now computes heteroplasmy fractions.
- `/Users/gardenofcomputation/how-to-live-much-longer/validate_fixed_mapping.py` – Intervention dictionaries extended to full 6D.

## Validation Output

- `artifacts/ca_ode_validation_fixed_20260222_212730/` – Full validation results (summary, detailed, global stats).

## Recommendations

- **Proceed with step 2** (edge‑patient validation) to obtain a realistic distribution across all bins before adjusting thresholds.
- After edge‑patient validation, compute empirical percentiles for each variable and update `BIN_SCHEMA` thresholds/centers accordingly.
- Keep the tuned rule set (45 rules) unchanged; its performance is already good (0.80 agreement).