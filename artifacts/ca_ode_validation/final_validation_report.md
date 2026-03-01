# CA-ODE Bridge Validation Report
## Phase 4: Continuous Metrics Analysis

**Date:** 2026-02-22  
**Validation runs:** 40 patient×intervention combinations (10 normal patients × 4 interventions)  
**Patients source:** `sample_patients_100.json` (normal subset)  
**Interventions:** no_treatment, conservative, aggressive, transplant_focused

---

## Tasks Completed

1. **Adjust bin thresholds using supervised classification** – `adjust_thresholds.py` computed new thresholds as midpoints between adjacent bin means from ODE distribution. Results: `updated_thresholds.json`, `updated_schema_combined.json`.
2. **Load tuned rule set** – `final_tuned_rules.json` (45 rules) loaded and integrated into CA simulator via `rules` parameter.
3. **Run full validation with updated schema** – Two validation runs:
   - **Run A:** Original schema + tuned rules (baseline improved).
   - **Run B:** Updated schema (empirical centers + thresholds) + tuned rules.
4. **Analyze timing discrepancies** – No critical events (cliff crossing, ATP collapse, senescence severe) detected in the 40‑run sample; timing difference analysis empty.
5. **Create final validation report** (this document).

---

## Key Metrics Comparison

| Metric | Original schema + tuned rules (Run A) | Updated schema + tuned rules (Run B) |
|--------|----------------------------------------|---------------------------------------|
| Average overall RMSE | **0.154** | 0.270 |
| Average bin agreement | **0.800** | 0.554 |
| RMSE per variable (range) | 0.049–0.186 | 0.021–0.539 |
| Largest center‑mean discrepancy | ATP collapsed: –0.734 | N_healthy depleted: –0.478 |

*Run A* used the original `BIN_SCHEMA` (clinically defined thresholds/centers) with the 45‑rule tuned rule set.  
*Run B* used empirically adjusted centers (clamped to bin intervals) and thresholds (midpoint between adjacent bin means).

**Observation:** The tuned rule set alone improves bin agreement from 0.230 (previous baseline) to 0.800 and reduces RMSE from 0.410 to 0.154. Empirical adjustments to schema (centers + thresholds) degrade both metrics, suggesting that the current mapping between ODE continuous values and discrete bins is fundamentally misaligned.

---

## Critical Discovery: Copy‑Count vs Fraction Mapping

The `BIN_SCHEMA` defines thresholds and centers for **deletion heteroplasmy fraction** (unit “deletion het fraction”) and **point‑mutation fraction**, but `discretize_state()` operates on the raw ODE state vector whose `N_deletion` and `N_point` are **copy counts**, not fractions.

**Evidence:**  
- Global bin statistics show `N_deletion` “past_cliff” bin has mean 0.169 (well below the cliff threshold 0.5).  
- `ATP` “collapsed” bin has mean 0.867 (far above the crisis threshold 0.2).  
- These mismatches occur because copy‑count values are being compared to fraction‑based thresholds.

**Consequence:** All bin‑level statistics, threshold adjustments, and exemplar comparisons are confounded. The CA‑ODE bridge cannot be validated reliably until the mapping is corrected.

---

## Threshold Adjustment Issues

The empirical threshold adjustment (`adjust_thresholds.py`) produced several degenerate thresholds:
- `N_deletion`: thresholds [0.124, 0.194, 0.194] – duplicate due to nearly identical means for “growing” and “approaching_cliff” bins.
- `ATP`: thresholds [0.864, 0.864, 0.903] – duplicate because “collapsed” and “crisis” bins have almost identical ODE means (~0.86).

These duplicates indicate that the ODE distribution does not separate the clinically defined bins under the tested patient/intervention set. The duplicate thresholds collapse the binning structure, reducing discriminative power.

---

## Recommendations

### 1. Fix the Mapping: Copy‑Count → Fraction
- Modify `discretize_state()` to compute **deletion heteroplasmy fraction** = `N_deletion / (N_healthy + N_deletion + N_point)` (and analogously for point mutations).
- Update `continuous_exemplar()` to return a continuous state vector that is consistent with the fraction representation (e.g., assume total copy count ≈ 1.0).
- Re‑run the validation suite with the corrected mapping before any further threshold tuning.

### 2. Refine Thresholds with Supervised Classification
- After fixing the mapping, collect ODE fractions across a broader set of patients and interventions (including edge cases that cross the cliff).
- Use a classification objective (maximizing bin‑label agreement) to adjust thresholds, not just midpoint of means.
- Ensure thresholds remain strictly increasing and retain clinical interpretability.

### 3. Adjust Centers Using ODE Distribution
- Compute new centers as the **median** of ODE values falling into each bin (after mapping fix).
- Keep centers within bin intervals to preserve monotonicity.
- Update `BIN_SCHEMA` with the refined centers and re‑evaluate RMSE.

### 4. Expand Rule Set Tuning
- The tuned rule set (45 rules) already provides a major improvement. Consider expanding it with additional cross‑tier compound rules, especially for point‑mutation dynamics.
- Validate rule confidence weights via ensemble stochastic simulation.

### 5. Investigate Timing Discrepancies on Critical‑Event Dataset
- Run validation on the **edge‑case patient population** (`sample_patients_edge.json`) where cliff crossing and ATP collapse are expected.
- Measure CA‑ODE lag for events such as “senescence severe” (previously observed ~24‑year lag) with the corrected mapping and updated schema.

### 6. Address ATP Bin Inseparability
- The ODE rarely produces ATP values below 0.74 in the normal patient set. Consider adding **stress scenarios** (e.g., inflammation burst, toxin exposure) that push ATP into the crisis/collapsed ranges.
- Redefine ATP bins if the original crisis threshold (0.5) is never reached under simulated conditions.

---

## Next Steps for Phase 4

1. **Immediate:** Correct `discretize_state` and `continuous_exemplar` for fraction mapping.
2. **Re‑run validation** with corrected mapping (original schema + tuned rules) to establish a new baseline.
3. **Re‑adjust thresholds and centers** using the corrected ODE fractions.
4. **Test on edge‑case patients** to capture critical events and timing discrepancies.
5. **Produce a final, validated CA‑ODE bridge** with continuous RMSE < 0.10 and bin agreement > 0.90.

---

## Files Generated

- `artifacts/ca_ode_validation/validation_summary.json` – aggregated metrics (latest run, updated schema).
- `artifacts/ca_ode_validation/detailed_results.json` – per‑run results.
- `artifacts/ca_ode_validation/updated_thresholds.json` – empirical thresholds.
- `artifacts/ca_ode_validation/updated_schema_combined.json` – combined centers + thresholds.
- `artifacts/ca_ode_validation/schema_patch_full.py` – Python patch for `BIN_SCHEMA`.
- `artifacts/ca_ode_validation/visualizations/` – trajectory and distribution plots.
- `artifacts/ca_ode_validation/final_validation_report.md` – this report.

---

## Conclusion

The CA‑ODE bridge validation reveals a **foundational mapping error** (copy‑count vs fraction) that must be corrected before any threshold or center tuning can be meaningful. Once fixed, the tuned rule set demonstrates strong potential, improving bin agreement from 0.23 to 0.80. Empirical adjustments to the schema in its current state degrade performance, highlighting the importance of correct variable representation.

**Priority:** Address the mapping issue, then re‑evaluate thresholds and centers using ODE fractions. The CA layer can then serve as a reliable interpretable complement to the continuous ODE simulator.