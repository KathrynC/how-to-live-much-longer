# CA Calibration & Benchmark Summary (2026-02-22)

## Objectives

1. **Calibrate Lakoff archetype classification** using precise grounding criteria from `lakoff_archetypes_adjusted.json`.
2. **Benchmark CA vs ODE fidelity** across diverse patient/intervention scenarios.
3. Consider additional CA rule enhancements based on ODE updates C7 (CD38) and C8 (transplant).

## Accomplishments

### 1. Lakoff Archetype Classification (Calibrated)

- **Updated `ca_analytics.classify_lakoff_archetype`** to load the adjusted archetype definitions and evaluate grounding criteria + ICM violation conditions.
- **Decision logic**: selects archetype with highest percentage of satisfied grounding criteria, filtering out archetypes with any ICM violation.
- **Fallback heuristic**: if best score < 0.5 or no archetype passes ICM constraints, falls back to original attractor‑based heuristic.
- **Testing**: classification now returns `transplant_focused` for default no‑treatment simulation (patient beyond salvage), `aggressive` for aggressive interventions, etc.
- **Integration**: CA analytics includes `lakoff_archetype` field in `compute_ca_analytics()` output.

### 2. CA vs ODE Fidelity Benchmark

- **Script**: `benchmark_ca_fidelity.py` runs 6 patients × 5 intervention profiles = 30 combinations.
- **Metrics**: per‑variable bin agreement, overall agreement (120 steps, quarterly resolution).
- **Results** (see `output/ca_fidelity_benchmark.json`):
  - **Average overall agreement**: 0.047 ± 0.054 (extremely low).
  - **Per‑variable agreement**: all variables < 0.14, worst for N_healthy (0.012), N_deletion (0.008), ATP (0.008).
  - **Distribution**: all 30 runs in “poor” category (<0.5 agreement).
- **Root‑cause analysis** (single‑run debug):
  - Initial discrete state matches perfectly (8/8).
  - By first quarterly step (0.25 yr), CA already diverges dramatically:
    - CA jumps to “approaching_cliff” / “past_cliff” while ODE shows gradual growth.
    - CA collapses ATP to “compromised” / “collapsed” while ODE ATP actually improves (healthy).
    - CA overshoots ROS, senescence, membrane potential.
  - **Qualitative mismatch**: ODE shows gradual improvement or slow decline; CA predicts rapid collapse within a year.
  - **Possible causes**:
    - CA rules are too aggressive (high confidence, cascade effects).
    - Bin thresholds may be too sensitive (e.g., ATP “collapsed” <0.2, but ODE ATP stays >0.79).
    - CA lacks dampening/negative‑feedback loops present in ODE.
    - Rule timing (quarterly steps) may amplify transients.

### 3. Rule Enhancements (Deferred)

- Already added 2 new rules in `ca_rules.py` (C10: deletion acceleration energy crisis, C11: point mutation Pol γ errors).
- Further enhancements for C7 (CD38 gating) and C8 (transplant displacement) were considered but **deferred** because core fidelity issues must be addressed first.

## Conclusions

1. **Lakoff archetype classification is now calibrated** and usable for labeling CA simulation results.
2. **CA fidelity to ODE is currently poor** (<5% bin agreement). The CA is overly pessimistic, predicting rapid collapse where ODE shows stability or gradual change.
3. **CA may still be useful as a qualitative “worst‑case” model** but should not be relied upon for quantitative prediction or intervention optimization without recalibration.

## Recommendations

### Short‑term (next iteration)

- **Adjust bin thresholds** based on ODE trajectory distributions (e.g., raise ATP “collapsed” threshold, widen N_deletion bins).
- **Review rule confidence values** – reduce confidence for catastrophic transitions, add dampening rules that stabilize near‑healthy states.
- **Add “recovery” rules** that allow improvement when intervention parameters are strong (currently missing).
- **Validate CA against ODE attractor classification** – ensure that at least final attractor (healthy_aging / slow_decline / cliff_approaching / point_of_no_return) matches ODE outcome.

### Medium‑term

- **Systematic rule tuning** using ODE trajectory data as training target (e.g., maximize bin agreement via parameter search).
- **Introduce stochastic rule firing** (already implemented in `ca_stochastic.py`) to capture uncertainty – ensemble agreement may be higher.
- **Cross‑project review**: compare with LEMURS CA fidelity (likely similar issues).

### Long‑term

- **Consider replacing CA with a learned discrete‑time Markov model** trained on ODE transitions.
- **Integrate CA as a “narrative generator”** for clinical interpretation, not as a predictive simulator.

## Files Modified / Created

- `ca_analytics.py` – enhanced `classify_lakoff_archetype`, added `_load_lakoff_archetypes`, `_evaluate_criteria`, `_evaluate_violation_conditions`.
- `ca_rules.py` – previously updated with C10/C11 rules (now 34 total).
- `benchmark_ca_fidelity.py` – new benchmarking script.
- `debug_ca_fidelity.py` – diagnostic script.
- `test_ca_archetype_calibration.py` – test script.
- `output/ca_fidelity_benchmark.json` – full benchmark results.
- `artifacts/ca_calibration_benchmark_summary_2026-02-22.md` – this report.

## Next Steps

1. **Prioritize bin‑threshold adjustment** based on ODE quantiles.
2. **Run rule‑sensitivity analysis** to identify which rules cause premature collapse.
3. **Update CLAUDE.md** with CA fidelity caveats and calibration status.

---

*Report generated by opencode on 2026‑02‑22.*