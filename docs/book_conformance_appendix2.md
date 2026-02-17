# Appendix 2 Conformance Matrix (Book -> Model)

Source of truth used here:
- John G. Cramer, *How to Live Much Longer: The Mitochondrial DNA Connection* (forthcoming from Springer Verlag in 2026), especially **Appendix 2**.

Purpose:
- Make model-to-book alignment auditable.
- Distinguish strict matches from intentional approximations and open deviations.
- Record compliance-driven model updates separately from calibration-driven updates.

## Status Legend

- `match`: Implemented as stated or within numerical tolerance.
- `partial`: Implemented with a documented approximation.
- `deviation`: Known mismatch from book statement.
- `untested`: Mapped but not yet covered by executable assertion.

## Claims

| Claim ID | Appendix 2 / Book Claim | Code Mapping | Test Mapping | Status | Notes |
|---|---|---|---|---|---|
| A2-01 | Deletion mutation doubling time is ~11.81 years before age 65 | `constants.py` (`DOUBLING_TIME_YOUNG=11.8`) | `tests/test_book_conformance.py::test_appendix2_doubling_time_constants` | `partial` | Rounded to 11.8 from 11.81. |
| A2-02 | Deletion mutation doubling time is ~3.06 years after age 65 | `constants.py` (`DOUBLING_TIME_OLD=3.06`) | `tests/test_book_conformance.py::test_appendix2_doubling_time_constants` | `match` | Direct value match. |
| A2-03 | Transition between regimes occurs at age 65 | `constants.py` (`AGE_TRANSITION=65.0`) | `tests/test_book_conformance.py::test_appendix2_age_transition_constant` | `match` | Restored per Cramer correction. |
| A2-04 | Deletion-mutated mtDNA has replication advantage (shorter rings replicate faster) | `constants.py` (`DELETION_REPLICATION_ADVANTAGE=1.21`) | `tests/test_book_conformance.py::test_deletions_have_replication_advantage_over_point_pool` | `match` | Directional claim is satisfied. |
| A2-05 | Point mutations do **not** drive cliff; deletion burden drives cliff dynamics | `simulator.py` (`_cliff_factor` called on deletion heteroplasmy) | `tests/test_book_conformance.py::test_cliff_depends_on_deletion_fraction_not_total_mutation_burden` | `match` | ATP cliff term depends on deletion heteroplasmy. |
| A2-06 | Point-mutation accumulation is linear-like relative to deletion clonal expansion | `simulator.py` (`N_point` separate equation, no replication advantage constant) | `tests/test_book_conformance.py::test_deletion_rate_accelerates_with_age` | `partial` | Structural support present; strict linearity is an approximation due coupled dynamics. |
| A2-07 | Deletions become primary aging driver in old age due accelerating clonal expansion | `simulator.py` (`_deletion_rate` transition + advantage) | `tests/test_book_conformance.py::test_deletion_rate_accelerates_with_age` | `match` | Explicit acceleration with age is modeled. |
| A2-08 | “At least 21% faster” replication for large deletions (>3kbp) (book phrasing) | `constants.py` (`DELETION_REPLICATION_ADVANTAGE=1.21`) | `tests/test_book_conformance.py::test_book_minimum_21pct_replication_advantage_compliant` | `match` | Updated on 2026-02-17 for strict Appendix-2 compliance. |
| C7-01 | CD38 degrades NMN/NR at low dose (~40% survival) and apigenin-suppressed protocols can approach 100% survival at max dose | `constants.py` (`CD38_BASE_SURVIVAL=0.4`, `CD38_SUPPRESSION_GAIN=0.6`) | `tests/test_book_conformance.py::test_cd38_survival_endpoints_match_c7_text` | `match` | Formula gives 0.4 at min and 1.0 at max. |
| C7-02 | CD38 gating should create nonlinear NAD-benefit scaling as dose rises | `simulator.py` (NAD term in `derivatives`) | `tests/test_book_conformance.py::test_nad_dynamics_reflect_cd38_gated_nonlinearity` | `match` | High-dose NMN/NR yields stronger NAD derivative than low-dose. |
| A2-09 | Figure-23-style young/old deletion-growth regimes should map to the two doubling scales | `simulator.py` (`_deletion_rate`) | `tests/test_book_conformance.py::test_figure23_regime_rates_match_reported_doubling_scales` | `match` | Checked with tolerance under dynamic smoothing. |

## Open Conformance Questions

1. Should a dedicated empirical-fit test be added against raw Figure 23 trajectories (if Va23 calibration data is checked into repo)?
2. Should point-mutation “linear accumulation” be tested with a reduced coupling mode that isolates `N_point` dynamics?
3. Should we add a strict manuscript-quote test layer for additional non-Appendix chapters (VI-VIII)?

## Implementation Notes

- This matrix is paired with executable checks in `tests/test_book_conformance.py`.
- Any future model change touching Appendix 2 claims should update:
  1. this matrix row status, and
  2. the corresponding test(s).
- Strict enforcement mode:
  - Set `STRICT_BOOK_CONFORMANCE=1` to activate hard-fail CI gates for
    Appendix-2 compliance checks.
- Compliance note:
  - `DELETION_REPLICATION_ADVANTAGE` was changed from `1.10` to `1.21`
    on 2026-02-17 specifically for Appendix-2 conformance.
  - This was an explicit compliance-driven update, distinct from
    independent parameter re-calibration.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This conformance matrix uses John G. Cramer's book as the model-level ground truth for mechanism and parameter claims, especially Appendix 2.
- **Model-claim mapping assumption:** Each claim is translated into code-level checks that capture the intended mechanism direction and scale, not every biological micro-detail.
- **Validation assumption:** Passing conformance tests demonstrates alignment with the encoded interpretation of book claims; it does not by itself establish clinical truth.
