# Appendix 2 Conformance Matrix (Book -> Model)

Source of truth used here:
- John G. Cramer, *How to Live Much Longer: The Mitochondrial DNA Connection* (forthcoming from Springer Verlag in 2026), especially **Appendix 2**.

Purpose:
- Make model-to-book alignment auditable.
- Distinguish strict matches from intentional approximations and open deviations.

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
| A2-04 | Deletion-mutated mtDNA has replication advantage (shorter rings replicate faster) | `constants.py` (`DELETION_REPLICATION_ADVANTAGE=1.10`) | `tests/test_book_conformance.py::test_deletions_have_replication_advantage_over_point_pool` | `match` | Directional claim is satisfied. |
| A2-05 | Point mutations do **not** drive cliff; deletion burden drives cliff dynamics | `simulator.py` (`_cliff_factor` called on deletion heteroplasmy) | `tests/test_book_conformance.py::test_cliff_depends_on_deletion_fraction_not_total_mutation_burden` | `match` | ATP cliff term depends on deletion heteroplasmy. |
| A2-06 | Point-mutation accumulation is linear-like relative to deletion clonal expansion | `simulator.py` (`N_point` separate equation, no replication advantage constant) | `tests/test_book_conformance.py::test_deletion_rate_accelerates_with_age` | `partial` | Structural support present; strict linearity is an approximation due coupled dynamics. |
| A2-07 | Deletions become primary aging driver in old age due accelerating clonal expansion | `simulator.py` (`_deletion_rate` transition + advantage) | `tests/test_book_conformance.py::test_deletion_rate_accelerates_with_age` | `match` | Explicit acceleration with age is modeled. |
| A2-08 | “At least 21% faster” replication for large deletions (>3kbp) (book phrasing) | `constants.py` (`DELETION_REPLICATION_ADVANTAGE=1.10`) | `tests/test_book_conformance.py::test_book_minimum_21pct_replication_advantage_known_deviation` | `deviation` | Current model uses conservative 10% not >=21%. Tracked explicitly. |

## Open Conformance Questions

1. Should `DELETION_REPLICATION_ADVANTAGE` be raised from `1.10` to `>=1.21` to strictly reflect Appendix 2 wording?
2. Should a dedicated empirical-fit test be added against Figure 23 trajectories (if raw Va23 calibration data is checked into repo)?
3. Should point-mutation “linear accumulation” be tested with a reduced coupling mode that isolates `N_point` dynamics?

## Implementation Notes

- This matrix is paired with executable checks in `tests/test_book_conformance.py`.
- Any future model change touching Appendix 2 claims should update:
  1. this matrix row status, and
  2. the corresponding test(s).
- Strict enforcement mode:
  - Default run tracks A2-08 as `xfail` (visible mismatch, non-blocking).
  - Set `STRICT_BOOK_CONFORMANCE=1` to activate a hard-fail CI gate for
    the >=21% replication-advantage requirement.
