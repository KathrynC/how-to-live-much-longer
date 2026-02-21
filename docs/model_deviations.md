# Model Deviations Report

This document tracks known gaps between implementation and the source-model text
(*How to Live Much Longer*, forthcoming from Springer in 2026; ISBN 978-3-032-17740-7), with a
focus on explicit, testable deltas.

## Compliance Update (2026-02-17)

- **Change made for compliance reasons:** `DELETION_REPLICATION_ADVANTAGE` was
  raised from `1.10` to `1.21` to satisfy Appendix 2 wording ("at least 21%
  faster" for large deletions >3kbp).
- **Why this is documented separately:** this was a **conformance-driven**
  update for source-text compliance, not a fresh parameter optimization pass.
- **Traceability:**
  - Code: `constants.py`
  - Tests: `tests/test_book_conformance.py`
  - Matrix: `docs/book_conformance_appendix2.md`

## Current Deviation Inventory

At present, no unresolved **strict Appendix 2** deviations are tracked in the
conformance suite.

Remaining items are interpretation/calibration questions rather than direct
textual mismatches:

1. `DOUBLING_TIME_YOUNG` uses `11.8` (rounded) vs textual `11.81`.
2. Figure-23 behavior is checked by regime-level rate tests, not full raw-data
   curve-fit residuals.
3. Point-mutation "linear-like" behavior is enforced structurally, but not yet
   isolated in a reduced-coupling experiment.

## CI Modes

- Default project tests:
  - `pytest -q`
- Conformance-only:
  - `pytest -q tests/test_book_conformance.py`
- Strict manuscript conformance gate:
  - `STRICT_BOOK_CONFORMANCE=1 pytest -q tests/test_book_conformance.py`

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** Cramer's forthcoming 2026 book is treated
  as ground truth for model-selection and compliance checks in this project.
- **Deviation definition assumption:** a deviation here means mismatch between
  code behavior and explicit source-text claims, not disagreement with external
  literature.
- **Validation assumption:** conformance tests establish alignment with the
  encoded interpretation of source claims; they do not on their own establish
  clinical validity.
