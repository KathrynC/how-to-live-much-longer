# Grief-Model Patient Profile Recommendations (2026-02-17)

Question: Which grief-model parameters should be considered for addition to Cramer-style patient profiles?

## Recommended additions

- `V_prior` (prior vulnerability)
  - Stable predisposition to prolonged stress dysregulation.
  - Interpretable as a trait-level susceptibility modifier.

- `B` (bond strength)
  - Controls stress amplitude after bereavement events.
  - Helps distinguish biological burden for similar event types.

- `M` (meaning centrality)
  - Captures identity-level impact of loss.
  - Separates event severity from personal significance.

- `D_circ` (expectedness / circumscription of death)
  - Governs shock-versus-adaptation trajectory shape.
  - Useful for acute risk versus chronic burden modeling.

- `E_ctx` (context exposure)
  - Represents cue/reactivation pressure in daily life.
  - Strongly relevant for persistence of stress signals over time.

- `infl_0` (inflammation at grief onset)
  - Directly compatible with the existing inflammation layer.
  - Practical bridge variable from psychosocial stress into mito dynamics.

## Do not duplicate

- `age` should not be added as a new field, because patient profiles already include `baseline_age`.

## Suggested rollout

### Minimal v1 set
- `V_prior`
- `D_circ`
- `E_ctx`

### Optional v2 depth fields
- `B`
- `M`
- `infl_0`

## Rationale

This set adds psychosocial trajectory-shaping information without overloading the core biological profile. The minimal set prioritizes portability and immediate utility in scenario conditioning; the optional set captures richer bereavement phenomenology when needed.
