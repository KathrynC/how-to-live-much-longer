# EA Exploration Summary (2026-02-17)

## Scope
- Algorithm comparison (`--compare`) on `default` and `near_cliff_80` patients.
- Deep CMA-ES optimization + protocol transfer across 4 patient profiles:
  - `default`, `near_cliff_80`, `post_chemo_55`, `melas_35`
- Landscape analysis (`--landscape`) for `default` and `near_cliff_80`.

## Key findings

### 1) CMA-ES is the best-performing optimizer in this setup
Across 300-eval head-to-head comparisons:
- `default`: CMA-ES ranked #1 (`fitness=0.292335`)
- `near_cliff_80`: CMA-ES ranked #1 (`fitness=0.449423`)

ES was a close second in both cases.

### 2) The EA converges to the same protocol family across profiles
CMA-ES best protocols (400 evals each) were all near:
- high `rapamycin_dose`
- high `nad_supplement`
- high `senolytic_dose`
- high `transplant_rate`
- near-zero `yamanaka_intensity`
- near-zero `exercise_level`

Representative best vectors:
- `default`: `[rapa=1.0, nad=1.0, seno=1.0, yama=0.0, transplant=1.0, exercise=0.0]`
- `near_cliff_80`: `[0.995, 1.0, 0.945, 0.0, 1.0, 0.0]`
- `post_chemo_55`: `[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]`
- `melas_35`: `[1.0, 1.0, 0.787, 0.0, 0.671, 0.001]`

### 3) Protocol transfer is strong (one family generalizes well)
Transfer-fitness matrix (source-optimized protocol -> destination patient) shows minimal degradation between source profiles, especially among `default`, `near_cliff_80`, and `post_chemo_55` optimized protocols.

Selected values:
- Default-optimized protocol on near-cliff patient: `0.4504`
- Near-cliff-optimized protocol on default patient: `0.2919`
- Post-chemo-optimized protocol mirrors default almost exactly.

The `melas_35`-optimized protocol is slightly more conservative on transplant/senolytics and performs slightly worse out-of-domain.

### 4) Landscape is moderately rugged but not chaotic
Budget 300 landscape probes:
- `default`: roughness `0.2937`, mean gradient magnitude `0.4550`, sign-flip rate `0.0000`
- `near_cliff_80`: roughness `0.2868`, mean gradient magnitude `0.4690`, sign-flip rate `0.0000`

Interpretation: optimization is nontrivial (rugged), but local gradients are directionally coherent at this sampling density.

## Artifacts
- `artifacts/ea_protocol_transfer_2026-02-17.json`
- `artifacts/ea_landscape_default_2026-02-17.json`
- `artifacts/ea_landscape_near_cliff_80_2026-02-17.json`
- `artifacts/ea_landscape_summary_2026-02-17.json`
- `artifacts/ea_exploration_summary_2026-02-17.md`
