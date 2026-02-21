# Zotero Integration Handoff

This handoff captures the next implementation steps from the local Zotero reading pass, focused on turning literature/patent signals into concrete, testable repo upgrades.

## Current Status

- Zotero indexing is available via `scripts/zotero_index.py` and Make targets:
  - `make zotero-index`
  - `make zotero-query q="..."`
- Item-to-file resolution was validated against local Zotero DB.
- 11/12 requested items have local PDF attachments.
- Missing local attachment:
  - `liuTargetedDeliveryMitochondria2020` (`10.1111/jgh.15091`)

## Priority Implementation Plan

### 1) Add Evidence/Readiness Metadata to Literature Outputs

Files:
- `lit_spider.py`
- `artifacts/lit_spider_report.md` (output format)

Changes:
- Add `evidence_tier` field (`preclinical`, `pilot_clinical`, `clinical`, `review`, `editorial`, `patent`).
- Add `translational_readiness` score (0-5 rubric).
- Add `confidence_flag` (`high`, `medium`, `low`) based on source type + replication signal.
- Add `corrected_publication` boolean when correction notices exist (e.g., Adlimoghaddam 2024 correction).

Why:
- Separates mechanistic promise from clinical maturity and reduces overclaim risk.

### 2) Add Blood/EV Pre-Analytical QC Schema

Files:
- `schemas.py`
- `lit_spider.py` (if ingesting experimental metadata)
- `docs/schemas.md`

Changes:
- Add optional schema block for EV pre-analytics:
  - `hemolysis_index`
  - `platelet_contamination_flag`
  - `lipoprotein_confounder_flag`
  - `sample_processing_notes`
  - `protocol_reporting_completeness`
- Validate but do not hard-fail older artifacts (backward-compatible defaults).

Why:
- MIBlood-EV (Lucien et al.) indicates reproducibility depends strongly on pre-analytical controls.

### 3) Expand Simulator for Mitochondrial Delivery Modality

Files:
- `simulator.py`
- `multi_tissue_sim.py`
- `constants.py`
- `docs/simulator.md`

Changes:
- Introduce intervention parameters (default-neutral):
  - `delivery_target` (`systemic`, `brain`, `liver`, `ocular`, `immune`)
  - `delivery_efficiency` (0-1)
  - `payload_quality` (0-1)
  - `vesicle_carrier_mode` (`none`, `mitlet`, `pev`)
- Map target-specific effect multipliers (small, conservative priors).
- Add uncertainty knobs for manufacturing variability.

Why:
- Supports hypotheses from transfusion/transplantation and patent-specified routing concepts without overfitting to one claim.

### 4) Add Immune/Sepsis and Inflammation Stress Scenarios

Files:
- `disturbances.py`
- `grief_mito_scenarios.py`
- `kcramer_bridge.py`
- `tests/test_disturbance_suite.py`

Changes:
- Add scenario templates:
  - `sepsis_like_inflammation`
  - `viral_cytokine_burden`
  - `immune_senescence_aged`
- Add outputs:
  - inflammatory burden proxy
  - recovery latency
  - survival-proxy score

Why:
- Directly tests infectious/inflammatory claims in Benson preprint/patent context.

### 5) Add Complex-II/SDHB-Linked Proxy Metric

Files:
- `simulator.py`
- `analytics.py`
- `docs/analytics.md`

Changes:
- Add latent metric `complex_ii_support_index` (derived proxy, not direct wet-lab claim).
- Include in analytics output and perturbation scans.

Why:
- Aligns model outputs with Adlimoghaddam 2022 signal (SDHB upregulation) while staying simulation-native.

### 6) Add Local Literature Evidence Pack Generator

Files:
- New: `scripts/zotero_evidence_pack.py`
- New output: `artifacts/zotero_evidence_pack_YYYY-MM-DD.md`

Changes:
- Input: citation keys or DOI list.
- Resolve local attachment presence.
- Emit per-item summary fields:
  - source type
  - correction status
  - mechanistic claim tags
  - actionability tags for this repo
  - missing-attachment warnings

Why:
- Makes future handoffs reproducible and cheap to rerun under budget limits.

## Requested Items -> Actionability Map

1. `adlimoghaddamCorrectionMitochondrialTransfusion2024`
- Action: Track correction lineage and mark corrected figures/evidence in artifacts.

2. `albensiEditorialAreMitochondrial2023`
- Action: Treat as directional context only; low mechanistic weight in scoring.

3. `bensonFormulationsMethodsCellular2024`
- Action: Add CAR-cell energetics hypothesis scenario + delivery mode hooks.

4. `lucienMIBloodEVMinimalInformation2023`
- Action: Implement blood-EV quality metadata schema + reporting completeness.

5. `adlimoghaddamMitochondrialTransfusionImproves2022`
- Action: Add Complex-II-linked proxy metric and transfusion dose sweeps.

6. `bensonMitochondrialTransplantationIts2023`
- Action: Add aged-vs-young infection scenario comparisons with cytokine proxy output.

7. `bensonPlateletderivedExtracellularVessicles2023`
- Action: Add PEV/mitlet carrier mode assumptions in intervention encoding.

8. `bensonPlateletderivedMitochondriacontainingExtracellular2023`
- Action: Add ocular delivery target mode in transport multipliers.

9. `caicedoPoweringPrescriptionMitochondria2024`
- Action: Add ATMP/living-drug taxonomy tags + manufacturing uncertainty parameters.

10. `bensonSystemsMethodsGrowing2024`
- Action: Add manufacturing quality/yield variability knobs.

11. `liuTargetedDeliveryMitochondria2020`
- Action: Add liver-targeted uptake parameter; attachment missing locally, fetch before claim extraction.

12. `rinaldiFountainYouthMitochondrial2023`
- Action: Add hype-risk guardrails in generated recommendations (confidence + replication gates).

## Minimal Test Additions

Files:
- `tests/test_schemas.py`
- `tests/test_simulator.py`
- `tests/test_disturbance_suite.py`

Add tests for:
- Backward compatibility of new schema fields.
- Neutral defaults preserve previous baseline behavior.
- New disturbance scenarios produce bounded outputs.

## Suggested Execution Order

1. Schema and artifact metadata (`schemas.py`, `lit_spider.py`).
2. Simulator parameter hooks (`constants.py`, `simulator.py`, `multi_tissue_sim.py`).
3. Disturbance/scenario extensions.
4. Evidence pack script.
5. Tests + docs pass.

## Open Gaps

- Missing local full text for `10.1111/jgh.15091` in Zotero attachment store.
- Several patent PDFs are scan/image-only; OCR pipeline is needed for deep extraction if required.
