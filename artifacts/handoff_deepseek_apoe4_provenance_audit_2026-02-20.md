# Handoff: APOE4 Parameter Provenance Audit

**Date:** 2026-02-20
**Assigned to:** DeepSeek
**Priority:** High — model credibility depends on these numbers being traceable to literature
**Deliverable:** A completed version of the tables below with verification status, correct citations, and recommended adjustments

## Background

The mitochondrial aging simulator (`how-to-live-much-longer/`) models APOE4 genotype effects across six biological pathways. All numerical multipliers were introduced on 2026-02-19 as part of a precision medicine expansion. They are attributed to published papers in code comments, but **no derivation trail exists** showing how each specific number was extracted from its cited source. Some references lack full bibliographic information.

The core model is based on John G. Cramer's *How to Live Much Longer: The Mitochondrial DNA Connection* (2025, ISBN 979-8-9928220-0-4). APOE4 is **not discussed in the Cramer book** — it was added to extend the model toward personalized medicine.

All APOE4 constants live in `constants.py` lines 673–732. They are consumed by `genetics_module.py`, `lifestyle_module.py`, `downstream_chain.py`, and `parameter_resolver.py`.

## Task 1: Identify the Actual Papers

Six references are cited. Three lack full bibliographic information. For each, provide: full author list, title, journal, year, DOI.

| Cited As | Likely Identity | Full Citation Needed |
|----------|----------------|---------------------|
| O'Shea et al. 2024 | Unknown — described as "APOE4 × lifestyle interactions." Could be a review or meta-analysis. No journal or title given anywhere in the codebase. This is the **primary source** for all genotype multipliers. | YES |
| Ivanich et al. 2025 | Unknown — described as "sex-specific APOE4 effects, keto × female APOE4 microbiome." | YES |
| Anttila 2004 | Likely: Anttila T et al., "Alcohol drinking in middle age and dementia in older age: a prospective population study" or similar. | VERIFY |
| Downer et al. 2014 | Unknown — described as corroborating alcohol × APOE4 synergy. | YES |
| Nature Metabolism 2024 | Possibly Membrez et al. on trigonelline and NAD+ metabolism, or another article on coffee metabolites. No author given. | YES |
| Horner et al. 2020 | Unknown — described as "APOE4 male, ketogenic diet, cognitive improvement." | YES |

**If any of these papers do not exist or the citation is garbled, flag it.** The references may have been hallucinated or conflated by an AI assistant during the design phase.

## Task 2: Verify Each Numerical Value

For every APOE4-related constant, determine whether the value is:
- **A** = Directly reported in the cited paper (quote the relevant passage)
- **B** = Derivable from data in the cited paper (show the derivation)
- **C** = A reasonable estimate of a qualitative finding (the paper says "reduced" or "impaired" but gives no number)
- **D** = Not supported by the cited paper
- **E** = Supported by a different paper than the one cited (provide the correct reference)

### Core Genotype Multipliers (`constants.py:674–702`)

Attributed to: "O'Shea 2024"

| Constant | Het Value | Hom Value | Biological Claim | Verification Status | Notes |
|----------|-----------|-----------|-------------------|--------------------|----|
| `mitophagy_efficiency` | 0.65 | 0.45 | 35%/55% reduction in mitophagy | | |
| `inflammation` | 1.2 | 1.4 | 20%/40% increase in baseline inflammation | | |
| `vulnerability` | 1.3 | 1.6 | 30%/60% increase in genetic vulnerability | | |
| `grief_sensitivity` | 1.3 | 1.5 | 30%/50% amplification of grief-driven damage | | |
| `alcohol_sensitivity` | 1.3 | 1.5 | 30%/50% amplification of alcohol effects | | |
| `mef2_induction` | 1.2 | 1.3 | 20%/30% higher MEF2 induction requirement | | |
| `amyloid_clearance` | 0.7 | 0.5 | 30%/50% reduction in amyloid-beta clearance | | |

### Sex-Specific Interaction (`constants.py:705`)

Attributed to: "Ivanich et al. 2025"

| Constant | Value | Claim | Verification Status | Notes |
|----------|-------|-------|--------------------|----|
| `FEMALE_APOE4_INFLAMMATION_BOOST` | 1.1 | Additional 10% inflammation for female carriers (post-menopause) | | |

### Alcohol Interaction (`constants.py:712`)

Attributed to: "Anttila 2004, Downer 2014"

| Constant | Value | Claim | Verification Status | Notes |
|----------|-------|-------|--------------------|----|
| `ALCOHOL_APOE4_SYNERGY` | 1.3 | 30% amplification of alcohol damage for carriers | | |

### Coffee Interaction (`constants.py:719`)

Attributed to: "Nature Metabolism 2024"

| Constant | Value | Claim | Verification Status | Notes |
|----------|-------|-------|--------------------|----|
| `COFFEE_APOE4_BENEFIT_MULTIPLIER` | 1.2 | 20% additional coffee/trigonelline NAD+ benefit for carriers | | |

### Diet Interaction (`constants.py:732`)

No explicit citation given.

| Constant | Value | Claim | Verification Status | Notes |
|----------|-------|-------|--------------------|----|
| `KETO_FEMALE_APOE4_MULTIPLIER` | 1.3 | 30% additional keto benefit for female carriers | | Is this from Ivanich 2025 or Horner 2020? |

### Amyloid Clearance (`constants.py:786`)

No separate citation (mirrors het multiplier).

| Constant | Value | Claim | Verification Status | Notes |
|----------|-------|-------|--------------------|----|
| `AMYLOID_CLEARANCE_APOE4_FACTOR` | 0.7 | Reduced amyloid-beta clearance in carriers | | Does the APOE4-amyloid literature support a 30% reduction specifically? Key papers: Castellano 2011, Wildsmith 2013. |

## Task 3: Flag Qualitative vs. Quantitative Support

For each constant rated **C** (reasonable estimate of qualitative finding), answer:

1. What does the literature actually say? (e.g., "APOE4 carriers show impaired mitophagy" without a percentage)
2. What quantitative range would be defensible? (e.g., "20–50% reduction based on [specific study]")
3. Is the chosen value in the middle, at the edge, or outside that range?

## Task 4: Check for Missing APOE4 Effects

Are there well-established APOE4 effects in the literature that the model **omits**? For example:

- Blood-brain barrier integrity
- Lipid metabolism / cholesterol transport
- Synaptic plasticity effects beyond MEF2
- Cerebrovascular effects
- Interaction with tau phosphorylation (independent of amyloid)
- APOE4-specific response to exercise

Flag any omissions that would materially change the simulation's behavior for APOE4 carriers.

## Task 5: Dosage Relationship Between Het and Hom

The model assumes a consistent pattern where homozygous effects are roughly 1.5–2x the heterozygous effects (e.g., inflammation 1.2 → 1.4, vulnerability 1.3 → 1.6). Does the literature support this het-to-hom scaling, or is the relationship more complex (e.g., nonlinear, threshold-dependent, pathway-specific)?

## Output Format

Return a completed copy of this document with all tables filled in. For each constant, provide:
1. Verification status (A/B/C/D/E)
2. The supporting evidence (quote or derivation)
3. If status is C or D: a recommended replacement value with its source
4. If status is E: the correct citation

End with a summary section listing:
- Constants that are well-supported (A or B)
- Constants that are reasonable estimates (C) — acceptable but should be flagged in documentation
- Constants that need revision (D or E) — with recommended changes
- Missing effects that should be added
