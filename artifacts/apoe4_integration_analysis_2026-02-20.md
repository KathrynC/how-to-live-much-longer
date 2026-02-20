# APOE4 Integration Analysis: DeepSeek Audit vs. Cramer Constraints

**Date:** 2026-02-20
**Context:** DeepSeek completed a provenance audit of all APOE4 parameters. This document analyzes each recommendation against John Cramer's book and correspondence, then categorizes proposed changes.

## Governing Constraint: The Cramer Core ODE Is Untouched

John Cramer's *How to Live Much Longer* (2025) does not discuss APOE4. The book's scope is mitochondrial DNA damage and cellular energy crisis. All APOE4 parameters were introduced in the precision medicine expansion (2026-02-19), which operates in two layers **outside** the core:

```
Upstream (ParameterResolver) → [Cramer Core ODE - 8 states, UNTOUCHED] → Downstream (DownstreamChain - 6 ODEs)
```

**Any APOE4 change that modifies `simulator.py` is forbidden.** All changes must stay in the upstream modifiers (`genetics_module.py`, `lifestyle_module.py`, `parameter_resolver.py`, `constants.py`) or the downstream chain (`downstream_chain.py`).

Cramer corrections C7–C11 do not involve APOE4. The only tangential overlap is C7 (CD38 degrades NMN/NR), which shares the `genetics_module.py` pathway — but the `cd38_risk` genotype is independent of `apoe_genotype`.

---

## Summary of DeepSeek's Verification Verdicts

| Constant | Status | Meaning |
|----------|--------|---------|
| `ALCOHOL_APOE4_SYNERGY` (1.3) | **A/B** | Well-supported. Conservative estimate from Anttila 2004 hazard ratios (actual data suggests 2.3–3.6x for carriers). |
| `amyloid_clearance` (0.7/0.5) | **C** | Qualitatively established (Castellano et al. 2011). The 30%/50% figures are reasonable estimates. |
| `AMYLOID_CLEARANCE_APOE4_FACTOR` (0.7) | **C** | Same as above. Mirrors het multiplier. |
| `FEMALE_APOE4_INFLAMMATION_BOOST` (1.1) | **C** | Qualitative direction supported by Ivanich 2025. 10% is a conservative placeholder. |
| `KETO_FEMALE_APOE4_MULTIPLIER` (1.3) | **C** | Synthesized from Ivanich 2025 + Horner 2020. Reasonable placeholder. |
| `mitophagy_efficiency` (0.65/0.45) | **C/D** | Not from O'Shea. General literature supports the direction. Defensible range 20–50%. |
| `inflammation` (1.2/1.4) | **C/D** | Direction supported. Specific values are qualitative estimates. |
| `vulnerability` (1.3/1.6) | **C/D** | Model-specific composite concept. Reasonable as aggregation. |
| `grief_sensitivity` (1.3/1.5) | **C/D** | Highly speculative. Model-specific. Indirect literature link. |
| `alcohol_sensitivity` (1.3/1.5) | **C/D** | Should be derived more explicitly from Anttila's hazard ratios. |
| `mef2_induction` (1.2/1.3) | **D** | Unsupported. No literature links APOE4 to MEF2 induction rate. |
| `COFFEE_APOE4_BENEFIT_MULTIPLIER` (1.2) | **D** | Fabricated attribution. Membrez et al. 2024 contains zero APOE4 data. |

---

## Tier 1: Citation and Documentation Fixes (No Code Logic Changes)

These are unambiguously correct and should be done immediately.

### 1a. Fix the O'Shea 2024 attribution

**Current:** `constants.py:673` says `# ── Genetic multipliers (APOE4 literature, O'Shea 2024)` — implying O'Shea is the source of the numerical values.

**Reality:** O'Shea, D. M. et al. (2024), "APOE e4 carrier status moderates the effect of lifestyle factors on cognitive reserve," *Alzheimer's & Dementia*, 20(11), 8062-8073. This paper provides evidence for APOE4 × lifestyle *interaction effects* but does NOT report specific multiplier values for mitophagy, inflammation, etc.

**Fix:** Change the comment to accurately describe provenance:
```python
# ── Genetic multipliers (qualitative estimates; see provenance notes below) ──
```
Add a provenance block after the dict:
```python
# Provenance notes:
#   Direction of effects informed by general APOE4 literature.
#   O'Shea et al. 2024 (Alzh. Dement. 20:8062) confirms APOE4 × lifestyle
#   interaction but does not provide these specific numerical values.
#   mitophagy_efficiency: qualitative estimate (C), range 20-50% plausible
#   inflammation: qualitative estimate (C), consistent with Friday 2025
#   vulnerability: model-specific composite (C)
#   grief_sensitivity: model-specific estimate (C), indirect literature link
#   alcohol_sensitivity: conservative (A/B), cf. Anttila 2004 HR 2.3-3.6x
#   amyloid_clearance: qualitative estimate (C), cf. Castellano et al. 2011
#     (Sci Transl Med 3:89ra57) — isoform-specific clearance demonstrated
#   mef2_induction: UNSUPPORTED (D) — flagged for revision
```

**Cramer audit:** No conflict. This is documentation only.

### 1b. Fix amyloid clearance attribution

**Current:** `AMYLOID_CLEARANCE_APOE4_FACTOR` at line 786 has no separate citation.

**Fix:** Add comment:
```python
AMYLOID_CLEARANCE_APOE4_FACTOR = 0.7  # Castellano et al. 2011 (Sci Transl Med 3:89ra57); qualitative estimate (C)
```

**Cramer audit:** No conflict. Documentation only.

### 1c. Add full citations to genetics_module.py docstring

**Current docstring references:**
```
APOE4: O'Shea et al. 2024
```

**Fix:**
```
APOE4 interaction effects: O'Shea et al. 2024 (Alzh. Dement. 20:8062)
APOE4 × alcohol: Anttila et al. 2004 (BMJ 329:539); Downer et al. 2014 (Alcohol Alcohol. 49:17)
APOE4 amyloid clearance: Castellano et al. 2011 (Sci Transl Med 3:89ra57)
APOE4 × sex: Ivanich et al. 2025 (J Neurochem PMID 40890565)
Note: Specific multiplier values are qualitative estimates unless marked otherwise.
```

**Cramer audit:** No conflict.

---

## Tier 2: Code Changes — Well-Justified

### 2a. Remove `COFFEE_APOE4_BENEFIT_MULTIPLIER` as genotype-specific

**DeepSeek verdict:** Status D. Membrez et al. 2024 (Nature Metabolism 6:433) is about trigonelline and NAD+ in aging/sarcopenia. It contains **no mention of APOE, APOE4, or Alzheimer's**. The genotype-specific coffee benefit is a fabrication.

**Proposed change:** Remove the APOE4-specific multiplier from the coffee pathway. The general trigonelline NAD+ effect (`COFFEE_TRIGONELLINE_NAD_EFFECT`) remains — it benefits everyone, not just carriers.

In `lifestyle_module.py`, the line:
```python
genotype_mult = COFFEE_APOE4_BENEFIT_MULTIPLIER if apoe_genotype > 0 else 1.0
```
should become:
```python
genotype_mult = 1.0  # No genotype-specific coffee benefit supported by literature
```

In `constants.py`, deprecate the constant:
```python
# DEPRECATED: No literature supports APOE4-specific coffee benefit.
# Membrez et al. 2024 (Nature Metabolism) contains no APOE4 data.
# Retained as 1.0 (neutral) for backwards compatibility.
COFFEE_APOE4_BENEFIT_MULTIPLIER = 1.0
```

**Cramer audit:** No conflict. Coffee effects are in the upstream lifestyle module. The core ODE is untouched. The general NAD+ benefit from coffee remains intact.

**Test impact:** Tests asserting APOE4 carriers get extra coffee benefit will need updating. The behavioral change is small (20% bonus removed from one pathway).

### 2b. Neutralize `mef2_induction` until supported

**DeepSeek verdict:** Status D. Neither O'Shea 2024 nor any common APOE literature mentions a "MEF2 induction requirement" in the context of APOE4. This appears to be a model-specific assumption without literature basis.

**However:** The MEF2 pathway itself (Barker et al. 2021, Science Translational Medicine) is well-cited at `constants.py:754`. The problem is specifically that APOE4's *effect on* MEF2 induction is unsupported, not that MEF2 itself is wrong.

**Proposed change:** Set `mef2_induction` to 1.0 (neutral) for both het and hom in `GENOTYPE_MULTIPLIERS`, with a comment explaining why:

```python
'mef2_induction': 1.0,  # Neutralized: no literature supports APOE4-specific MEF2 effect (DeepSeek audit 2026-02-20, status D). Restore if evidence found.
```

**Do NOT replace MEF2 with DeepSeek's proposed `synaptic_function` parameter yet.** Reason: DeepSeek proposes replacing the `mef2_induction` multiplier (which modulates the MEF2 ODE's induction rate) with a `synaptic_function` multiplier (which would modulate the synaptic strength ODE). These are architecturally different — `mef2_induction` affects `mef2_derivative()`, while a synaptic modifier would affect `synaptic_derivative()`. Swapping one for the other changes the downstream chain's ODE structure, which should be planned separately.

**Cramer audit:** No conflict. MEF2 is in the downstream chain, outside the Cramer core. Neutralizing the APOE4 multiplier to 1.0 makes the MEF2 ODE behave identically for carriers and non-carriers (which is the correct default when evidence is lacking).

### 2c. Derive `alcohol_sensitivity` more explicitly from Anttila data

**DeepSeek notes:** Anttila 2004 reports hazard ratios: APOE4 carriers who drank infrequently had 2.3x dementia risk, frequent drinkers 3.6x, compared to carriers who never drank. The model's 1.3x is described as a "simple conservative estimate."

**Assessment:** The 1.3x value is deliberately conservative because Anttila's hazard ratios measure *dementia risk* (a downstream clinical outcome), not the direct biological amplification of alcohol-induced inflammation and NAD+ damage that the model simulates. The model applies 1.3x to *intermediate biological variables*, not to clinical endpoints. A 1.3x biological amplification could easily compound into a 2–3x hazard ratio over 30 simulated years.

**Proposed change:** Keep the value at 1.3 but improve the comment:
```python
'alcohol_sensitivity': 1.3,  # Conservative biological amplification estimate.
                              # Anttila 2004 (BMJ 329:539): HR 2.3-3.6x for clinical dementia,
                              # but this models intermediate inflammation/NAD damage, not endpoint risk.
```

**Cramer audit:** No conflict. Alcohol effects are upstream. Value unchanged.

---

## Tier 3: Deferred — Requires Architectural Discussion

DeepSeek proposes four new parameter families. Each is evaluated for Cramer compliance and implementation complexity.

### 3a. `tau_pathology_sensitivity` (proposed het: 1.25, hom: 1.4)

**Literature support:** Strong. Nature 2017 (APOE4 mice show markedly more tau pathology), JAMA Neurology 2020 (tau-PET uptake independently of amyloid in human carriers). DeepSeek rates this B/C.

**Where it would go:** The existing `tau_derivative()` in `downstream_chain.py` currently has no APOE4 interaction — tau is seeded by amyloid and promoted by inflammation, but APOE4's *direct* effect on tau (independent of amyloid) is missing. This is a genuine gap.

**Implementation sketch:** Add a new constant `APOE4_TAU_SENSITIVITY` (het: 1.25, hom: 1.4) to `GENOTYPE_MULTIPLIERS`. In `tau_derivative()`, multiply the seeding and inflammation terms by this factor:

```python
def tau_derivative(tau, amyloid, inflammation, apoe_tau_mult=1.0):
    seeding = TAU_SEEDING_RATE * amyloid * TAU_SEEDING_FACTOR * apoe_tau_mult
    infl = inflammation * TAU_INFLAMMATION_FACTOR * apoe_tau_mult
    clearance = TAU_CLEARANCE_BASE * tau
    return seeding + infl - clearance
```

**Cramer audit:** Safe. This is entirely within the downstream chain. The core ODE is untouched. The tau ODE already exists; this adds a genotype multiplier to it.

**Recommendation:** Good candidate for near-term addition. Well-supported, small code change, no architectural disruption.

### 3b. `synaptic_function` (proposed het: 0.8, hom: 0.65)

**Literature support:** Dumanis et al. 2010 (J Neuroscience) — APOE4 mice have significantly fewer dendritic spines at all ages. DeepSeek rates this C.

**Where it would go:** DeepSeek proposes this as a *replacement* for `mef2_induction`. However, these affect different ODEs — MEF2 vs synaptic strength. A cleaner approach would be to add a synaptic modifier to `synaptic_derivative()` independently:

```python
def synaptic_derivative(ss, ha, engagement, apoe_synaptic_mult=1.0):
    plasticity = PLASTICITY_FACTOR_BASE + ha * (PLASTICITY_FACTOR_HA_MAX - PLASTICITY_FACTOR_BASE)
    growth = LEARNING_RATE_BASE * engagement * plasticity * apoe_synaptic_mult * (1.0 - ss / MAX_SYNAPTIC_STRENGTH)
    decay = SYNAPTIC_DECAY_RATE * (ss - 1.0)
    return growth - decay
```

**Cramer audit:** Safe. Downstream chain only.

**Recommendation:** Reasonable, but the numerical values are qualitative estimates (status C). Could be added alongside the `mef2_induction` neutralization — effectively shifting APOE4's downstream effect from MEF2 (unsupported) to synaptic strength (supported).

### 3c. `bbb_permeability` (proposed het: 1.15, hom: 1.25)

**Literature support:** Neurobiology of Disease 2025 (elevated cortical BBB permeability in cognitively normal APOE4 carriers), PNAS Nexus 2024 (reduced capillary blood volume). DeepSeek rates this B/C.

**Where it would go:** DeepSeek proposes feeding this into `vulnerability`. In the current architecture, `vulnerability` (`genetic_vulnerability` in the core patient dict) modulates ROS production and damage accumulation in the core ODE. Adding BBB permeability as an upstream modifier on vulnerability is architecturally clean but has a subtle issue: it would change the core ODE's behavior by altering its input value.

**Cramer audit:** Technically safe — the core ODE code is untouched, only its input changes. But this is the most impactful proposed change because `genetic_vulnerability` directly drives the core ROS/damage dynamics. The existing `vulnerability` multiplier (het: 1.3, hom: 1.6) already captures a composite of APOE4 vulnerability effects. Adding BBB on top would compound with the existing multiplier.

**Recommendation:** Defer. The existing `vulnerability` multiplier is already a composite estimate. Adding BBB permeability as a separate multiplier would either: (a) compound with the existing value, making carriers too vulnerable, or (b) require reducing the existing vulnerability multiplier to accommodate it. This rebalancing needs careful simulation testing.

### 3d. `lipid_dysregulation` / `lipid_transport_efficiency` (proposed het: 0.85, hom: 0.7)

**Literature support:** Cells 2022 (altered cholesterol turnover), Cell Reports 2022 (APOE4 astrocytes secrete toxic lipoproteins). DeepSeek rates this C.

**Where it would go:** DeepSeek proposes a multiplier on membrane repair and synaptic maintenance, interacting with dietary fat intake.

**Cramer audit:** The Cramer book's mitochondrial model does not include a lipid metabolism variable. This would be a genuinely new pathway that doesn't map onto existing state variables. While APOE's primary biological function is lipid transport, modeling this properly would require at minimum a new state variable (or a new modifier in the downstream chain), plus interactions with dietary parameters.

**Recommendation:** Defer. High biological importance but high implementation complexity. The model's focus is mitochondrial aging; lipid metabolism is a parallel pathway that would need its own ODE design.

---

## Changes NOT Recommended

### Do not increase `ALCOHOL_APOE4_SYNERGY` to match hazard ratios

DeepSeek notes that Anttila's hazard ratios (2.3–3.6x) are much larger than the model's 1.3x multiplier. However, the model applies this multiplier to *intermediate biological variables* (inflammation, NAD damage), not to clinical endpoints. A 1.3x amplification of biological damage, compounded over 30 simulated years through a nonlinear ODE system, can produce hazard ratios consistent with Anttila's observations. Increasing the biological multiplier to 2.3x would likely produce unrealistically extreme outcomes in simulation.

### Do not make het-to-hom scaling pathway-specific yet

DeepSeek notes that the literature shows non-linear and conditional het-to-hom effects (e.g., Cai et al. 2025: 3x faster hippocampal atrophy for hom vs het *in the presence of amyloid*). While biologically accurate, implementing conditional scaling (where hom effects depend on amyloid load) would require significant refactoring of the modifier architecture. The current uniform ~1.5–2x scaling is an acceptable first approximation.

---

## Proposed Implementation Order

| Priority | Change | Type | Files Modified | Risk |
|----------|--------|------|----------------|------|
| **1** | Fix citations and add provenance comments | Documentation | `constants.py`, `genetics_module.py` | None |
| **2** | Neutralize `COFFEE_APOE4_BENEFIT_MULTIPLIER` → 1.0 | Code | `constants.py`, `lifestyle_module.py`, tests | Low |
| **3** | Neutralize `mef2_induction` → 1.0 | Code | `constants.py`, tests | Low |
| **4** | Add `tau_pathology_sensitivity` to downstream | Code | `constants.py`, `downstream_chain.py`, `scenario_runner.py`, tests | Medium |
| **5** | Add `synaptic_function` modifier to downstream | Code | `constants.py`, `downstream_chain.py`, `scenario_runner.py`, tests | Medium |
| **Deferred** | BBB permeability | Architecture | Multiple | High — rebalancing needed |
| **Deferred** | Lipid dysregulation | Architecture | Multiple | High — new pathway |
| **Deferred** | Pathway-specific het/hom scaling | Architecture | Multiple | High — conditional modifiers |

---

## Summary

**Cramer compliance:** All proposed Tier 1 and Tier 2 changes are fully compatible with the Cramer core ODE. They modify only upstream documentation, upstream modifiers (lifestyle_module), or downstream chain constants — never `simulator.py`. Tier 3 additions (tau sensitivity, synaptic function) are also architecturally safe as downstream chain modifications.

**Key finding from DeepSeek:** Only 1 of 12 APOE4 constants is well-supported by its cited source (ALCOHOL_APOE4_SYNERGY, status A/B). Seven are reasonable qualitative estimates (status C). Two are unsupported (mef2_induction and COFFEE_APOE4_BENEFIT_MULTIPLIER, status D). Two are C/D (direction supported but values are estimates with wrong attribution).

**Bottom line:** The APOE4 parameter set is directionally sound but poorly attributed. The fix is primarily documentary (correcting citations, flagging estimate status) rather than requiring wholesale numerical revision. The two status-D constants should be neutralized. The proposed additions (tau sensitivity, synaptic function) are better-supported than several existing constants.
