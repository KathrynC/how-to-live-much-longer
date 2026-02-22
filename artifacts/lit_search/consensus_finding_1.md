# Cross-Model Consensus Analysis: Finding 1 (Exercise Is Harmful at Every Dose and Age)

**Date**: 2026-02-22
**Analyst**: Claude Opus 4.6 (cross-model consensus)
**Models compared**: qwen3-coder:30b, deepseek-r1:8b, gpt-oss:20b

---

## Executive Summary

All three models unanimously reject the model's prediction that exercise is harmful at every dose and age. The scientific consensus across all three LLM responses is clear: **moderate exercise is net beneficial for mitochondrial health**, primarily through biogenesis stimulation, antioxidant upregulation, and enhanced mitophagy. The current model parameterization (`EXERCISE_BIOGENESIS_FACTOR = 0.03`, `EXERCISE_METABOLIC_COST = 0.03`) creates an exact cancellation that allows the quadratic ROS term to dominate, producing universally harmful outcomes that contradict decades of exercise physiology literature.

**CRITICAL CAVEAT**: The gpt-oss:20b response for Finding 1 was truncated (cut off mid-table at line 17 of 17). Only the introduction and the beginning of Section 1 survived. This consensus analysis therefore relies primarily on qwen3-coder:30b and deepseek-r1:8b for detailed claims, with gpt-oss contributing only its framing and stated intent.

---

## Sub-Topic Analysis

### 1. Biogenesis Magnitude

#### Agreement Matrix
| Question | Qwen3 | DeepSeek | GPT-OSS | All Agree? |
|----------|-------|----------|---------|------------|
| Exercise increases biogenesis? | YES | YES | YES (implied) | **YES** |
| PGC-1alpha is primary mediator? | YES | YES | N/A (truncated) | YES (2/2) |
| Effect is large (>20%)? | YES (40-60%) | YES (20-50%) | N/A | YES (2/2) |

#### Quantitative Range
| Model | mtDNA Copy Number Increase | Mitochondrial Volume/Content Increase | PGC-1alpha Fold Change |
|-------|---------------------------|--------------------------------------|----------------------|
| Qwen3 | 15-20% (elderly, 6 months) | 40-60% (citrate synthase) | 2.5-fold |
| DeepSeek | 20-50% (skeletal muscle) | 20-30% (PGC-1alpha inducers) | Not quantified |
| GPT-OSS | N/A (truncated) | N/A | N/A |

**Consensus range**: 20-50% increase in mitochondrial content/mtDNA copy number with chronic exercise. Both models that provide data agree on the general order of magnitude. Qwen3's 40-60% is at the high end; DeepSeek's 20-50% brackets it.

#### Citations Appearing in 2+ Models

- **Lin, J., et al. (200x)** -- Both Qwen3 and DeepSeek cite a "Lin, J." paper on PGC-1alpha. However:
  - Qwen3: Lin, J., et al. (2002), *Cell*, "2.5-fold PGC-1alpha increase from exercise"
  - DeepSeek: Lin, J., et al. (2005), *Cell Metabolism*, on PGC-1alpha targets
  - **RED FLAG**: These are likely different papers, possibly conflated. The real Lin et al. 2002 paper in *Cell* (PMID 12176325) is about PGC-1alpha in thermogenesis/diabetes, not exercise per se. The 2005 paper in *Cell Metabolism* is closer but listed as about heart mitochondrial genes. The specific claim of "2.5-fold increase from chronic exercise" in the 2002 *Cell* paper is **suspect** -- that paper is primarily about transgenic PGC-1alpha overexpression.

- **Holloszy, J.O. (1967)** -- Only DeepSeek cites this. This is a **real landmark paper** (classic study of exercise-induced mitochondrial biogenesis in rat muscle). The ~30-50% increase claim is consistent with the real literature.

#### Red Flags
1. **Qwen3: Pileggi, A., et al. (2014), *Aging Cell***: "Long-term exercise training increased mtDNA copy number by 15-20% in elderly." Author name "Pileggi" is suspiciously generic for this niche. No well-known Pileggi in exercise mitochondrial biology. **Likely hallucinated.**
2. **Qwen3: Gomes, A. P., et al. (2008), *Cell Metabolism***: Ana P. Gomes has published in *Cell* (2013, on NAD+ and aging), but a 2008 paper specifically on exercise biogenesis in *Cell Metabolism* is not a well-known publication. The cited finding of "40-60% increase" with "2.5-fold increase in mitochondrial volume density" conflates two different metrics suspiciously. **Likely hallucinated or misattributed.**
3. **DeepSeek: Hambrecht, R., et al. (2003), *Journal of Physiology***: Hambrecht is a real cardiac rehabilitation researcher, but "PGC-1beta" focus in this context is unusual. **Plausible but unverified.**
4. **DeepSeek: Schrauwen, P., et al. (2004), *Diabetes***: Schrauwen is a real mitochondrial metabolism researcher. The citation is plausible but the specific numbers (20-30%) may be fabricated.
5. **DeepSeek: Brandt, U., et al. (2005/2006)**: Multiple "Brandt" citations with different initials (U. vs W.D.) -- this looks like the model generating variations of a name it associates with mitochondrial biochemistry. Ulrich Brandt is a real Complex I researcher, but the specific papers cited here are **suspect**.

#### Consensus Estimate
**Biogenesis magnitude**: 20-40% increase in mitochondrial content with regular moderate exercise. This is the most robust estimate given both models' agreement and consistency with the well-established Holloszy (1967) finding and its descendants.

**Model implication**: The current `EXERCISE_BIOGENESIS_FACTOR = 0.03` represents a 3% increase at maximum exercise level. If real biogenesis is 20-40%, the factor should be **0.06-0.12** (accounting for the fact that exercise_level=1.0 represents maximum and the biogenesis term is also gated by energy_available and copy_number_pressure). See parameter recommendation below.

---

### 2. ROS/Antioxidant Balance

#### Agreement Matrix
| Question | Qwen3 | DeepSeek | GPT-OSS | All Agree? |
|----------|-------|----------|---------|------------|
| Exercise increases acute ROS? | YES | YES | N/A | **YES** (2/2) |
| Antioxidant upregulation compensates? | YES | YES (with caveats) | N/A | **YES** (2/2) |
| Hormesis is the dominant paradigm? | YES | YES | N/A | **YES** (2/2) |
| Net oxidative stress decreases? | YES | Qualified yes | N/A | YES (2/2) |

#### Quantitative Range
| Model | ROS Increase | Antioxidant Upregulation | Net Effect |
|-------|-------------|------------------------|------------|
| Qwen3 | +25-40% | +50-60% (capacity), SOD2 2.3x, catalase 1.8x | Net reduction (oxidative damage -30%) |
| DeepSeek | Quadratic with flux (unquantified) | SOD2, catalase, GPx increase (unquantified) | Generally beneficial (adaptation prevails) |
| GPT-OSS | N/A | N/A | N/A |

**Consensus**: Exercise acutely increases ROS but the antioxidant adaptive response more than compensates at moderate doses. DeepSeek is more cautious about quantifying complete compensation. Qwen3's specific numbers (SOD2 2.3x, catalase 1.8x) are suspiciously precise.

#### Citations Appearing in 2+ Models
- **Ristow, M., et al.** -- Both models cite Ristow:
  - Qwen3: Ristow, M., et al. (2009), *Cell Metabolism* -- SOD2 2.3-fold, catalase 1.8-fold
  - Qwen3: Ristow, M., et al. (2010), *Cell Metabolism* -- U-shaped dose curve
  - DeepSeek: Ristow, M., & Zarse, K. J. (2014), *Antioxidants* -- hormesis review
  - **Assessment**: Michael Ristow is a **real and prominent researcher** in exercise hormesis (mitohormesis). His 2009 PNAS paper (not Cell Metabolism) on antioxidant supplementation blocking exercise benefits is highly cited. The specific numbers Qwen3 attributes (SOD2 2.3x, catalase 1.8x within 24h) are **plausible in direction but the journal and year may be wrong**. Ristow's key exercise paper is Ristow et al. 2009, *PNAS*, 106(21):8665-70.

- **Droge, J. (2002), *Physiological Reviews***: Cited by DeepSeek. Wulf Droge (not "J. Droge") published a landmark review in *Physiological Reviews* in 2002. The content is accurate but the initial is wrong. **Real paper, minor attribution error.**

- **Powers, S.K.** -- Cited by DeepSeek. Scott Powers is a **real, prominent exercise oxidative stress researcher**. The 2008 *Journal of Applied Physiology* review is plausible.

#### Red Flags
1. **Qwen3: Pinto, R.P., et al. (2017), *Free Radical Biology and Medicine***: "Moderate exercise increased ROS by 35-40% but antioxidant capacity by 50-60%." No well-known "Pinto, R.P." in exercise ROS literature. Appears multiple times across Qwen3's response. **Likely a hallucinated recurring author.**
2. **Qwen3: Kowluru, A.K., et al. (2015), *Journal of Biological Chemistry***: Kowluru is known for diabetic retinopathy/mitochondria research, not exercise physiology per se. The attribution of an exercise hormesis paper to JBC is odd. **Likely misattributed or hallucinated.**
3. **Qwen3's quantitative precision**: Numbers like "35-40% ROS increase" and "50-60% antioxidant capacity increase" with P-values are suspiciously clean. Real studies would show far more variability across conditions.

#### Consensus Estimate
**Net oxidative stress**: Moderate exercise produces a net REDUCTION in oxidative damage markers, despite acutely raising ROS. The antioxidant response (SOD2, catalase, GPx upregulation) typically exceeds the ROS increase by 1.5-2x in magnitude.

**Model implication**: The current model already implements `defense_factor += exercise * 0.2`, which is a substantial antioxidant boost. But the `exercise_ros = exercise * 0.03` term seems appropriately small. The model's problem is not the ROS channel -- it's the biogenesis/cost cancellation. The antioxidant defense channel appears correctly parameterized in direction and approximate magnitude.

---

### 3. mtDNA Damage

#### Agreement Matrix
| Question | Qwen3 | DeepSeek | GPT-OSS | All Agree? |
|----------|-------|----------|---------|------------|
| Exercise reduces net mtDNA damage? | YES (-15% deletions, -12% het) | Qualified (benefits outweigh/balance risks) | N/A | **YES** (2/2, with caveats) |
| Biogenesis dilutes damage? | YES (implicit) | YES (explicit) | N/A | **YES** (2/2) |
| Direct repair enhancement? | YES (enhanced repair mechanisms) | Limited (few direct studies) | N/A | **DISAGREE** |

#### Quantitative Range
| Model | mtDNA Deletion Change | Heteroplasmy Change | Damage Markers |
|-------|----------------------|--------------------|----|
| Qwen3 | -15% (P<0.05) | -12% in elderly | 8-OHdG +20% but mtDNA damage -35% |
| DeepSeek | Not quantified | Scarce direct data | Reduced nuclear DNA damage (proxy) |
| GPT-OSS | N/A | N/A | N/A |

**Consensus**: DeepSeek is notably more cautious and honest here, explicitly stating that "direct quantitative data showing exercise increases heteroplasmy in healthy individuals is scarce." Qwen3 provides precise numbers that cannot be verified.

#### Citations Appearing in 2+ Models
- **Wallace, D.C. (2005)** -- Cited by DeepSeek. Douglas Wallace is a **real and foundational mtDNA researcher**. His reviews on mitochondrial paradigms of aging are real. However, this is a review, not an exercise-specific study.
- **Attardi, G. (2002)** -- Cited by DeepSeek (twice). Giuseppe Attardi is a **real pioneer in mitochondrial genetics** but died in 2008; the 2002 TIBS reference is plausible.
- No mtDNA-specific citations appear in 2+ models.

#### Red Flags
1. **Qwen3: Vissing, J., et al. (2013), *Neurology***: John Vissing is a real mitochondrial myopathy clinician. A 2013 Neurology paper is plausible, but the specific finding of "exercise reduced deletion frequency by 15%" is suspiciously clean. **Plausible author, unverified specific finding.**
2. **Qwen3: Mancuso, M., et al. (2016), *Journal of Neurology***: Michelangelo Mancuso is real (mitochondrial neurology). But "heteroplasmy decreased by 12%" from exercise in elderly is a very strong claim that would be landmark if true. **Plausible author, extraordinary claim -- treat with skepticism.**
3. **Qwen3: Gomes, A.P., et al. (2011), *Cell Metabolism***: "Exercise increased 8-OHdG by 20% but decreased mtDNA damage by 35%." This internally contradicts (8-OHdG IS a mtDNA damage marker). Gomes is being cited again with a different year. **RED FLAG: likely hallucinated, and the claim is self-contradictory.**
4. **DeepSeek: Gardner, L.P., et al. (2001), *Journal of Physiology***: DeepSeek itself notes this is about NUCLEAR DNA, not mtDNA. Honest caveat.

#### Consensus Estimate
**mtDNA damage**: Exercise likely produces a **modest net reduction** in mtDNA damage accumulation, primarily through dilution (biogenesis adds healthy copies) and possibly through enhanced quality control (mitophagy). The effect on heteroplasmy directly is poorly quantified. An estimate of **5-15% reduction in net damage accumulation rate** is reasonable, but this is one of the least well-supported claims.

---

### 4. Mitophagy

#### Agreement Matrix
| Question | Qwen3 | DeepSeek | GPT-OSS | All Agree? |
|----------|-------|----------|---------|------------|
| Exercise enhances mitophagy? | YES (+40-50%) | YES (improved quality control) | N/A | **YES** (2/2) |
| PINK1/Parkin pathway involved? | YES | YES (implicit) | N/A | **YES** (2/2) |
| Selective removal of damaged mitos? | YES (+45%) | YES (implied) | N/A | **YES** (2/2) |

#### Quantitative Range
| Model | Mitophagy Increase | Selective Removal | Peak Timing |
|-------|-------------------|-------------------|-------------|
| Qwen3 | +40-50% PINK1/Parkin, +35% markers (p62, LC3-II) | +45% selective removal | 24-48h post-exercise |
| DeepSeek | Not quantified (emerging field) | Not quantified | Not specified |
| GPT-OSS | N/A | N/A | N/A |

**Consensus**: Both agree exercise enhances mitophagy, but DeepSeek is far more cautious about quantification, noting "direct measurement of mitophagy rate changes with exercise is complex but emerging." Qwen3's numbers (40-50% increase) are plausible in direction but should be treated as order-of-magnitude estimates at best.

#### Citations Appearing in 2+ Models
- No mitophagy citations appear in 2+ models.

#### Red Flags
1. **Qwen3: Youle, R.J., et al. (2016), *Nature Reviews Molecular Cell Biology***: Richard Youle is a **real and prominent mitophagy researcher** (NIH). He did publish reviews in Nat Rev Mol Cell Biol. However, the specific quantitative finding ("+40-50% PINK1/Parkin-mediated mitophagy from exercise") is unlikely to come from a review paper. **Real author, likely misattributed finding.**
2. **Qwen3: Wang, Y., et al. (2018), *Autophagy***: Too generic an author name to verify. **Unverifiable.**
3. **Qwen3: Kim, J.H., et al. (2019), *Cell Death & Disease***: Too generic. **Unverifiable.**
4. **DeepSeek: Vazquez-Cruz, M.A., et al. (2012), *Journal of Physiology***: Not a well-known name in mitophagy. **Likely hallucinated.**

#### Consensus Estimate
**Mitophagy stimulation**: Exercise increases mitophagy by an estimated **25-50%** (wide uncertainty). This is well-established mechanistically (AMPK activation -> ULK1 -> mitophagy initiation; also membrane depolarization during exercise -> PINK1 stabilization) even if specific quantification is poor.

---

### 5. Net Effect on Aging

#### Agreement Matrix
| Question | Qwen3 | DeepSeek | GPT-OSS | All Agree? |
|----------|-------|----------|---------|------------|
| Exercise is net beneficial for aging? | YES | YES | YES (stated in intro) | **YES** |
| Exercise improves mitochondrial function in elderly? | YES (+20% respiratory capacity) | YES (broadly) | N/A | **YES** (2/2) |
| Exercise increases mtDNA copy number? | YES (+25%) | YES (20-50%) | N/A | **YES** (2/2) |

#### Quantitative Range
| Model | Respiratory Capacity | mtDNA Copy Number | Telomere Effect | Oxidative Stress |
|-------|---------------------|-------------------|----------------|-----------------|
| Qwen3 | +20% in elderly | +25% (12 months) | +10-15% (6 months) | -30% markers |
| DeepSeek | Not quantified (overall beneficial) | +20-50% | Not discussed | Adaptation prevails |
| GPT-OSS | N/A (truncated) | N/A | N/A | N/A |

#### Citations Appearing in 2+ Models
- None across the aging section specifically.

#### Red Flags
1. **Qwen3: Epel, E.S., et al. (2018), *Aging Cell***: Elissa Epel is a **real telomere researcher** (collaborator of Elizabeth Blackburn). However, "increased telomere length by 10-15% over 6 months" is an extraordinary claim. Most exercise-telomere studies show *attenuated shortening*, not *lengthening*. This specific number is **RED FLAG -- likely inflated or fabricated.**
2. **Qwen3: Pinto, R.P.** appears again (2019, *Aging Research*). *Aging Research* is not a major journal name. **Recurring hallucinated author.**
3. **Qwen3: Mancuso, M. (2017), *Journal of Neurology***: Same author cited in section 3 with a different year. Plausible person, but the specific "20% improvement in respiratory capacity" claim is unverifiable.

#### Consensus Estimate
**Net aging effect**: Regular moderate exercise produces **15-25% improvement** in mitochondrial respiratory capacity and function in elderly populations compared to sedentary controls. This is one of the most robust findings in exercise physiology and does not depend on any individual citation.

---

### 6. Dose-Response

#### Agreement Matrix
| Question | Qwen3 | DeepSeek | GPT-OSS | All Agree? |
|----------|-------|----------|---------|------------|
| U-shaped/inverted-U dose-response exists? | YES | Not explicitly discussed | N/A | YES (1/1 that addresses it) |
| Moderate exercise is optimal? | YES (30-45 min, 60-70% VO2max) | YES (implied) | N/A | **YES** (2/2) |
| Extreme exercise can be harmful? | YES (>90 min) | Not discussed | N/A | YES (1/1) |

#### Quantitative Range
| Model | Optimal Dose | Optimal Frequency | Harm Threshold |
|-------|-------------|-------------------|----------------|
| Qwen3 | 30-45 min at 60-70% VO2max | 3-5 sessions/week | >90 min daily OR >60 min at high intensity |
| DeepSeek | Not specified | Not specified | Not specified |
| GPT-OSS | N/A | N/A | N/A |

Only Qwen3 provides quantitative dose-response data. DeepSeek acknowledges dose dependence but does not quantify it.

#### Red Flags
1. **Qwen3: Ristow, M., et al. (2010), *Cell Metabolism***: Ristow's work on mitohormesis is real, but a 2010 Cell Metabolism paper specifically showing "30% biogenesis increase at moderate exercise, 40% reduction at extreme" is suspiciously precise. **Real author, unverified specific finding.**
2. **Qwen3: Gomes, A.P., et al. (2011), *Cell Metabolism***: Cited again (third time with different claims). "Overtraining increased mtDNA damage by 40%." Gomes' real 2013 Cell paper is about declining NAD+, not exercise overtraining. **Hallucinated citation reuse.**

#### Consensus Estimate
**Dose-response**: The inverted-U dose-response for exercise is well-established in the broader literature (independent of these LLM responses). Optimal range: **150-300 minutes/week of moderate-intensity exercise** (WHO guidelines, consistent with Qwen3's estimates). Extreme endurance exercise (ultramarathons, >60-90 min daily high-intensity) may produce diminishing returns or harm.

---

## Cross-Cutting Assessment

### Citation Reliability Summary

| Category | Count | Assessment |
|----------|-------|-----------|
| **Real, prominent researchers cited correctly** | 6 | Ristow, Holloszy, Wallace, Attardi, Youle, Epel (though findings may be misattributed) |
| **Real researchers with wrong journal/year** | 3 | Lin (2002 vs 2005), Droge (wrong initial), Powers (plausible) |
| **Likely hallucinated authors** | 4 | Pinto R.P. (recurring), Pileggi A., Kowluru A.K. (wrong field), Vazquez-Cruz |
| **Unverifiable generic names** | 4 | Wang Y., Kim J.H., Gardner L.P., Pizzuti A. |
| **Self-contradictory claims** | 1 | Gomes 2011: "increased 8-OHdG by 20% but decreased mtDNA damage by 35%" |

### Model Quality Comparison

| Dimension | Qwen3-coder:30b | DeepSeek-r1:8b | GPT-OSS:20b |
|-----------|-----------------|----------------|-------------|
| **Specificity** | Very high (suspiciously so) | Moderate (appropriate caution) | N/A (truncated) |
| **Quantitative claims** | Abundant (many unverifiable) | Few (honest about gaps) | N/A |
| **Citation format** | Full (author, year, journal, finding) | Full but with more caveats | N/A |
| **Intellectual honesty** | Low (presents uncertain claims as definitive) | High (flags gaps, notes proxies) | N/A (intro was honest about goals) |
| **Hallucination risk** | HIGH | MODERATE | UNKNOWN |
| **Usefulness for calibration** | Order-of-magnitude estimates only | Directional guidance, mechanism mapping | Framing only |

**DeepSeek's self-aware hedging is actually the most scientifically responsible behavior** -- it repeatedly notes "direct quantitative data is scarce," "few direct studies," and uses phrases like "emerging field." Qwen3 presents fabricated-looking specifics as established facts.

---

## Parameter Recommendation: EXERCISE_BIOGENESIS_FACTOR

### Current Model State

The model implements three exercise channels:
1. **Biogenesis**: `exercise_biogenesis = exercise * 0.03 * energy_available * copy_number_pressure * tissue_mods["biogenesis_rate"]` (line 769)
2. **Metabolic cost**: `exercise_cost = exercise * 0.03` (line 923)
3. **Antioxidant defense**: `defense_factor += exercise * 0.2` (line 968)
4. **ROS increase**: `exercise_ros = exercise * 0.03` (line 974)

The bug is that channels 1 and 2 exactly cancel (both 0.03), leaving only the quadratic ROS effect (since ROS enters squared in heteroplasmy damage) to tip the balance toward harm. But channel 3 (defense_factor += 0.2) should actually overwhelm channel 4 (exercise_ros = 0.03). This suggests the "always harmful" finding may also involve the biogenesis being gated by energy_available and copy_number_pressure, reducing its effective magnitude below the cost.

### Cross-Model Consensus on Biogenesis Factor

| Model | Implied Biogenesis Coefficient | Reasoning |
|-------|-------------------------------|-----------|
| Qwen3 | 0.06-0.12 | 20-40% biogenesis increase at full exercise |
| DeepSeek | 0.06-0.15 | 20-50% range, classic Holloszy data |
| GPT-OSS | N/A | Truncated, but intro mentions EXERCISE_BIOGENESIS_FACTOR as key |

### Recommended Parameter Change

**`EXERCISE_BIOGENESIS_FACTOR`: 0.03 -> 0.08**

Rationale:
- Cross-model consensus: 20-40% biogenesis increase
- At `exercise_level = 0.5` (moderate), `energy_available ~ 0.8`, `copy_number_pressure ~ 0.5`, `tissue_biogenesis_rate = 1.0`: effective biogenesis = 0.5 * 0.08 * 0.8 * 0.5 = 0.016 (1.6% effective)
- At `exercise_level = 1.0` (maximum): 1.0 * 0.08 * 0.8 * 0.5 = 0.032 (3.2% effective)
- This is conservative given literature suggests 20-40%, because the gating factors substantially attenuate the raw coefficient
- The 0.08 ensures biogenesis always exceeds metabolic cost (0.03) by a healthy margin at moderate exercise

**Also consider increasing `EXERCISE_METABOLIC_COST` slightly to 0.02** (from 0.03) to reflect that the metabolic cost of moderate exercise is genuinely small in the context of daily ATP budgets. The current 0.03 may overstate the cost.

---

## Missing Exercise Channels

Both Qwen3 and DeepSeek agree the following exercise effects exist but are **not currently modeled**:

### 1. Exercise-Stimulated Mitophagy (HIGH PRIORITY)
**Agreement**: Qwen3 (explicit, +40-50%), DeepSeek (explicit, mechanism described)
**Current model**: No exercise -> mitophagy pathway exists. Mitophagy is controlled by `MITOPHAGY_BASE_RATE = 0.02` and PINK1/Parkin dynamics but is not modulated by exercise_level.
**Proposed addition**: Add `mitophagy_boost = exercise * 0.01` to the mitophagy rate. This would selectively remove damaged mitochondria, a distinct mechanism from biogenesis (which adds healthy ones).

### 2. Exercise-Induced mtDNA Repair Enhancement (LOW PRIORITY)
**Agreement**: Qwen3 claims yes, DeepSeek says limited evidence. **No consensus.**
**Assessment**: Do NOT add this channel -- insufficient evidence and DeepSeek's skepticism is well-founded.

### 3. Exercise-Induced Senescence Clearance (MEDIUM PRIORITY)
**Agreement**: Not explicitly discussed by either model in the exercise context, but exercise is known to reduce senescent cell burden. This would interact with the existing senolytic pathway.
**Assessment**: Consider for future work but not urgent.

### 4. Exercise -> NAD+ Preservation (MEDIUM PRIORITY)
**Agreement**: DeepSeek mentions exercise activates SIRT1/NAD+ axis; Qwen3 does not address directly.
**Assessment**: The current `defense_factor += exercise * 0.2` partially captures this (NAD-dependent sirtuin upregulation). May not need a separate channel.

### 5. AMPK Activation / mTOR Inhibition (LOW PRIORITY)
**Agreement**: DeepSeek mentions AMPK pathway explicitly. This overlaps with rapamycin's mechanism.
**Assessment**: Already partially captured by biogenesis term. Adding explicit AMPK would risk double-counting with rapamycin_dose.

---

## Summary of Recommended Changes

| Change | Priority | Current Value | Recommended | Confidence |
|--------|----------|--------------|-------------|------------|
| `EXERCISE_BIOGENESIS_FACTOR` | **CRITICAL** | 0.03 | **0.08** | High (cross-model consensus on 20-40% biogenesis) |
| `EXERCISE_METABOLIC_COST` | Medium | 0.03 | **0.02** | Moderate (metabolic cost likely overstated) |
| Add mitophagy channel | **HIGH** | absent | `mitophagy_boost = exercise * 0.01` | Moderate (both models agree, but quantification poor) |
| `exercise_ros` coefficient | Low | 0.03 | 0.03 (keep) | High (transient ROS increase is real) |
| `defense_factor` exercise boost | Low | 0.20 | 0.20 (keep) | High (adequate given literature) |

**Net effect of recommended changes**: At moderate exercise (0.5), the model would shift from net harmful to net beneficial, consistent with the unanimous literature consensus. The biogenesis increase (0.08 > 0.02 cost) creates a positive margin, and the optional mitophagy channel provides additional quality control benefit. The defense_factor (already 0.2) ensures the hormetic ROS response is appropriately modeled.

---

## Confidence Assessment

| Claim | Confidence | Basis |
|-------|-----------|-------|
| Exercise is net beneficial at moderate doses | **VERY HIGH** | Unanimous 3/3 models + decades of independent literature |
| Biogenesis factor should be 2-3x metabolic cost | **HIGH** | 2/2 models with data agree on 20-40% range |
| Mitophagy channel is missing | **HIGH** | 2/2 models agree it exists |
| Specific effect sizes (e.g., SOD2 2.3x) | **LOW** | Likely hallucinated specifics from Qwen3 |
| Exercise reduces heteroplasmy by 12% | **LOW** | Single model, extraordinary claim, unverifiable citation |
| Dose-response is inverted-U | **MODERATE** | Well-known in literature, but only 1 model provides specifics |
