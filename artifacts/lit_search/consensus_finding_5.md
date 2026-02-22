# Cross-Model Consensus: Finding 5 -- Mitochondrial Transplant Saturates at 10% Dose

**Date**: 2026-02-22
**Analyst**: Claude Opus 4.6 (cross-model consensus)
**Models compared**: qwen3-coder:30b, deepseek-r1:8b, gpt-oss:20b

---

## Methodology

Three local LLMs were independently queried on Finding 5 (mitochondrial transplant dose saturation). Each produced a structured literature review covering 8 sub-topics. This document synthesizes their responses into an agreement matrix, identifies shared and conflicting claims, flags fabrication risks, and produces consensus estimates for model calibration.

**Important caveat**: All three models are generating literature reviews from parametric memory, not from live database queries. Citation accuracy is suspect across all three -- see Red Flags sections. The value of this exercise is in identifying (a) claims where all models converge (likely reflecting real literature patterns) vs (b) claims where models diverge or fabricate (requiring manual PubMed verification).

**Note on gpt-oss output**: The gpt-oss:20b response was truncated at 83 lines, cutting off mid-table in section 7 (Long-Term Dynamics). Sections 7 and 8 are therefore based on partial gpt-oss data only.

---

## 1. Transplant Mechanisms

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Isolated mitochondria can be injected and taken up by cells | YES | YES | YES | **3/3** |
| Multiple delivery methods exist (injection, exosomes, vesicles) | YES | YES | YES | **3/3** |
| Platelet-derived vesicles are a valid delivery vehicle | YES | YES | YES | **3/3** |
| Intracardiac injection is the primary clinical route studied | YES | YES | YES | **3/3** |
| McCully group is the foundational cardiac transplant team | YES | YES | YES | **3/3** |

### Quantitative Range

- Qwen3: Up to 30% engraftment from direct injection; exosome delivery lower but sustained.
- DeepSeek: Does not quantify mechanism efficiency.
- GPT-OSS: 1x10^7 mitochondria per rat heart restored LVEF from 30% to 42%.

### Shared Citations

| Citation | Qwen3 | DeepSeek | GPT-OSS | Verified? |
|----------|:-----:|:--------:|:-------:|:---------:|
| McCully et al. | 2019 (Nature Medicine) | 2003 (no journal) | 2014 (JACC: Basic Transl Sci) | SUSPECT -- three different years/journals |
| Emani et al. | 2021 (JTCVS) | 2016 (no journal) | 2018 (JCI Insight) | SUSPECT -- three different years/journals |
| Cowan et al. | 2020 (Cell Metabolism) | 2018 (no journal) | 2016 (Front Cell Dev Biol) | SUSPECT -- three different years/journals |

### Red Flags

**CRITICAL**: All three models cite the same author groups (McCully, Emani, Cowan) but assign them different years and journals. This is a hallmark of LLM confabulation -- the models know these are relevant research groups but are fabricating specific publication details. The McCully group's foundational cardiac mitochondrial transplant work is real (verified: McCully lab at Boston Children's Hospital, with Emani as a collaborator), but the specific years, journals, and quantitative findings need manual verification against PubMed.

- Qwen3 cites a "Cowan et al. 2020" in Cell Metabolism comparing delivery methods including "platelet-derived mitlets" -- this is almost certainly fabricated. The term "mitlets" appears to originate from Cramer's book, not from Cowan's published work.
- DeepSeek cites Emani et al. 2016 as demonstrating platelet-derived mitlets in old dogs -- this conflates Cramer's concept with the Emani group's actual cardiac work (which focused on pediatric patients, not dogs).
- GPT-OSS cites "Cramer et al., 2024 -- Cell Reports" for platelet-derived mitlets. This is fabricated; Cramer's work is a 2026 Springer book, not a 2024 Cell Reports paper.

### Consensus Estimate

The transplant mechanism itself (isolated mitochondria taken up by recipient cells after injection) is well-supported across all three models and corresponds to real published work. Multiple delivery routes exist. The McCully/Emani group at Boston Children's Hospital is correctly identified as the pioneering clinical team. However, specific quantitative claims about delivery method efficiency are unreliable.

---

## 2. Dose-Response

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Dose-response saturation exists | YES | YES | YES | **3/3** |
| Higher doses yield diminishing returns | YES | YES | YES | **3/3** |
| A clear plateau is reached beyond which no further benefit | YES | TENTATIVE | YES | **2.5/3** |
| Saturation occurs at a specific dose threshold | YES (20M/kg) | UNCLEAR | YES (5x10^5 - 5x10^6 per tissue) | **2/3** |

### Quantitative Range

- Qwen3: Saturation at ~20 million mitochondria/kg body weight.
  - 10M/kg -> 30% improvement; 50M/kg -> 60% improvement; beyond 50M no further benefit.
  - Engraftment plateaus at 20M/kg.
- DeepSeek: Specific dose of 10^9 mitlets/kg in dogs (Emani). No explicit saturation dose.
  - Notes that comprehensive dose-response curves are "relatively scarce."
- GPT-OSS: Multiple dose-response curves cited:
  - Li et al. 2017: Plateau at 5x10^5 mitochondria/g tissue.
  - Wang et al. 2020: Plateau after 5x10^6 per injection.
  - Patel et al. 2023: No further benefit with 2x10^9 (vs 1x10^9 per patient).

### Red Flags

- Qwen3's claim that McCully (2019) showed "10M/kg -> 30% improvement, 50M/kg -> 60% improvement" is suspiciously precise for what would be a cardiac surgery study. This reads like fabricated numbers that conform to the model's expectation of what a dose-response curve should look like.
- GPT-OSS cites Li et al. 2017, Wang et al. 2020, Li et al. 2019, and Patel et al. 2023 -- none cited by the other two models. These are plausible-sounding citations that need PubMed verification. The specificity (exact dose ranges and percentage improvements) increases fabrication risk.
- DeepSeek is the most honest here, explicitly stating that "direct, comprehensive dose-response curves showing saturation are relatively scarce, especially in aging models."

### Consensus Estimate

**All three models agree that dose-response saturation exists.** This is biologically plausible: cells have finite uptake capacity, and intracellular mitochondrial populations are regulated. However, the specific saturation dose is unreliable across models.

**Is 10% dose -> 62% of max benefit realistic?** This is the key model question. The saturation curve shape depends on whether "dose" means the fraction of mitochondrial population replaced or an absolute number injected. If 10% refers to the transplant_rate parameter (0.10 on a 0-1 scale), and the model already includes competitive displacement and feedback amplification, then achieving 62% of maximum benefit from a submaximal dose is plausible -- it reflects a saturating (Michaelis-Menten-like) dose-response. **The saturation point at 10% may actually be too low** if the model's feedback loops are too strong, but the shape (early steep rise, then plateau) is consistent with biological dose-response relationships.

**Recommendation**: The 10% saturation finding is a model prediction, not a literature constraint. It should be reported as "the model predicts early saturation due to feedback amplification" rather than "literature supports saturation at 10%."

---

## 3. Engraftment Efficiency

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Only a fraction of transplanted mitochondria engraft | YES | YES | YES | **3/3** |
| Engraftment efficiency is low (<30%) | YES (10-30%) | UNKNOWN | YES (3-15%) | **2/3** |
| Recipient cell health affects engraftment | YES | YES (implied) | NO DATA | **2/3** |
| Engraftment quantification is technically difficult | NO MENTION | YES (highlighted) | YES (implied) | **2/3** |

### Quantitative Range

- Qwen3: 20-30% in healthy cells, 10-15% in aged/diseased cells.
- DeepSeek: "Unknown exact efficiency." Explicitly states no study quantified engraftment directly.
- GPT-OSS: 5% at 24h, 1% at 7 days (Liu et al. 2018); 15% at 6 months (Patel et al. 2023); 3% in human myocardium (Kim et al. 2022).

### Red Flags

- Qwen3's 20-30% engraftment is dramatically higher than GPT-OSS's 3-15%. These numbers cannot both be correct for the same type of measurement.
- DeepSeek is again the most honest, explicitly noting that no cited study directly quantified engraftment.
- GPT-OSS's claim of 15% persistence at 6 months (Patel et al. 2023) seems high given 5% at 24h and rapid decline to 1% at 7 days from another study.
- The "Kim et al., 2022 -- Lancet" citation (3% engraftment in human myocardium) would be a landmark clinical result if real; it needs urgent PubMed verification.

### Consensus Estimate

**Engraftment efficiency is likely in the low single digits to ~15%, with high variability.** The models converge on the qualitative conclusion that most transplanted mitochondria do not persist, but the survivors can still have outsized functional impact. GPT-OSS's range of 3-15% is probably the most realistic bracket.

**For model calibration**: The current TRANSPLANT_ADDITION_RATE=0.30 represents the rate at which transplanted healthy copies are added per year (not engraftment efficiency per se). If real engraftment is 5-15%, and the model's transplant_rate parameter ranges 0-1 as a dose dial, then the effective addition rate at maximum dose would be 0.30 healthy copies/year -- which could correspond to ~15-30% of the existing population per year. This seems **plausible but on the high side** given literature estimates.

---

## 4. Competitive Dynamics (Displacement)

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Healthy mitochondria can displace damaged ones | YES | YES (plausible) | YES | **3/3** |
| Selective mitophagy is a displacement mechanism | YES | YES | YES | **3/3** |
| Healthy mtDNA has a replication advantage | YES | YES | YES (heteroplasmy shift) | **3/3** |
| Direct quantitative displacement rates exist in literature | YES (specific) | NO (gap flagged) | YES (specific) | **2/3** |

### Quantitative Range

- Qwen3: Henderson et al. 2018: healthy mtDNA 2.5x replication rate advantage. Zhao et al. 2020: 40% damaged removed via mitophagy, 1.8x replication advantage.
- DeepSeek: No quantitative data. Explicitly calls out "direct quantitative measurements of displacement rates are largely absent."
- GPT-OSS: Heteroplasmy dropped from 70% to 50% mutant in 3 weeks. PINK1/Parkin increased 1.8-fold. Damaged mtDNA decreased 25% after injection.

### Red Flags

- **The 2.5x replication advantage (Qwen3) is biologically implausible for healthy mitochondria.** Cramer's book places the deletion replication advantage at 1.10-1.21x (deletions are smaller, replicate faster). Healthy (full-length) mitochondria should replicate SLOWER than deletions, not 2.5x faster. Qwen3 appears to have inverted the direction -- it is the DAMAGED (deleted) mitochondria that have the replication advantage, not the healthy ones. This is a critical conceptual error.
- GPT-OSS's "D. J. R. et al., 2019 -- Nature Genetics" is a suspicious citation format (initials only, no full author name).
- DeepSeek is once again the most calibrated, noting the quantitative gap honestly.
- The 1.8x PINK1/Parkin increase (GPT-OSS, Zhang et al. 2021) is plausible for mitophagy upregulation but does not directly quantify competitive displacement.

### Consensus Estimate

**The competitive displacement mechanism is conceptually supported (3/3 agree) but has almost no direct quantitative calibration data.** The mechanism operates through two channels: (1) selective mitophagy preferentially removes damaged mitochondria (supported), and (2) replication dynamics (complex -- damaged deletions actually replicate faster, but healthy mitochondria have functional advantages in quality-control contexts).

**For model calibration**: TRANSPLANT_DISPLACEMENT_RATE=0.12 represents competitive displacement of damaged copies by transplanted healthy ones. This is a modeling assumption, not a literature-derived parameter. The mechanism is plausible but the rate is unconstrained by empirical data.

**IMPORTANT CORRECTION**: The replication advantage in the model should favor DELETIONS (which it does -- the Cramer core ODE gives deletions 1.10x advantage). Transplant displacement operates through mitophagy enhancement, not through healthy mitos replicating faster than deletions.

---

## 5. Feedback Loops (ATP -> Mitophagy -> Further Improvement)

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Positive feedback loop exists (ATP -> mitophagy -> less damage -> more ATP) | YES | YES | YES | **3/3** |
| Transplant-induced ATP increase enhances mitophagy | YES | YES (plausible) | YES | **3/3** |
| Direct quantitative evidence for the feedback loop post-transplant | YES (30% mitophagy increase) | NO (lacking) | YES (1.8-2.5 fold) | **2/3** |
| Feedback creates self-reinforcing improvement cycle | YES (explicit) | YES (theoretical) | YES (explicit) | **3/3** |

### Quantitative Range

- Qwen3: 30% mitophagy enhancement from transplant-induced ATP improvement.
- DeepSeek: "Direct quantitative measurements... are scarce." Notes AMPK and sirtuin pathways as plausible mediators.
- GPT-OSS: ATP 2-fold increase; mitophagy markers 1.8-fold increase; PINK1/Parkin 2.5-fold increase; patients with >1.5-fold mitophagy increase had 12% greater LVEF improvement.

### Red Flags

- Qwen3 attributes the 30% mitophagy enhancement to McCully et al. 2019 -- this specific claim should be verified.
- GPT-OSS's quantitative claims are impressively specific (2-fold ATP, 1.8-fold mitophagy, 2.5-fold PINK1/Parkin) but cite Zhang et al. 2021, Wang et al. 2022, and Kim et al. 2022 -- none of which are cited by the other models. The specificity increases confabulation risk.
- DeepSeek correctly notes that the evidence linking transplant-induced ATP to improved mitophagy "specifically after transplant is lacking" -- the pathway is known from other interventions (exercise, caloric restriction) but not directly measured post-transplant.

### Consensus Estimate

**The positive feedback loop is biologically plausible and supported by indirect evidence (3/3 agree on the mechanism).** The pathway is: more healthy mitochondria -> more ATP -> better energy for quality control -> enhanced mitophagy (PINK1/Parkin) -> removal of damaged mitochondria -> even higher healthy fraction -> more ATP. This is well-established biology.

**However, direct quantitative evidence specifically from transplant studies is weak.** The feedback loop strength in the model (how much ATP improvement translates to mitophagy improvement) remains a calibration parameter without strong empirical anchoring.

**For model calibration**: The feedback loop is implemented in the ODE through ATP-dependent mitophagy terms and is central to the saturation behavior. The 62% benefit at 10% dose is largely DRIVEN by this feedback. If the feedback is weaker in reality, saturation would occur at a higher dose fraction.

---

## 6. Clinical Dose Ranges

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Clinical doses are in the millions-to-billions of mitochondria | YES | YES | YES | **3/3** |
| Mammalian cells contain ~1000-2000 mitochondria each | NO MENTION | YES | YES | **2/3** |
| Human clinical trials exist for mitochondrial transplant | YES (implied) | NO (animal only) | YES (Kim 2022, Patel 2023) | **2/3** |
| Optimal dose is 10-30 million mitochondria/kg | YES | NO (10^9 mitlets/kg) | NO (10^8-10^9 per patient) | **1/3** |

### Quantitative Range

- Qwen3: 10-30 million mitochondria/kg; optimal at 15-20 million/kg.
- DeepSeek: 10^9 mitlets/kg (from dog study); translates to ~7x10^10 mitlets for a 70kg human.
- GPT-OSS: 10^8-10^9 mitochondria per patient (single injection). For aged mice, 10^8 mitlets per injection.

### Red Flags

- The dose ranges differ by orders of magnitude between models. Qwen3 says 10-30 million/kg (= 0.7-2.1 x 10^9 total for 70kg human). GPT-OSS says 10^8-10^9 per patient (total). DeepSeek says 10^9/kg (= 7 x 10^10 total). These span a 100-fold range.
- GPT-OSS cites "Kim et al., 2022 -- Lancet" and "Patel et al., 2023 -- Circulation Research" as human clinical trials. If real, these would be landmark papers. The Lancet citation in particular is extraordinary and needs verification. As of early 2025, there were very few published human clinical trials of mitochondrial transplantation (primarily the Emani/McCully pediatric cardiac work).

### Consensus Estimate

**The clinical dose range is poorly constrained, spanning 10^8 to 10^11 total mitochondria per patient across models.** The only reliable anchor is that the McCully/Emani group has performed mitochondrial transplant in human cardiac patients (pediatric, ischemic heart disease), and animal studies use doses in the millions to billions.

**For model calibration**: The transplant_rate parameter (0-1) is an abstract dose dial, not a specific mitochondrial count. The saturation behavior in the model is driven by the ODE dynamics (feedback loops, displacement, headroom), not by absolute numbers. Therefore, clinical dose scaling is relevant for eventual clinical translation but does not directly constrain the current model parameters.

---

## 7. Long-Term Dynamics

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Transplanted mitochondria persist for months | YES (6-12 months) | UNKNOWN | YES (6 months) | **2/3** |
| Transplanted mitochondria replicate | YES (1.5x rate) | UNKNOWN | YES (10% of total mtDNA) | **2/3** |
| Long-term data is lacking | NO MENTION | YES (explicitly) | TRUNCATED | **1/3** |
| Repeated dosing may be needed | YES (some patients) | YES (implied) | TRUNCATED | **2/3** |

### Quantitative Range

- Qwen3: Persistence 6-12 months. Replication at 1.5x native rate. Benefit sustained 6 months.
- DeepSeek: "Largely unknown." Explicitly flags that persistence and replication are critical unknowns.
- GPT-OSS: 15% of transplanted mitochondria persisted at 6 months (Patel et al. 2023). Maintained 10% of total mtDNA via replication. (File truncated, incomplete data.)

### Red Flags

- Qwen3's claim that transplanted mitochondria replicate at 1.5x the rate of native mitochondria is biologically problematic. Why would transplanted (healthy, full-length) mitochondria replicate faster than endogenous healthy mitochondria? There is no known mechanism for this. This appears to be a confabulation.
- GPT-OSS's "15% persistence at 6 months" (Patel et al. 2023) is the most specific claim and would be an important data point if verified.
- DeepSeek is once again the most epistemically honest, explicitly flagging the critical unknowns.

### Consensus Estimate

**Long-term dynamics are poorly characterized (DeepSeek is correct).** Some persistence over months is plausible (transplanted mitochondria integrated into the network would participate in normal fission/fusion/replication), but the specific rates are unconstrained.

**For model calibration**: TRANSPLANT_HEADROOM=1.5 allows the total mitochondrial population to expand 50% above baseline with transplant. This is a modeling assumption that compensates for the model's lack of explicit persistence/clearance dynamics. It may be too generous if real persistence is only 15% at 6 months.

---

## 8. Platelet-Derived Mitlets (Cramer's Concept)

### Agreement Matrix

| Claim | Qwen3 | DeepSeek | GPT-OSS | Consensus |
|-------|:-----:|:--------:|:-------:|:---------:|
| Platelet-derived mitochondrial transfer is a real phenomenon | YES | YES | YES | **3/3** |
| Mitlets show higher engraftment than isolated mitochondria | YES (40% higher) | NO DATA | YES (implied by ATP data) | **1.5/3** |
| Emani group has published on platelet-derived vesicles | YES | YES | YES (truncated) | **3/3** |
| Cramer's specific concept is supported by literature | YES (directly) | YES (tentatively) | YES (fabricated citation) | **2/3** |
| No immune rejection with platelet-derived delivery | YES | NO DATA | NO DATA | **1/3** |

### Quantitative Range

- Qwen3: 40% higher engraftment for platelet-derived mitochondria vs isolated. 10-20 million mitlets/kg sufficient. Cites "Smith et al., 2023 -- Frontiers in Cell and Developmental Biology."
- DeepSeek: Emani et al. 2016 demonstrated platelet-derived vesicle transfer in vitro and in vivo. Functional mitochondria delivered.
- GPT-OSS: "Cramer et al., 2024 -- Cell Reports": 5x10^5 mitlets/cell raised ATP 40% (from 3.0 to 4.2 umol/cell). Also "Cramer et al., 2025 -- Aging Cell" for aged mice. (Note: Both citations are fabricated.)

### Red Flags

- **GPT-OSS fabricates two Cramer papers**: "Cramer et al., 2024 -- Cell Reports" and "Cramer et al., 2025 -- Aging Cell." John G. Cramer is a physicist, not a cell biologist. His mitochondrial aging work is a forthcoming 2026 Springer book, not a series of cell biology papers. This is a clear hallucination.
- Qwen3's "Smith et al., 2023 -- Frontiers in Cell and Developmental Biology" showing 40% higher engraftment for platelet-derived mitochondria is unverifiable and likely fabricated.
- DeepSeek is the most measured, correctly attributing the platelet vesicle work to the Emani group and characterizing it as supporting Cramer's concept without fabricating Cramer-authored papers.
- The term "mitlets" (platelet-derived mitochondrial vesicles) appears to be Cramer's neologism. Its appearance in "published literature" cited by the models is a telltale sign of confabulation -- the models are back-projecting the query's terminology into fabricated citations.

### Consensus Estimate

**Platelet-derived mitochondrial transfer is a real biological phenomenon (3/3 agree).** Platelets contain functional mitochondria and can release mitochondria-containing vesicles. The Emani/McCully group has published work on this. However, the specific term "mitlets" and Cramer's particular framing (platelet-derived mitlets as the optimal delivery vehicle for aging intervention) are from Cramer's forthcoming 2026 book, not from peer-reviewed cell biology literature.

**The claim that mitlets show dramatically higher engraftment than isolated mitochondria (Qwen3: 40% higher) is unsupported.** This may be a reasonable hypothesis (vesicle packaging could protect mitochondria during delivery and facilitate cellular uptake), but it lacks direct empirical calibration.

---

## KEY QUESTION ANSWERS

### Q1: Is dose saturation at 10% (62% of max benefit) realistic?

**Consensus: Plausible but probably too aggressive.**

All three models agree that dose-response saturation exists in mitochondrial transplantation. The shape of the curve (steep initial rise, then plateau) is biologically standard. However, achieving 62% of maximum benefit at only 10% of maximum dose implies very strong feedback amplification.

The 10% saturation is a MODEL prediction driven by:
1. Competitive displacement (TRANSPLANT_DISPLACEMENT=0.12)
2. Positive feedback (ATP -> mitophagy -> less damage)
3. Headroom expansion (TRANSPLANT_HEADROOM=1.5)

If any of these are weaker in reality, saturation would shift rightward (to a higher dose). DeepSeek explicitly notes that the feedback strength is uncalibrated. **Recommendation: Saturation at 20-30% dose fraction would be more conservative and still consistent with the literature's qualitative findings. Consider reducing TRANSPLANT_DISPLACEMENT or headroom to shift the saturation point.**

### Q2: What engraftment efficiency do the models agree on?

**Consensus: 3-15% at 24h to 6 months, with high variability.**

- Qwen3: 10-30% (likely overstated)
- DeepSeek: Unknown (honest)
- GPT-OSS: 3-15%

The GPT-OSS range of 3-15% is most plausible. Real engraftment is likely at the lower end (3-5%) for systemic delivery and higher (10-15%) for direct tissue injection.

### Q3: Is competitive displacement supported?

**Consensus: Mechanistically plausible (3/3), quantitatively unconstrained (0/3 have reliable data).**

All three models agree that selective mitophagy can preferentially remove damaged mitochondria, and that this is enhanced when ATP levels improve. However, NO model provides reliable quantitative displacement rates. TRANSPLANT_DISPLACEMENT=0.12 is a reasonable modeling assumption but has no direct empirical anchor.

**Critical note**: The displacement mechanism in the model should work through ENHANCED MITOPHAGY of damaged mitochondria (quality-control pathway), NOT through healthy mitochondria out-replicating damaged ones (deletions actually replicate faster due to their smaller size).

### Q4: Is the positive feedback loop supported?

**Consensus: YES -- the pathway is well-established biology (3/3 agree).**

The ATP -> AMPK/sirtuin -> mitophagy -> damage clearance pathway is well-characterized in the broader mitochondrial biology literature. However, the specific magnitude of this feedback AFTER TRANSPLANT is not directly measured. The loop exists; its strength in the transplant context is unknown.

### Q5: Do any models cite evidence for Cramer's platelet-derived mitlets specifically?

**Consensus: The biological basis exists, but "mitlets" as Cramer's specific concept is not in the peer-reviewed literature.**

- Platelet-derived mitochondrial release is real (supported by Emani group work).
- The term "mitlets" is Cramer's neologism from his forthcoming book.
- GPT-OSS fabricated two Cramer journal papers. Qwen3 fabricated a "Smith et al." study.
- DeepSeek correctly identified the Emani work as supporting the concept without fabricating Cramer-authored publications.

### Q6: Are the current model parameters in the right ballpark?

| Parameter | Current Value | Literature Consensus | Assessment |
|-----------|:------------:|:-------------------:|:----------:|
| TRANSPLANT_ADDITION_RATE | 0.30 | No direct calibration data | **HIGH SIDE** -- 0.30 copies/year at max dose is aggressive. Consider 0.15-0.25 if saturation is too early. |
| TRANSPLANT_DISPLACEMENT | 0.12 | No direct calibration data | **UNCONSTRAINED** -- the mechanism is plausible but the rate is a free parameter. 0.12 is as good a guess as any. |
| TRANSPLANT_HEADROOM | 1.5 | 15% persistence at 6 months (GPT-OSS, unverified) | **POSSIBLY HIGH** -- if only 15% of transplanted mitochondria persist, a 50% headroom expansion may overstate long-term benefit. Consider 1.2-1.3. |

---

## MODEL RELIABILITY RANKING

1. **deepseek-r1:8b** -- Most epistemically honest. Explicitly flagged data gaps, avoided fabricating quantitative claims, correctly noted that many critical parameters lack direct empirical support. Best for understanding what we DON'T know.

2. **gpt-oss:20b** -- Most structured and quantitative, but truncated output and likely fabricated several citations (Kim et al. 2022 Lancet, Patel et al. 2023 Circulation Research, Cramer et al. 2024/2025). Best for getting plausible quantitative ranges, worst for citation accuracy.

3. **qwen3-coder:30b** -- Middle ground. Provided organized literature review but fabricated specific findings (e.g., 2.5x replication advantage for healthy mitochondria -- biologically backwards). Invented "Smith et al. 2023" for the mitlets claim. Least reliable for quantitative specifics.

---

## CITATIONS REQUIRING MANUAL PUBMED VERIFICATION

High priority (would change calibration if real):

1. **Kim et al., 2022 -- Lancet**: 3% human myocardial engraftment, 1.5-fold mitophagy correlation with 12% LVEF improvement. (GPT-OSS)
2. **Patel et al., 2023 -- Circulation Research**: 15% persistence at 6 months, 10% total mtDNA via replication. (GPT-OSS)
3. **Liu et al., 2018 -- Nature Communications**: 5% engraftment at 24h, 1% at 7 days. (GPT-OSS)
4. **Zhang et al., 2021 -- Cell Metabolism**: ATP doubled, mitophagy markers 1.8-fold. (GPT-OSS)
5. **Henderson et al., 2018 -- Nature Communications**: 2.5x healthy mtDNA replication. (Qwen3 -- likely confabulated or inverted)

Medium priority (mechanism confirmation):

6. **Wang et al., 2020 -- Stem Cell Research & Therapy**: Dose-response plateau. (GPT-OSS)
7. **Li et al., 2017 -- J Mol Cell Cardiol**: Dose-response with LVEF. (GPT-OSS)
8. **Zhao et al., 2020 -- Cell Death & Disease**: 40% damaged mitochondria removed, 1.8x replication advantage. (Qwen3)

---

## SUMMARY TABLE

| Sub-topic | 3-Model Agreement | Quantitative Confidence | Key Gap |
|-----------|:-----------------:|:----------------------:|---------|
| Transplant mechanisms | STRONG (3/3) | LOW (conflicting details) | Delivery method efficiency comparison |
| Dose-response saturation | STRONG (3/3 existence) | LOW (doses vary 100x) | No aging-specific dose-response curve |
| Engraftment efficiency | MODERATE (2/3 quantify) | LOW-MODERATE (3-30% range) | No standardized measurement method |
| Competitive displacement | STRONG (3/3 plausible) | VERY LOW (no direct rates) | Critical gap for model calibration |
| Positive feedback | STRONG (3/3 mechanism) | LOW (post-transplant data scarce) | Feedback strength unknown |
| Clinical doses | MODERATE (orders of magnitude differ) | VERY LOW (10^8-10^11 range) | Few human trials |
| Long-term dynamics | WEAK (1/3 flags unknown) | VERY LOW | Persistence/replication unstudied |
| Platelet-derived mitlets | STRONG (3/3 biology exists) | VERY LOW (Cramer-specific) | "Mitlets" is Cramer's concept, not literature |

---

## ACTIONABLE RECOMMENDATIONS

1. **Run lit_spider.py** with specific PubMed queries for the high-priority citations above. If Kim et al. 2022 (Lancet) and Patel et al. 2023 (Circulation Research) are real, they would provide the best calibration anchors.

2. **Consider parameter sensitivity sweep** on TRANSPLANT_ADDITION_RATE (0.10-0.30) and TRANSPLANT_HEADROOM (1.1-1.5) to characterize how the saturation dose fraction shifts. If saturation at 20-30% dose is more biologically plausible than 10%, the current parameters may need reduction.

3. **Flag the feedback loop strength** as the most impactful unconstrained parameter. The 62% benefit at 10% dose is dominated by the positive feedback (ATP -> mitophagy -> clearance). A weaker feedback would shift saturation rightward and reduce the benefit amplitude.

4. **Separate Cramer's claims from literature claims** in any publication. The "mitlets" concept, specific dose recommendations, and transplant-as-primary-rejuvenation thesis are Cramer's arguments, not consensus literature positions.

5. **REVIEW WITH CRAMER**: Does the competitive displacement mechanism operate through enhanced mitophagy (as the biology suggests) or through direct replication competition? The model's TRANSPLANT_DISPLACEMENT term adds healthy copies AND removes damaged copies -- is this double-counting with the mitophagy enhancement from improved ATP?
