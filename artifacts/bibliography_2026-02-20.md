# Bibliography & Reference Audit

**Repository:** how-to-live-much-longer
**Date:** 2026-02-20
**Format:** APA 7th Edition
**Scope:** All literature citations in Python source files, markdown documentation, and artifact files

---

## Audit Summary

| Category | Count |
|----------|-------|
| References verified as accurate | 17 |
| References with minor discrepancies | 7 |
| References with significant errors | 3 |
| Publication status unverifiable | 1 |
| Lit-spider references (PubMed-indexed, not individually audited) | ~80 |
| **Total unique publications referenced** | **~108** |

### Critical Errors Requiring Code Correction

| # | Reference | Error | Severity |
|---|-----------|-------|----------|
| 1 | Dumanis et al. | Year cited as 2010; actual publication is **2009** | Medium |
| 2 | Ivanich et al. 2025 | Claimed finding ("sex differences in neuroinflammation") **completely mischaracterizes** paper (actually about ketogenic diet / gut microbiota) | **Critical** |
| 3 | Norwitz et al. 2021 | Hill-function dose-response parameters for 11 supplements **do not appear in any Norwitz publication**; likely phantom citation | **Critical** |
| 4 | Bender et al. 2006 | Conflates total deletion burden (~43% controls) with the specific common 4,977 bp deletion; the common deletion was found only 7 times | Significant |
| 5 | Anttila et al. 2004 | Values 2.3-3.6x are **odds ratios**, not hazard ratios as labeled in code | Medium |
| 6 | Cramer book | Forthcoming from Springer in 2026 (ISBN 978-3-032-17740-7); earlier docs had contradictory metadata, now standardized | Resolved |

---

## I. Primary Works

### The Cramer Book (Primary Theoretical Source)

Cramer, J. G. (forthcoming 2026). *How to live much longer: The mitochondrial DNA connection*. Springer. ISBN 978-3-032-17740-7.

> **Audit status: FORTHCOMING.** Listed on Springer at https://link.springer.com/book/9783032177407. ISBN prefix 978-3-032 is consistent with Springer Verlag. John G. Cramer is Professor Emeritus of Physics, University of Washington, with a documented interest in mitochondrial aging (AV-233, *Analog*, Nov-Dec 2024; Mitrix Bio trial, Feb 2026). The project author has direct access to the manuscript with chapter/page-level citations throughout the codebase (30+ specific references). Citation standardized across all files on 2026-02-20.

### Mitrix Bio Press Coverage

Marcus, A. D. (2026, February 20). Longevity treatments for a 91-year-old? A bold bet in Silicon Valley's immortality race. *The Information* (Weekend).

> Documents John G. Cramer's participation as one of two patient-investors in Mitrix Bio's mitochondrial transplant program. Cramer (age 91) received his third IV infusion of mitochondria donated by his 26-year-old granddaughter Selena Shea at a clinic in Dallas. Mitrix Bio (Pleasanton, CA; CEO Tom Benson; $4M funding) focuses on transplanting young mitochondria into older recipients — the cellular energy restoration that the how-to-live-much-longer simulator models. The article contrasts Mitrix's mitochondria-first approach ("if the battery is not working, nothing works" — investor Ronjon Nag) with the cellular reprogramming strategies pursued by Retro Biosciences ($180M, Sam Altman), NewLimit ($200M, Brian Armstrong), and Altos Labs ($3B, Jeff Bezos). Cramer's anti-aging research and his book are referenced. The second patient-investor is Clay Rawlings, 71, of Houston.

### Zimmerman Dissertation

Zimmerman, J. W. (2025). *Locality, relation, and meaning construction in language, as implemented in humans and large language models (LLMs)* [Doctoral dissertation, University of Vermont]. Graduate College Dissertations and Theses, 2082.

> **Audit status: VERIFIED.** Defense confirmed March 19, 2025, UVM SOCKS group. Author Julia Witte Zimmerman. Section-level claims (2.2.3, 3.5.2, 3.5.3, 4.6.4, 4.7.6) could not be independently verified from web sources but are plausible given confirmed scope. Kathryn Cramer is a co-author on the related 2024 *Plutonics* paper.

---

## II. APOE4 & Neurogenetics

Anttila, T., Helkala, E.-L., Viitanen, M., Kareholt, I., Fratiglioni, L., Winblad, B., Soininen, H., Tuomilehto, J., Nissinen, A., & Kivipelto, M. (2004). Alcohol drinking in middle age and subsequent risk of mild cognitive impairment and dementia in old age: A prospective population based study. *BMJ*, *329*(7465), 539. https://doi.org/10.1136/bmj.38181.418958.BE

> **Audit status: VERIFIED with discrepancy.** Paper exists, journal/volume/DOI correct. The 2.3-3.6x figures are **odds ratios (ORs)**, not hazard ratios as labeled in `constants.py`. OR 2.3 = infrequent-drinking APOE4 carriers; OR 3.6 = frequent-drinking APOE4 carriers (vs. non-carrier non-drinkers). Interaction term significant (P = 0.04). The conservative 1.3x multiplier used in code is a reasonable dampening for biological (vs. clinical) modeling.

Barker, S. J., Raju, R. M., Milman, N. E. P., Wang, J., Davila-Velderrain, J., Gunter-Rahman, F., Parro, C. C., Bozzelli, P. L., Abdurrob, F., Abdelaal, K., Bennett, D. A., Kellis, M., & Tsai, L.-H. (2021). MEF2 is a key regulator of cognitive potential and confers resilience to neurodegeneration. *Science Translational Medicine*, *13*(618), eabd7695. https://doi.org/10.1126/scitranslmed.abd7695

> **Audit status: VERIFIED with minor nuance.** Paper uses "overexcitability" and "cognitive resilience" rather than "excitotoxicity protection" as characterized in code. Directionally correct but slightly oversimplified. MEF2 simulation parameters (induction rate, decay rate, suppression factor) are modeling constructs, not values from this paper.

Castellano, J. M., Kim, J., Stewart, F. R., Jiang, H., DeMattos, R. B., Patterson, B. W., Fagan, A. M., Morris, J. C., Mawuenyega, K. G., Cruchaga, C., Goate, A. M., Bales, K. R., Paul, S. M., Bateman, R. J., & Holtzman, D. M. (2011). Human apoE isoforms differentially regulate brain amyloid-beta peptide clearance. *Science Translational Medicine*, *3*(89), 89ra57. https://doi.org/10.1126/scitranslmed.3002156

> **Audit status: VERIFIED.** All metadata correct. Finding accurately attributed: ApoE4 is the slowest isoform for amyloid-beta clearance, with no effect on production. The 0.7 clearance factor used in code is a reasonable modeling simplification.

Downer, B., Zanjani, F., & Fardo, D. W. (2014). The relationship between midlife and late life alcohol consumption, APOE e4 and the decline in learning and memory among older adults. *Alcohol and Alcoholism*, *49*(1), 17--22. https://doi.org/10.1093/alcalc/agt144

> **Audit status: VERIFIED.** All metadata correct. Gene-environment interaction accurately characterized: alcohol consumption associated with greater cognitive decline in APOE4 carriers but better outcomes in non-carriers.

Dumanis, S. B., Tesoriero, J. A., Babus, L. W., Nguyen, M. T., Trotter, J. H., Ladu, M. J., Weeber, E. J., Turner, R. S., & Rebeck, G. W. (2009). ApoE4 decreases spine density and dendritic complexity in cortical neurons in vivo. *Journal of Neuroscience*, *29*(48), 15317--15322. https://doi.org/10.1523/JNEUROSCI.3839-09.2009

> **Audit status: VERIFIED with year error.** Finding is accurate (spine density reductions of 27.7%, 24.4%, 55.6% at 4 weeks, 3 months, 1 year). **ERROR: Cited as "2010" in `constants.py` and `genetics_module.py`; actual publication date is December 2, 2009.** Correction needed.

Ivanich, K., Yackzan, A., Flemister, A., Chang, Y.-H., Xing, X., Chen, A., Yanckello, L. M., Sun, M., Aware, C., Govindarajan, M., Kramer, S., Ericsson, A., & Lin, A.-L. (2025). Ketogenic diet modulates gut microbiota-brain metabolite axis in a sex- and genotype-specific manner in APOE4 mice. *Journal of Neurochemistry*, *169*(9), e70216. https://doi.org/10.1111/jnc.70216

> **Audit status: VERIFIED but FINDING IS WRONG.** Paper exists at PMID 40890565, journal and year correct. **CRITICAL ERROR: Code claims this paper documents "sex differences in neuroinflammation" (`genetics_module.py`, `constants.py`). The paper is actually about ketogenic diet effects on gut microbiota and brain metabolites in APOE4 mice.** Neuroinflammation is not the focus. The finding attribution is a fundamental mischaracterization. Needs replacement with an actual sex-differences-in-neuroinflammation reference, or the claim should be removed.

O'Shea, D. M., Zhang, A. S., Rader, K., Shakour, R. L., Besser, L., & Galvin, J. E. (2024). APOE epsilon-4 carrier status moderates the effect of lifestyle factors on cognitive reserve. *Alzheimer's & Dementia*, *20*(11), 8062--8073. https://doi.org/10.1002/alz.14304

> **Audit status: VERIFIED.** All metadata correct. "APOE4 interaction effects with lifestyle" is a reasonable shorthand, though the paper specifically studies cognitive reserve as outcome and highlights mindfulness/social engagement rather than lifestyle broadly.

Shi, Y., Yamada, K., Liddelow, S. A., Smith, S. T., Zhao, L., Luo, W., Tsai, R. M., Spina, S., Grinberg, L. T., Rojas, J. C., Gallardo, G., Wang, K., Roh, J., Robinson, G., Finn, M. B., Jiang, H., Sullivan, P. M., Baufeld, C., Wood, M. W., ... Holtzman, D. M. (2017). ApoE4 markedly exacerbates tau-mediated neurodegeneration in a mouse model of tauopathy. *Nature*, *549*(7673), 523--527. https://doi.org/10.1038/nature24016

> **Audit status: VERIFIED.** All metadata correct. Finding accurately attributed: ApoE4 exacerbates tau pathology independently of amyloid-beta via a "toxic" gain of function in P301S tau transgenic mice.

Therriault, J., Benedet, A. L., Pascoal, T. A., Mathotaarachchi, S., Chamoun, M., Savard, M., Thomas, E., Kang, M. S., Lussier, F., Tissot, C., Parsons, N., Qureshy, A., Rosa-Neto, P., et al. (2020). Association of apolipoprotein E epsilon-4 with medial temporal tau independent of amyloid-beta. *JAMA Neurology*, *77*(4), 470--479. https://doi.org/10.1001/jamaneurol.2019.4421

> **Audit status: VERIFIED with unconfirmed detail.** Paper exists, all metadata correct. Core claim (APOE4 elevates tau-PET independently of amyloid) confirmed. The specific "~0.33 SD" effect size could not be verified from abstracts alone; may require full-text confirmation. Cross-sectional study reports associations, not hazard ratios.

---

## III. Mitochondrial Biology & NAD+ Metabolism

Bender, A., Krishnan, K. J., Morris, C. M., Taylor, G. A., Reeve, A. K., Perry, R. H., Jaros, E., Hersheson, J. S., Betts, J., Klopstock, T., Taylor, R. W., & Turnbull, D. M. (2006). High levels of mitochondrial DNA deletions in substantia nigra neurons in aging and Parkinson disease. *Nature Genetics*, *38*(5), 515--517. https://doi.org/10.1038/ng1769

> **Audit status: VERIFIED with significant overclaim.** Paper exists. **ERROR: Code states "common 4,977 bp deletion reaches >50% in aged substantia nigra neurons." Actual finding: total mtDNA deletion burden (multiple deletion types) averaged 43.3% in aged controls and 52.3% in PD patients. The common 4,977 bp deletion was found only 7 times total and was NOT the dominant form.** The high percentages reflect an aggregate of multiple clonally expanded deletions of various sizes (1,763-9,445 bp). Recommendation: Rephrase to "total mtDNA deletion burden" and note the >50% figure applies to PD, not normal aging.

Camacho-Pereira, J., Tarrago, M. G., Chini, C. C. S., Nin, V., Escande, C., Warner, G. M., Puranik, A. S., Schoon, R. A., Reid, J. M., Galina, A., & Chini, E. N. (2016). CD38 dictates age-related NAD decline and mitochondrial dysfunction through an SIRT3-dependent mechanism. *Cell Metabolism*, *23*(6), 1127--1139. https://doi.org/10.1016/j.cmet.2016.05.006

> **Audit status: VERIFIED.** All metadata correct. Finding accurately attributed: CD38 expression increases with aging, is required for age-related NAD decline, and is the main enzyme degrading NMN in vivo. Cited in code as "Ca16" via Cramer Ch. VI.A.3.

Membrez, M., et al. (2024). Trigonelline is an NAD+ precursor that improves muscle function during ageing and is reduced in human sarcopenia. *Nature Metabolism*, *6*, 433--447. https://doi.org/10.1038/s42255-024-00997-x

> **Audit status: VERIFIED.** All metadata correct including DOI. Code correctly notes "NO APOE4 data" -- confirmed, paper contains zero APOE mentions. The former `COFFEE_APOE4_BENEFIT_MULTIPLIER` was correctly identified as fabricated attribution and neutralized to 1.0.

Miwa, S., Kashyap, S., Chini, E., & von Zglinicki, T. (2022). Mitochondrial dysfunction in cell senescence and aging. *Journal of Clinical Investigation*, *132*(13), e158447. https://doi.org/10.1172/JCI158447

> **Audit status: VERIFIED but weak attribution.** Paper exists and is a review on mitochondrial dysfunction in senescence. The specific claim "half-life of mammalian mitochondria is ~2-4 weeks" may appear as a general statement in this review but is not a primary finding. Actual mitochondrial half-lives vary enormously by tissue (liver ~4 days, cardiac ~17 days, skeletal muscle mtDNA ~132-216 days). The "2-4 weeks" is a rough cardiac/brain approximation.

Norwitz, N. G., Saif, N., Ariza, I. E., & Isaacson, R. S. (2021). Precision nutrition for Alzheimer's prevention in ApoE4 carriers. *Nutrients*, *13*(4), 1362. https://doi.org/10.3390/nu13041362

> **Audit status: LIKELY PHANTOM CITATION.** This is the closest match to "Norwitz et al. 2021" found. **CRITICAL ERROR: Code attributes Hill-function dose-response parameters (MAX_EFFECT, HALF_MAX) for 11 nutraceuticals (NR, DHA, CoQ10, resveratrol, PQQ, ALA, vitamin D, B-complex, magnesium, zinc, selenium) to this paper. No Norwitz 2021 publication provides any such quantitative data.** The *Nutrients* paper discusses qualitative supplement recommendations for APOE4 carriers. Several supplements in code (PQQ, ALA, magnesium, zinc, selenium) do not appear in any known Norwitz publication. The Hill-function parameters are simulation constructs without a specific literature source. Likely hallucinated during AI-assisted code generation. **Recommendation: Relabel these parameters as "simulation estimates (no single literature source)" or replace with per-supplement primary literature.**

Rossignol, R., Faustin, B., Rocher, C., Malgat, M., Mazat, J.-P., & Letellier, T. (2003). Mitochondrial threshold effects. *Biochemical Journal*, *370*(3), 751--762. https://doi.org/10.1042/BJ20021594

> **Audit status: VERIFIED with minor simplification.** Paper exists, all metadata correct. The paper reviews phenotypic and biochemical threshold effects in mitochondrial disease, discussing mutation- and tissue-specific thresholds typically in the 60-90% range. The code's use of ~70% as a representative threshold is a defensible midpoint simplification. The subsequent recalibration to 0.50 (deletion-only heteroplasmy after C11 mutation split) is well-documented. **Recommendation: Characterize as "within the range reported by Rossignol et al. (2003)" rather than implying Rossignol specified 70% exactly.**

Vandiver, A. R., Hoang, A. N., Herbst, A., Lee, C. C., Aiken, J. M., McKenzie, D., Teitell, M. A., Timp, W., & Wanagat, J. (2023). Nanopore sequencing identifies a higher frequency and expanded spectrum of mitochondrial DNA deletion mutations in human aging. *Aging Cell*, *22*(6), e13842. https://doi.org/10.1111/acel.13842

> **Audit status: VERIFIED but specific values are Cramer-derived.** Paper exists, journal/volume correct. Studies mtDNA deletions across the lifespan via nanopore sequencing. The specific doubling time values (11.81 yr young, 3.06 yr old, transition at age 65, >=21% replication advantage) are attributed to "Cramer Appendix 2, p.155, Fig. 23 (Va23)" -- i.e., Cramer's own curve fits of Vandiver's data, not values stated in the Vandiver paper. This attribution chain is correctly documented in code comments.

---

## IV. Ecological Resilience Framework

Holling, C. S. (1973). Resilience and stability of ecological systems. *Annual Review of Ecology and Systematics*, *4*(1), 1--23. https://doi.org/10.1146/annurev.es.04.110173.000245

> **Audit status: VERIFIED with attribution nuance.** Foundational paper introducing ecological resilience vs. engineering stability. The specific metric decomposition "resistance, recovery, regime retention" used in the codebase is a modern synthesis across Holling (1973), Pimm (1984), Walker et al. (2004), and Scheffer (2009) -- not Holling's own framework name. The code acknowledges this in `resilience_metrics.py` line 381. The term "regime retention" is modern terminology not found in Holling (1973).

Isbell, F., Craven, D., Connolly, J., Loreau, M., Schmid, B., Beierkuhnlein, C., Bezemer, T. M., Bonin, C., Bruelheide, H., de Luca, E., Ebeling, A., Griffin, J. N., Guo, Q., Hautier, Y., Hector, A., Jentsch, A., Kreyling, J., Lanta, V., Manning, P., ... Eisenhauer, N. (2015). Biodiversity increases the resistance of ecosystem productivity to climate extremes. *Nature*, *526*(7574), 574--577. https://doi.org/10.1038/nature15374

> **Audit status: VERIFIED, loose citation.** Paper is about biodiversity and resistance to climate extremes. Code uses it for a "5% recovery threshold" concept; the paper discusses resistance (not recovery) and found no dependence of resilience on biodiversity. Reasonable general citation but paper does not specifically define a 5% recovery threshold.

Pimm, S. L. (1984). The complexity and stability of ecosystems. *Nature*, *307*(5949), 321--326. https://doi.org/10.1038/307321a0

> **Audit status: VERIFIED with terminological discrepancy.** Pimm defined "resilience" as return rate to equilibrium and "resistance" as resistance to change. The code attributes "elasticity metrics" to this paper, but "elasticity" is not Pimm's term. The code's post-shock trajectory slope metric is a modern construct. Attributing resistance and recovery time to Pimm is reasonable.

Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical transitions. *Nature*, *461*(7260), 53--59. https://doi.org/10.1038/nature08227

> **Audit status: VERIFIED.** All metadata correct. Accurately used for regime shift detection and tipping point dynamics (critical slowing down, increased autocorrelation). Note: the codebase also references Scheffer's 2009 book *Critical Transitions in Nature and Society* (Princeton University Press) separately -- these are distinct publications.

Scheffer, M. (2009). *Critical transitions in nature and society*. Princeton University Press.

> **Audit status: VERIFIED.** Book exists. Referenced in `docs/resilience_metrics.md` separately from the Nature paper above.

---

## V. Pharmacological Models

Bliss, C. I. (1939). The toxicity of poisons applied jointly. *Annals of Applied Biology*, *26*(3), 585--615. https://doi.org/10.1111/j.1744-7348.1939.tb06990.x

> **Audit status: VERIFIED.** Foundational paper on the Bliss independence model for combined drug effects. Correctly attributed in `interaction_mapper.py`. Minor note: the actual code formula uses additive comparison rather than the multiplicative Bliss independence formula -- implementation discrepancy, not citation error.

Loewe, S. (1953). The problem of synergism and antagonism of combined drugs. *Arzneimittelforschung*, *3*(6), 285--290.

> **Audit status: VERIFIED.** PMID 13081480. Foundational paper on Loewe Additivity. Correctly attributed alongside Bliss (1939) for pharmacological combination screening.

---

## VI. Computational Methods

McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*, *21*(2), 239--245. https://doi.org/10.1080/00401706.1979.10489755

> **Audit status: VERIFIED.** Correctly attributed for Latin Hypercube Sampling in `reachable_set.py`.

---

## VII. Grief Biology

O'Connor, M.-F. (2022). *The grieving brain: The surprising science of how we learn from love and loss*. HarperOne.

> **Audit status: NOT INDIVIDUALLY AUDITED.** Referenced in grief-simulator project; imported into how-to-live-much-longer via `grief_bridge.py`. Provides biological stress mechanisms underlying grief-to-mitochondrial coupling.

O'Connor, M.-F. (2025). [Bereavement biological stress studies].

> **Audit status: NOT INDIVIDUALLY AUDITED.** Referenced alongside O'Connor (2022) for grief biology. Sleep coefficients in `constants.py` are from grief/bereavement literature, NOT from LEMURS data -- this distinction was correctly documented after the sleep coefficient audit.

---

## VIII. Computational Story Lab (Dodds & Danforth)

Bloomfield, A., Fudolig, M. I., et al. (2024). Predicting stress in first-year college students using sleep data from wearable devices. *PLOS Digital Health*, *3*(4), e0000473. https://doi.org/10.1371/journal.pdig.0000473

> **Audit status: NOT INDIVIDUALLY AUDITED.** LEMURS study (N=525, 3,112 observations). Referenced for sleep disruption impact coefficient calibration. Key finding: total sleep time OR 0.617 per lost hour (38.3% stress increase).

Dodds, P. S., & Watts, D. J. (2004). Universal behavior in a generalized model of contagion. *Physical Review Letters*, *92*(21), 218701. https://doi.org/10.1103/PhysRevLett.92.218701

> **Audit status: NOT INDIVIDUALLY AUDITED.** Referenced as threshold dynamics analogy for heteroplasmy cliff.

Dodds, P. S., & Watts, D. J. (2005). A generalized model of social and biological contagion. *Journal of Theoretical Biology*, *232*(4), 587--604. https://doi.org/10.1016/j.jtbi.2004.09.006

> **Audit status: NOT INDIVIDUALLY AUDITED.** Biological/social contagion unification; cliff mechanism analogy.

Dodds, P. S., & Danforth, C. M. (2010). Measuring the happiness of large-scale written expression: Songs, blogs, and presidents. *Journal of Happiness Studies*, *11*(4), 441--456.

> **Audit status: NOT INDIVIDUALLY AUDITED.** Hedonometrics methodology.

Dodds, P. S., Minot, J. R., et al. (2020). Allotaxonometry and rank-turbulence divergence: A universal instrument for comparing complex systems. *arXiv preprint*, arXiv:2002.09770.

> **Audit status: NOT INDIVIDUALLY AUDITED.** Information-theoretic protocol comparison methodology.

Dodds, P. S., Rothman, D. H., & Weitz, J. S. (2001). Re-examination of the "3/4-law" of metabolism. *Journal of Theoretical Biology*, *209*(1), 9--27. https://doi.org/10.1006/jtbi.2000.2238

> **Audit status: NOT INDIVIDUALLY AUDITED.** Metabolic scaling laws.

Dodds, P. S., et al. (2023). Ousiometrics and telegnomics: The essence of meaning conforms to a two-dimensional powerful-weak and dangerous-safe framework. *arXiv preprint*.

> **Audit status: NOT INDIVIDUALLY AUDITED.** PDS dimensional analysis for archetype mapping.

Fudolig, M. I., et al. (2024). The two fundamental shapes of sleep heart rate dynamics and their connection to mental health in college students. *Digital Biomarkers*, *8*(1), 120--131.

> **Audit status: NOT INDIVIDUALLY AUDITED.** LEMURS sleep architecture study (N~600, 25,000+ records).

Fudolig, M. I., Dodds, P. S., & Danforth, C. M. (2025). Collective sleep and activity patterns of college students from wearable devices. *npj Complexity*, *2*, 32. https://doi.org/10.1038/s44260-025-00055-x

> **Audit status: NOT INDIVIDUALLY AUDITED.** Population-level LEMURS sleep phenotypes. Chronic sleep debt ~45 min/day.

Gallagher, R. J., et al. (2021). Generalized word shift graphs: A method for visualizing and explaining pairwise comparisons between texts. *EPJ Data Science*, *10*(1), 4. https://doi.org/10.1140/epjds/s13688-021-00260-3

> **Audit status: NOT INDIVIDUALLY AUDITED.** Decomposition visualization methodology.

Reagan, A. J., et al. (2016). The emotional arcs of stories are dominated by six basic shapes. *EPJ Data Science*, *5*(1), 31. https://doi.org/10.1140/epjds/s13688-016-0093-1

> **Audit status: NOT INDIVIDUALLY AUDITED.** Trajectory classification; aging arc analogy.

Reece, A. G., & Danforth, C. M. (2017). Instagram photos reveal predictive markers of depression. *EPJ Data Science*, *6*(1), 15. https://doi.org/10.1140/epjds/s13688-017-0110-z

> **Audit status: NOT INDIVIDUALLY AUDITED.** Digital biomarker extraction methodology.

Zimmerman, J. W., Hudon, D., Cramer, K., St-Onge, J., Fudolig, M., Trujillo, M. Z., Danforth, C. M., & Dodds, P. S. (2024). A blind spot for large language models: Supradiegetic linguistic information. *Plutonics*, *17*, 107--156.

> **Audit status: NOT INDIVIDUALLY AUDITED.** LLM diegeticizer / narrative fidelity methodology. Note: Kathryn Cramer is a co-author.

---

## IX. Lit-Spider References (PubMed-Indexed)

The following references were retrieved by `lit_spider.py` and stored in `artifacts/lit_spider_report.md`. They are PubMed-indexed with valid PMIDs and DOIs. They have not been individually audited for claim accuracy but their existence is confirmed via PubMed metadata.

### Mitochondrial Dynamics & Aging

An, J., Nam, C. H., Kim, R., et al. (2024). Mitochondrial DNA mosaicism in normal human somatic cells. *Nature Genetics*, *56*(8), 1616--1628. https://doi.org/10.1038/s41588-024-01838-z

Bradshaw, E., Yoshida, M., & Ling, F. (2017). Regulation of small mitochondrial DNA replicative advantage by ribonucleotide reductase in *Saccharomyces cerevisiae*. *G3: Genes, Genomes, Genetics*, *7*(10), 3161--3172. https://doi.org/10.1534/g3.117.043851

Chabi, B., Mousson de Camaret, B., Duborjal, H., et al. (2003). Quantification of mitochondrial DNA deletion, depletion, and overreplication: Application to diagnosis. *Clinical Chemistry*, *49*(8), 1309--1317. https://doi.org/10.1373/49.8.1309

DiMauro, S., & Moraes, C. T. (1993). Mitochondrial encephalomyopathies. *Archives of Neurology*, *50*(11), 1197--1208. https://doi.org/10.1001/archneur.1993.00540110075008

Dubie, J. J., Caraway, A. R., & Stout, M. M. (2020). The conflict within: Origin, proliferation and persistence of a spontaneously arising selfish mitochondrial genome. *Philosophical Transactions of the Royal Society B*, *374*(1768), 20190174. https://doi.org/10.1098/rstb.2019.0174

Fu, Y., Land, M., & Kavlashvili, T. (2025). Engineering mtDNA deletions by reconstituting end joining in human mitochondria. *Cell*, *188*(2), 456--472. https://doi.org/10.1016/j.cell.2025.02.009

Kimoloi, S., Sen, A., Guenther, S., et al. (2022). Combined fibre atrophy and decreased muscle regeneration capacity driven by mitochondrial DNA alterations underlie the development of sarcopenia. *Journal of Cachexia, Sarcopenia and Muscle*, *13*(4), 1851--1867. https://doi.org/10.1002/jcsm.13026

Shammas, M. K., Nie, Y., Gilsrud, A., et al. (2023). CHCHD10 mutations induce tissue-specific mitochondrial DNA deletions with a distinct signature. *Human Molecular Genetics*, *32*(21), 3261--3275. https://doi.org/10.1093/hmg/ddad161

Wei, Y. H., Lee, C. F., & Lee, H. C. (2001). Increases of mitochondrial mass and mitochondrial genome in association with enhanced oxidative stress in human cells harboring 4,977 BP-deleted mitochondrial DNA. *Annals of the New York Academy of Sciences*, *928*(1), 97--112. https://doi.org/10.1111/j.1749-6632.2001.tb05640.x

Yokota, M., Hatakeyama, H., & Okabe, S. (2015). Mitochondrial respiratory dysfunction caused by a heteroplasmic mitochondrial DNA mutation blocks cellular reprogramming. *Human Molecular Genetics*, *24*(12), 3437--3446. https://doi.org/10.1093/hmg/ddv201

Yu-Wai-Man, P., Lai-Cheong, J., & Borthwick, G. M. (2010). Somatic mitochondrial DNA deletions accumulate to high levels in aging human extraocular muscles. *Investigative Ophthalmology & Visual Science*, *51*(6), 3347--3353. https://doi.org/10.1167/iovs.09-4660

Zhu, W., Yoshida, M., & Ling, F. (2025). Caloric restriction enhances inheritance of wild-type mitochondrial DNA in *Saccharomyces cerevisiae*. *Scientific Reports*, *15*, 23888. https://doi.org/10.1038/s41598-025-23888-x

### ROS & Oxidative Stress

Cadenas, S. (2018). Mitochondrial uncoupling, ROS generation and cardioprotection. *Biochimica et Biophysica Acta*, *1859*(10), 940--950. https://doi.org/10.1016/j.bbabio.2018.05.019

Fang, J., Wong, H. S., & Brand, M. D. (2020). Production of superoxide and hydrogen peroxide in the mitochondrial matrix is dominated by site I of complex I. *Redox Biology*, *36*, 101722. https://doi.org/10.1016/j.redox.2020.101722

Grivennikova, V. G., & Vinogradov, A. D. (2006). Generation of superoxide by the mitochondrial complex I. *Biochimica et Biophysica Acta*, *1757*(12), 1594--1600. https://doi.org/10.1016/j.bbabio.2006.03.013

Grivennikova, V. G., Kareyeva, A. V., & Vinogradov, A. D. (2018). Oxygen-dependence of mitochondrial ROS production as detected by Amplex Red assay. *Redox Biology*, *14*, 450--461. https://doi.org/10.1016/j.redox.2018.04.014

St-Pierre, J., Buckingham, J. A., & Roebuck, S. J. (2002). Topology of superoxide production from different sites in the mitochondrial electron transport chain. *Journal of Biological Chemistry*, *277*(47), 44784--44790. https://doi.org/10.1074/jbc.M207217200

### NAD+ Biology

Hu, Y., Wang, H., & Wang, Q. (2014). Overexpression of CD38 decreases cellular NAD levels and alters the expression of proteins involved in energy metabolism and antioxidant defense. *Journal of Proteome Research*, *13*(11), 5126--5140. https://doi.org/10.1021/pr4010597

Poljsak, B. (2018). NAMPT-mediated NAD biosynthesis as the internal timing mechanism: In NAD+ world, time is running in its own way. *Rejuvenation Research*, *21*(3), 263--283. https://doi.org/10.1089/rej.2017.1975

Rechsteiner, M., Hillyard, D., & Olivera, B. M. (1976). Turnover at nicotinamide adenine dinucleotide in cultures of human cells. *Journal of Cell Physiology*, *88*(2), 273--280. https://doi.org/10.1002/jcp.1040880210

### Senescence & Autophagy

Alsuraih, M., O'Hara, S. P., & Woodrum, J. E. (2021). Genetic or pharmacological reduction of cholangiocyte senescence improves inflammation and fibrosis in the MCD diet-induced injury model. *JHEP Reports*, *3*(3), 100250. https://doi.org/10.1016/j.jhepr.2021.100250

Ghamar Talepoor, A., Khosropanah, S., & Doroudchi, M. (2021). Partial recovery of senescence in circulating follicular helper T cells after dasatinib treatment. *International Immunopharmacology*, *92*, 107465. https://doi.org/10.1016/j.intimp.2021.107465

Goya, R. G., Lehmann, M., & Chiavellini, P. (2018). Rejuvenation by cell reprogramming: A new horizon in gerontology. *Stem Cell Reviews and Reports*, *14*(1), 13--26. https://doi.org/10.1186/s13287-018-1075-y

Tang, Q., Tang, K., & Markby, G. R. (2025). Autophagy regulates cellular senescence by mediating the degradation of CDKN1A/p21 and CDKN2A/p16 through SQSTM1/p62-mediated selective autophagy in myxomatous mitral valve degeneration. *Autophagy*. https://doi.org/10.1080/15548627.2025.2469315

### Exercise & Biogenesis

Latimer, L. E., Constantin-Teodosiu, D., & Popat, B. (2022). Whole-body and muscle responses to aerobic exercise training and withdrawal in ageing and COPD. *European Respiratory Journal*, *60*(4), 2101507. https://doi.org/10.1183/13993003.01507-2021

Molmen, K. S., Almquist, N. W., & Skattebo, O. (2025). Effects of exercise training on mitochondrial and capillary growth in human skeletal muscle: A systematic review and meta-regression. *Sports Medicine*, *55*, 31--52. https://doi.org/10.1007/s40279-024-02120-2

Reisman, E. G., Caruana, N. J., & Bishop, D. J. (2024). Exercise training and changes in skeletal muscle mitochondrial proteins: From blots to "omics." *Critical Reviews in Biochemistry and Molecular Biology*, *59*(5), 445--469. https://doi.org/10.1080/10409238.2024.2383408

### Cardiac & Metabolic

Bombicino, S. S., Iglesias, D. E., & Rukavina-Mikusic, I. A. (2017). Hydrogen peroxide, nitric oxide and ATP are molecules involved in cardiac mitochondrial biogenesis in diabetes. *Free Radical Biology and Medicine*, *110*, 146--155. https://doi.org/10.1016/j.freeradbiomed.2017.07.027

Rudokas, M. W., McKay, M., & Toksoy, Z. (2024). Mitochondrial network remodeling of the diabetic heart: Implications to ischemia related cardiac dysfunction. *Cardiovascular Diabetology*, *23*, 176. https://doi.org/10.1186/s12933-024-02357-1

Sun, C., Liu, X., & Wang, B. (2019). Endocytosis-mediated mitochondrial transplantation: Transferring normal human astrocytic mitochondria into glioma cells rescues aerobic respiration and enhances radiosensitivity. *Theranostics*, *9*(12), 3595--3610. https://doi.org/10.7150/thno.33100

### Longevity Genetics

Shadyab, A. H., & LaCroix, A. Z. (2015). Genetic factors associated with longevity: A review of recent findings. *Ageing Research Reviews*, *19*, 1--10. https://doi.org/10.1016/j.arr.2014.10.005

### Historical / Foundational

Hayakawa, M., Torii, K., & Sugiyama, S. (1991). Age-associated accumulation of 8-hydroxydeoxyguanosine in mitochondrial DNA of human diaphragm. *Biochemical and Biophysical Research Communications*, *179*(2), 1023--1029. https://doi.org/10.1016/0006-291x(91)91921-x

Hayakawa, M., Hattori, K., & Sugiyama, S. (1992). Age-associated oxygen damage and mutations in mitochondrial DNA in human hearts. *Biochemical and Biophysical Research Communications*, *189*(2), 979--985. https://doi.org/10.1016/0006-291x(92)92300-m

Lambeth, J. D., McCaslin, D. R., & Kamin, H. (1976). Adrenodoxin reductase-adrenodexin complex. *Journal of Biological Chemistry*, *251*(14), 4395--4400.

---

## X. Additional References (from code comments, not individually audited)

Kennedy, S. R., et al. (2013). [mtDNA point mutation accumulation]. Referenced in `constants.py` for Pol gamma error rates.

Zheng, W., et al. (2006). [mtDNA mutagenesis]. Referenced in `constants.py` for polymerase fidelity.

Cai, R., et al. (2025). [Implied citation]. Referenced in `constants.py` for homozygous APOE4 tau pathology scaling (~1.6x heterozygous).

Friday, [initial unknown]. (2025). [Implied reference]. Referenced in `constants.py` for APOE4 heightened inflammatory responses.

---

## Verification Methodology

1. **Extraction:** Two parallel agents scanned all `.py` files and all `.md` files in the repository, identifying every literature reference, citation, DOI, PMID, and bibliographic mention.

2. **Verification:** Three parallel agents performed web-based verification of 25 core references across three domains:
   - APOE4 & neurogenetics (8 references)
   - Mitochondrial biology & NAD+ metabolism (8 references)
   - Ecological resilience & computational methods (9 references)

3. **Verification criteria:**
   - Paper exists (PubMed, DOI resolution, publisher site)
   - Metadata correct (authors, year, journal, volume, pages, DOI)
   - Claimed finding matches actual paper content

4. **Lit-spider references:** The ~80 PubMed-indexed references retrieved by `lit_spider.py` were confirmed to exist via PMID/DOI but were not individually audited for claim accuracy. They are included in the bibliography with "NOT INDIVIDUALLY AUDITED" annotations.

5. **Scope:** This audit covers only the `how-to-live-much-longer` repository. References from sibling projects (grief-simulator, ea-toolkit, cramer-toolkit, zimmerman-toolkit) were included only where they are imported into this project.
