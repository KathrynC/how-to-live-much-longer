# Cramer Compliance Check: Proposed Parameter Changes

**Date**: 2026-02-22
**Analyst**: Claude Opus 4.6
**Purpose**: Cross-check proposed simulation changes against John G. Cramer's book (*How to Live Much Longer*, forthcoming 2026, Springer) and email corrections (C7-C11, 2026-02-15 through 2026-02-17).

**Sources consulted**:
- `CLAUDE.md` — Action Items section with Cramer email corrections C7-C11
- `constants.py` — All constants with provenance comments citing specific Cramer book pages
- `artifacts/cramer_corrections_2026-02-15.md` — Cramer's email corrections (C7, C8)
- `artifacts/cramer_questions_2026-02-16.md` — Questions from Cramer about the simulation
- `artifacts/lit_search/consensus_finding_{1-5}.md` — Cross-model literature consensus analyses

---

## Finding 1: Exercise Parameters

### Proposed Changes

| Parameter | Current | Proposed | Source |
|-----------|---------|----------|--------|
| EXERCISE_BIOGENESIS_FACTOR | 0.03 | 0.08 | consensus_finding_1.md |
| EXERCISE_METABOLIC_COST | 0.03 | 0.02 | consensus_finding_1.md |
| Exercise-to-mitophagy channel | absent | 0.01 | consensus_finding_1.md |

### Cramer's Stated Positions

**Book references**: Exercise is mentioned in the 12D parameter space as `exercise_level` with effect "hormetic_adaptation" (`constants.py`, line 398). The simulator implements exercise through three channels:
1. Biogenesis: `exercise * 0.03 * energy_available * copy_number_pressure * tissue_mods["biogenesis_rate"]` (`simulator.py`, line 769)
2. Metabolic cost: `exercise * 0.03` (`simulator.py`, line 923)
3. Antioxidant defense: `defense_factor += exercise * 0.2` (`simulator.py`, line ~968)

The biogenesis coefficient (0.03) and metabolic cost coefficient (0.03) are both marked as **simulation parameters**, not Cramer book values. The comment at `simulator.py:921` explicitly states: "This is deliberately small because exercise benefits (biogenesis, hormesis) outweigh the metabolic cost at moderate levels."

Cramer's email corrections (C7-C11) do not address exercise at all. The corrections focus on:
- C7: CD38 degradation of NMN/NR (NAD pathway)
- C8: Transplant as primary rejuvenation
- C9: AGE_TRANSITION restored to 65
- C10: AGE_TRANSITION coupled to ATP and mitophagy
- C11: Split mutation types (deletions vs point mutations)

In the Cramer corrections document (`cramer_corrections_2026-02-15.md`), exercise is mentioned only in passing: "All other interventions (rapamycin, NAD+, senolytics, exercise) can only slow the rate of damage accumulation." This indicates Cramer views exercise as a damage-slowing intervention, not a rejuvenation mechanism like transplant.

In `cramer_questions_2026-02-16.md`, exercise appears in predictions (rapamycin + exercise synergy) and tissue profiles (muscle has high exercise-responsive biogenesis), but Cramer does not comment specifically on exercise coefficient values.

### Does Cramer Discuss Exercise Effects on Mitochondria?

**Partially.** The book references in `constants.py` do not cite specific Cramer pages for exercise parameters. The `TISSUE_PROFILES` in `constants.py` reference Cramer Ch. V.J p.65 for tissue-specific copy number variation and exercise-responsive biogenesis in muscle (1.5x) vs brain (0.3x), but the exercise biogenesis coefficient itself is marked as a simulation parameter. The description "Hormesis: moderate ROS -> adaptation" is a standard exercise biology concept, not a Cramer-specific claim.

### Does Cramer Endorse Exercise as Beneficial?

**Implicitly yes.** Exercise is included as one of the six intervention parameters in the 12D parameter space. Its description emphasizes hormetic adaptation. Cramer's email correction on transplant (C8) groups exercise with rapamycin, NAD+, and senolytics as interventions that "can only slow the rate of damage accumulation" -- which is a positive but limited endorsement. There is no indication that Cramer considers exercise harmful.

### Verdict: SAFE TO PROCEED

**Rationale**: The exercise coefficients (EXERCISE_BIOGENESIS_FACTOR=0.03, EXERCISE_METABOLIC_COST=0.03) are explicitly marked as simulation parameters, not Cramer book values. No Cramer email correction addresses exercise. The proposed changes are consistent with Cramer's implicit endorsement of exercise as beneficial (he included it as an intervention parameter), and the current parameterization producing universally harmful exercise outcomes contradicts both Cramer's framing and the broader literature. The changes bring the model into alignment with what Cramer would expect exercise to do.

**One caution**: The proposed exercise-to-mitophagy channel (0.01) adds a new coupling pathway. While Cramer discusses the PINK1/Parkin mitophagy pathway extensively (Ch. VI.B p.75), he does not specifically link exercise to mitophagy enhancement. This channel is a modeling addition based on general exercise physiology literature, not Cramer's book.

**Recommendation**: Implement the changes. No need to flag for Cramer review, since these are simulation parameters that Cramer has not constrained. However, if communicating results to Cramer, note that exercise now correctly produces net benefit at moderate doses.

---

## Finding 2: Sleep Coefficients

### Proposed Changes

| Parameter | Current | Proposed | Source |
|-----------|---------|----------|--------|
| SLEEP_ROS_COEFF | 0.04 | 0.06 | consensus_finding_2.md |
| SLEEP_NAD_DRAIN_COEFF | 0.02 | 0.03 | consensus_finding_2.md |

### Cramer's Stated Positions

**These constants are entirely outside Cramer's book scope.** The sleep trajectory model is part of the Precision Medicine Expansion (2026-02-19), which the CLAUDE.md explicitly marks as: "IMPORTANT: Nothing below modifies the Cramer core ODE. These constants are consumed by the parameter resolver... never by simulator.py or derivatives()."

The sleep coefficients are marked with provenance grades in `constants.py`:
- `SLEEP_ROS_COEFF = 0.04` — provenance grade [B], citing Everson et al. 2005 (Sleep) and Villafuerte et al. 2015 (Sleep)
- `SLEEP_NAD_DRAIN_COEFF = 0.02` — provenance grade [C], citing Massudi et al. 2012 (PLoS ONE) and Ramsey et al. 2009 (Science)

Cramer's book focuses on mitochondrial DNA damage mechanisms, not sleep biology. None of Cramer's email corrections (C7-C11) mention sleep. The CLAUDE.md action item about sleep coefficients says: "ASK DODDS & DANFORTH (LEMURS data)" -- indicating these coefficients require validation from the LEMURS research team, not from Cramer.

### Verdict: SAFE TO PROCEED

**Rationale**: Sleep coefficients are entirely outside Cramer's scope. His book addresses mitochondrial DNA deletion dynamics, heteroplasmy cliffs, NAD+ metabolism, transplant biology, and mutation types -- not sleep physiology. The proposed changes are modest adjustments to simulation parameters that do not touch the Cramer core ODE. They are motivated by general sleep biology literature (Everson 2005, Ramsey 2009) and consensus analysis, not by any Cramer claim.

**No conflict, no constraint**: Cramer has neither endorsed nor constrained these values. They operate in the parameter resolver layer, which is architecturally separated from his core model. No need to flag for Cramer review.

---

## Finding 3: APOE4 x Sleep Interaction

### Proposed Changes

| Change | Description |
|--------|-------------|
| APOE4 amplification of sleep inflammation channel | Add `apoe4_infl_amplifier` to inflammation_delta in `sleep_trajectory.py` |
| APOE4 amplification of sleep ROS channel | Add `apoe4_ros_amplifier` to ros_boost in `sleep_trajectory.py` |
| Fix sleep_repair_factor formula | Change from `1.0 - (SLEEP_REPAIR_COEFF / mitophagy_eff) * deficit` to a multiplicative amplification that makes APOE4 carriers MORE vulnerable (not less) |

### Cramer's Stated Positions

**Cramer does not discuss APOE4.** A thorough search of all Cramer-related files in the project reveals:
- No mention of APOE4, APOE, or apolipoprotein E in `cramer_corrections_2026-02-15.md`
- No mention in `cramer_questions_2026-02-16.md`
- No APOE4 reference in the Cramer book page citations within `constants.py`
- The GENOTYPE_MULTIPLIERS section in `constants.py` (lines 774-831) is explicitly marked as "qualitative estimates" with provenance grade [C], referencing O'Shea et al. 2024, Anttila et al. 2004, Castellano et al. 2011, Shi et al. 2017, Therriault et al. 2020, and Dumanis et al. 2009 -- none of which are Cramer citations.

Cramer's book discusses haplogroup-dependent genetic vulnerability (`genetic_vulnerability` parameter, range 0.5-2.0), but this is about mtDNA haplogroups (inherited mitochondrial DNA variants), not nuclear gene variants like APOE4. The APOE4 modeling is entirely a precision medicine expansion addition.

The `CLAUDE.md` action item for Finding 3 already acknowledges this: "REVIEW WITH CRAMER: Does the APOE4 x mitochondrial function interaction align with Cramer's model?" This is flagged as an open question, not as something Cramer has opined on.

### Does Cramer Take a Position on Genotype-Dependent Vulnerability?

**Yes, but only for mtDNA haplogroups.** The `genetic_vulnerability` parameter (Cramer's concept) represents haplogroup-dependent susceptibility to mtDNA damage. This is a different genetic axis than APOE4, which is a nuclear gene affecting apolipoprotein E isoforms. Cramer's haplogroup vulnerability is about variations in the mitochondrial genome itself; APOE4 is about a nuclear gene that indirectly affects mitochondrial function through lipid metabolism, inflammation, and glymphatic clearance pathways.

### Verdict: SAFE TO PROCEED (with Cramer review recommended for awareness)

**Rationale**: The APOE4 x sleep interaction is entirely outside Cramer's stated scope. His book focuses on mitochondrial DNA, not nuclear gene variants. The proposed changes fix a clear modeling bug (APOE4 carriers showing LESS sleep vulnerability when they should show MORE) and are well-supported by the consensus literature analysis (3/3 models unanimously agree APOE4 amplifies sleep vulnerability).

**Flag for Cramer review (informational, not blocking)**: Since the APOE4 modifications interact with the core ODE through the parameter resolver (affecting rapamycin_dose, inflammation_level, and other resolver-mediated parameters), Cramer should be informed that the precision medicine layer now includes APOE4-dependent vulnerability amplification. This does not change any Cramer core ODE parameter or equation, but it does affect simulation outcomes for APOE4 carrier patients.

---

## Finding 4: Sleep Architecture (Zero-Point)

### Proposed Changes

| Change | Current | Proposed |
|--------|---------|----------|
| Deficit calculation | `deficit = 1.0 - quality` | `deficit = baseline_q - quality` |
| Neutral point | `sleep_intervention=0.5` always penalizing at age 70 | `sleep_intervention=0.5` neutral at any age |

### Cramer's Stated Positions

**Cramer's book does not address sleep quality modeling.** The sleep trajectory model is entirely a precision medicine expansion. Cramer's core model addresses:
- mtDNA deletion dynamics (doubling times, replication advantages)
- Heteroplasmy cliff threshold
- NAD+ metabolism and CD38 degradation
- Transplant biology
- Energy costs of interventions

Sleep is not a topic in Cramer's book. The concept of age-dependent sleep quality decline comes from Ohayon et al. 2004 (Sleep Medicine Reviews) and Mander et al. 2017 (Neuron), not from Cramer.

### Does the Book Treat Normal Aging Sleep as a Stressor or as the Baseline?

**Not applicable.** Cramer's model treats aging as a mitochondrial energy crisis driven by mtDNA deletion accumulation, not by sleep quality. The core ODE's age-dependent dynamics (deletion doubling time transition at age 65, NAD decline rate of 0.01/year, senescence rate) implicitly represent a "typical aging human" whose sleep quality is whatever is normal for their age. The raw simulator does not model sleep at all.

### Cramer's Position on How Sleep Quality Should Be Modeled Relative to Age

**No position.** Cramer has not commented on the sleep model. His email corrections (C7-C11) address NAD+ metabolism, transplant biology, mutation types, and deletion dynamics -- all core ODE topics.

### Architectural Implications

The key insight from the consensus analysis (Finding 4) is that the raw simulator's ODE already represents a typical aging person, and adding the sleep resolver on top should model DEVIATIONS from typical, not penalties from a hypothetical young-adult optimum. This is a modeling architecture decision, not a Cramer-constrained parameter.

### Verdict: SAFE TO PROCEED

**Rationale**: This is purely a modeling architecture decision about how the precision medicine resolver interfaces with the Cramer core ODE. The change ensures that adding the sleep channel does not systematically degrade outcomes relative to the raw simulator. Cramer has no stated position on sleep modeling, and the change does not modify any Cramer core ODE parameter or equation.

**No need to flag for Cramer review**: This is internal resolver architecture, not a biological claim that Cramer would weigh in on.

---

## Finding 5: Transplant Parameters

### Proposed Changes

| Parameter | Current | Proposed | Source |
|-----------|---------|----------|--------|
| TRANSPLANT_ADDITION_RATE | 0.30 | 0.15-0.25 | consensus_finding_5.md |
| TRANSPLANT_HEADROOM | 1.5 | 1.2-1.3 | consensus_finding_5.md |

### Cramer's Stated Positions

**THIS IS THE ONLY FINDING THAT DIRECTLY CONFLICTS WITH A CRAMER EMAIL CORRECTION.**

Cramer correction C8 (email 2026-02-15, documented in `cramer_corrections_2026-02-15.md` and `CLAUDE.md`) explicitly made two changes:

1. **TRANSPLANT_ADDITION_RATE doubled from 0.15 to 0.30**: "Addition rate doubled (0.15 -> 0.30)" per Cramer's directive that "The simulation underemphasized the value of transplantation with new externally-produced mitochondria containing unmutated mtDNA."

2. **TRANSPLANT_HEADROOM raised from 1.2 to 1.5**: "Headroom raised (1.2 -> 1.5)" per the same correction.

3. **TRANSPLANT_DISPLACEMENT_RATE = 0.12 added**: A new mechanism for competitive displacement of damaged copies, also per C8.

Cramer's reasoning (from `cramer_corrections_2026-02-15.md`):

> "The simulation underemphasized the value of transplantation with new externally-produced mitochondria containing unmutated mtDNA. This is the only available method of reversing the accumulated damage to mtDNA at a scale that could be called rejuvenation."

The correction was verified with 4 specific tests:
- Transplant benefit (0.416 het reduction) is 2.3x NAD benefit (0.183)
- Near-cliff rescue: 80-year-old at 65% het drops to 13.9% het with transplant + rapamycin
- ATP recovers from 0.049 to 0.642 in the near-cliff rescue scenario

### Does Reducing These Parameters Contradict Cramer's C8 Correction?

**YES. This directly reverses Cramer's explicit instruction.**

- Reducing TRANSPLANT_ADDITION_RATE from 0.30 back toward 0.15-0.25 partially or fully undoes Cramer's doubling.
- Reducing TRANSPLANT_HEADROOM from 1.5 back toward 1.2-1.3 partially or fully undoes Cramer's headroom increase.

The consensus analysis (Finding 5) recommends these reductions based on literature suggesting that current parameters may cause transplant to saturate too early (62% of max benefit at 10% dose). However, the literature consensus does not have direct empirical calibration data for these parameters -- it merely notes that:
- Engraftment efficiency is 3-15% (suggesting high addition rate may overstate persistence)
- Long-term persistence data is lacking
- The positive feedback loop (ATP -> mitophagy -> clearance) may be weaker than modeled

These are reasonable scientific concerns, but they are modeling uncertainties, not empirical constraints. Cramer made a deliberate judgment call that transplant was UNDEREMPHASIZED, and the current parameters reflect his correction.

### What Was Cramer's Reasoning for the Increase?

Cramer's reasoning was based on his assessment of the relative importance of interventions in his theoretical framework:

1. **Transplant is the ONLY method for true rejuvenation**: All other interventions (rapamycin, NAD+, senolytics, exercise) can only slow damage accumulation. Transplant adds healthy copies AND displaces damaged ones.

2. **The original simulation undervalued transplant**: At the old rate (0.15), transplant was only modestly better than NAD+ supplementation. Cramer considered this unrealistic given the mechanistic advantage of transplant (direct mtDNA replacement vs indirect metabolic support).

3. **Near-cliff rescue should be possible**: Cramer's theory implies that if you can deliver enough healthy mitochondria, you should be able to pull a patient back from the heteroplasmy cliff. The doubled rate enables this rescue scenario.

### TRANSPLANT_DISPLACEMENT_RATE (0.12)

The consensus analysis does not specifically recommend changing TRANSPLANT_DISPLACEMENT_RATE. It notes that displacement is "mechanistically plausible but quantitatively unconstrained." This parameter was added as part of C8 but was not a direct Cramer instruction -- it was an implementation detail to make transplant work as Cramer described (healthy mitos outcompeting damaged ones).

### Verdict: CONTRADICTS CRAMER -- Needs Cramer Review Before Any Change

**Rationale**: Cramer correction C8 is the most authoritative source for these parameters. Cramer explicitly reviewed the simulation and determined that transplant was underemphasized. Reducing TRANSPLANT_ADDITION_RATE or TRANSPLANT_HEADROOM directly reverses his correction without his approval.

**The literature consensus does not override Cramer's judgment here** because:
1. There is no direct empirical data constraining these parameters (consensus analysis Finding 5 rates all quantitative confidence as LOW to VERY LOW).
2. Cramer's book contains theoretical arguments for transplant primacy that the LLM literature search may not have captured.
3. The saturation behavior (62% benefit at 10% dose) may be addressable without changing the Cramer-directed parameters -- for example, by adjusting the feedback loop strength (MITOPHAGY_ATP_MIDPOINT, TRANSPLANT_HET_PENALTY parameters) rather than the base transplant rate.

**Recommended action**: Present the saturation analysis to Cramer and ask:
1. Is 62% of max benefit at 10% transplant dose consistent with his expectations?
2. Would he prefer to reduce the addition rate / headroom, or to modify the feedback dynamics that cause early saturation?
3. Does his book contain quantitative dose-response expectations for transplant?

**Do NOT reduce TRANSPLANT_ADDITION_RATE or TRANSPLANT_HEADROOM without Cramer's explicit approval.**

---

## Summary Table

| Finding | Proposed Change | Cramer Scope | Verdict |
|---------|----------------|:------------:|---------|
| **1. Exercise** | Biogenesis 0.03->0.08, cost 0.03->0.02, add mitophagy channel | OUTSIDE | **SAFE TO PROCEED** |
| **2. Sleep Coefficients** | ROS 0.04->0.06, NAD drain 0.02->0.03 | OUTSIDE | **SAFE TO PROCEED** |
| **3. APOE4 x Sleep** | Add APOE4 amplification to inflammation, ROS, fix repair formula | OUTSIDE | **SAFE TO PROCEED** (inform Cramer) |
| **4. Sleep Zero-Point** | Deficit from baseline instead of optimal | OUTSIDE | **SAFE TO PROCEED** |
| **5. Transplant Parameters** | Addition rate 0.30->0.15-0.25, headroom 1.5->1.2-1.3 | **DIRECTLY CONSTRAINED BY C8** | **CONTRADICTS CRAMER** |

### Decision Framework

| Verdict | Meaning | Action Required |
|---------|---------|----------------|
| **SAFE TO PROCEED** | No Cramer constraint exists; change is consistent with or orthogonal to Cramer's positions | Implement without blocking on Cramer review |
| **CONTRADICTS CRAMER** | Change reverses a specific Cramer email correction | Do NOT implement without explicit Cramer approval; present data and ask |

---

## Appendix: Cramer Email Corrections Reference

| ID | Date | Topic | Key Change | Files Modified |
|----|------|-------|-----------|----------------|
| C7 | 2026-02-15 | CD38 degrades NMN/NR | NAD+ boost gated by CD38 survival factor; coefficient reduced from 0.35 to 0.25 * cd38_survival | constants.py, simulator.py |
| C8 | 2026-02-15 | Transplant is primary rejuvenation | Addition rate 0.15->0.30; headroom 1.2->1.5; competitive displacement (0.12) added | constants.py, simulator.py |
| C9 | 2026-02-15 | AGE_TRANSITION restored to 65 | Fixed from incorrect age 40 | constants.py, simulator.py |
| C10 | 2026-02-15 | AGE_TRANSITION coupled to ATP/mitophagy | Dynamic transition age, sigmoid blend | simulator.py |
| C11 | 2026-02-17 | Split mutation types | N_damaged split into N_deletion (exponential, drives cliff) + N_point (linear, evades mitophagy) | constants.py, simulator.py |

## Appendix: Cramer Book Page References in constants.py

| Constant | Cramer Reference | Status |
|----------|-----------------|--------|
| HETEROPLASMY_CLIFF=0.50 | Recalibrated from literature 0.70 for C11 deletion-only het; Ch. V.K p.66 | CRAMER-DERIVED |
| DOUBLING_TIME_YOUNG=11.8 | Appendix 2, p.155, Fig. 23 (Va23) | CRAMER-SPECIFIED |
| DOUBLING_TIME_OLD=3.06 | Appendix 2, p.155, Fig. 23 (Va23) | CRAMER-SPECIFIED |
| AGE_TRANSITION=65.0 | Appendix 2, p.155 (corrected per C9) | CRAMER-SPECIFIED |
| BASELINE_ATP=1.0 | Ch. VIII.A, Table 3, p.100 | CRAMER-SPECIFIED |
| NAD_DECLINE_RATE=0.01 | Ch. VI.A.3, pp.72-73 (Ca16) | SIM PARAM (book-referenced mechanism) |
| SENESCENCE_RATE=0.005 | Ch. VII.A, pp.89-92; Ch. VIII.F, p.103 | SIM PARAM (book-referenced mechanism) |
| YAMANAKA_ENERGY_COST=3-5 MU | Ch. VIII.A, Table 3, p.100 | CRAMER-SPECIFIED |
| BASELINE_MITOPHAGY_RATE=0.02 | Ch. VI.B, p.75 (PINK1/Parkin) | SIM PARAM (book-referenced mechanism) |
| CD38_BASE_SURVIVAL=0.4 | Ch. VI.A.3, p.73 (per C7) | CRAMER-SPECIFIED |
| TRANSPLANT_ADDITION_RATE=0.30 | Ch. VIII.G, pp.104-107 (per C8) | CRAMER-SPECIFIED |
| TRANSPLANT_HEADROOM=1.5 | Ch. VIII.G, pp.104-107 (per C8) | CRAMER-SPECIFIED |
| DELETION_REPLICATION_ADVANTAGE=1.21 | Appendix 2, pp.154-155 (Va23) | CRAMER-SPECIFIED |
| EXERCISE_BIOGENESIS (0.03) | Not from book | SIM PARAM (unconstrained) |
| EXERCISE_COST (0.03) | Not from book | SIM PARAM (unconstrained) |
| SLEEP_* coefficients | Not from book | SIM PARAM (unconstrained) |
| APOE4 multipliers | Not from book | SIM PARAM (unconstrained) |
