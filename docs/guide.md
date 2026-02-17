# How to Live Much Longer — Reference Guide

Mitochondrial aging simulation and LLM-mediated intervention design

---

## From Book to Simulator

This project implements a computational simulator for the mitochondrial theory of aging described in John G. Cramer's forthcoming book *How to Live Much Longer* (Springer Verlag, 2026). The simulator models 7 coupled state variables — healthy and damaged mtDNA copy counts, ATP, ROS, NAD+, senescent fraction, and membrane potential — integrated via 4th-order Runge-Kutta over a 30-year horizon.

The project was created by Kathryn Cramer as a computational implementation and extension of John G. Cramer's biological framework, using Claude as a development utility. It adapts the TIQM (Transactional Interpretation of Quantum Mechanics) pipeline from the parent Evolutionary-Robotics project: instead of LLM → physics simulation → VLM scoring for robot locomotion, it uses LLM → mitochondrial ODE simulation → VLM scoring for intervention protocol design.

The Zimmerman-informed components — prompt templates, PDS mapping, POSIWID auditing — were added as a 2026-02-15 upgrade, operationalizing key findings from Julia Witte Zimmerman's 2025 PhD dissertation at the University of Vermont: *"Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)."* These were the original domain-specific implementations later generalized into the [zimmerman-toolkit](../../zimmerman-toolkit).

---

## Overview

The project provides five Zimmerman-informed tools alongside the core simulator and analytics:

| Tool | Question | Source |
|------|----------|--------|
| `prompt_templates.py` | How should we ask an LLM to design interventions? | Zimmerman §2.2.3, §3.5.3, §4.7.6 |
| `pds_mapping.py` | Can fictional character archetypes predict patient parameters? | Zimmerman §4.6.4; Dodds et al. 2023 |
| `posiwid_audit.py` | Does the LLM's intervention do what it claims? | Beer 1974; Zimmerman §3.5.2 |
| `archetype_matchmaker.py` | Which character types produce the best protocols for which patients? | Zimmerman §4.6.4 |
| `sobol_sensitivity.py` | Which of the 12 parameters actually drive outcomes? | Saltelli 2002; Jansen 1999 |

Plus the core infrastructure:

| Tool | Question |
|------|----------|
| `simulator.py` | What happens to this patient over 30 years? |
| `analytics.py` | How do we measure outcomes across 4 pillars? |
| `llm_common.py` | How do we parse and validate LLM-generated parameter vectors? |

---

## The 12D Parameter Space

The LLM generates a 12-dimensional vector for each clinical scenario:

**6 Intervention parameters** (0.0–1.0, snapped to grid [0, 0.1, 0.25, 0.5, 0.75, 1.0]):

| Parameter | Controls | Key Biology |
|-----------|----------|-------------|
| `rapamycin_dose` | mTOR inhibition → enhanced mitophagy | Clears damaged mitochondria |
| `nad_supplement` | NAD+ precursor (NMN/NR) | CD38 destroys low dose; needs apigenin (Cramer Ch. VI.A.3 p.73) |
| `senolytic_dose` | Senescent cell clearance | Frees energy budget from zombie cells |
| `yamanaka_intensity` | Partial reprogramming (OSKM) | **Costs 3-5 MU ATP** (Cramer Ch. VIII.A p.100) |
| `transplant_rate` | Healthy mtDNA via mitlets | **Only true rejuvenation** — displaces damaged copies |
| `exercise_level` | Hormetic adaptation | Moderate ROS → antioxidant upregulation |

**6 Patient parameters** (ranges vary):

| Parameter | Range | Key Biology |
|-----------|-------|-------------|
| `baseline_age` | 20–90 | |
| `baseline_heteroplasmy` | 0.0–0.95 | **Cliff at 0.70** — ATP collapses above this |
| `baseline_nad_level` | 0.2–1.0 | Declines with age |
| `genetic_vulnerability` | 0.5–2.0 | Haplogroup-dependent |
| `metabolic_demand` | 0.5–2.0 | Brain=high, skin=low |
| `inflammation_level` | 0.0–1.0 | Inflammaging |

---

## Zimmerman Integration

### Diegetic vs. Supradiegetic (§2.2.3)

Zimmerman distinguishes *diegetic* information — the semantic content of language ("imagine a word minus any letters or sounds") — from *supradiegetic* information — the arbitrary physical form (letter shapes, token boundaries). LLMs have "extremely curtailed access — essentially no access at all" to supradiegetic information.

**Application:** `prompt_templates.py` implements three prompt styles. The *numeric* style presents raw parameter values (supradiegetic-adjacent content the LLM processes poorly). The *diegetic* style embeds parameters in clinical narrative ("How aggressively should we clear damaged mitochondria?"), exploiting the LLM's semantic strengths.

### Flattening (§3.5.3)

Tokenization treats all content identically at the linear-algebraic level — *flattening*. Numbers like ages (20–90) and doses (0.0–1.0) get tokenized destructively, obscuring their different scales. `llm_common.py:detect_flattening()` identifies and corrects the most common failure mode: the LLM normalizing `baseline_age` to 0.0–1.0 instead of 20–90.

### PDS Framework (§4.6.4)

Power, Danger, and Structure are the three most significant axes from ousiometric analysis of word meanings (Dodds et al. 2023). `pds_mapping.py` maps these dimensions to the 6D patient parameter space:

- **Power** (Fool↔Hero) → `metabolic_demand` + inverse `genetic_vulnerability`
- **Danger** (Angel↔Demon) → `inflammation_level` + `genetic_vulnerability`
- **Structure** (Traditionalist↔Adventurer) → `baseline_nad_level`

### POSIWID (§3.5.2)

"The Purpose Of a System Is What It Does" (Beer 1974). `posiwid_audit.py` asks the LLM two questions: (1) "What outcome do you intend?" and (2) "Generate a 12D vector." It then simulates with the generated vector and scores alignment between stated intention and actual outcome.

### TALOT/OTTITT (§4.7.6)

"Things Are Like Other Things" vs. "Only The Thing Is The Thing." The contrastive prompt style in `prompt_templates.py` exploits this by generating opposing protocols — "Dr. Cautious" vs. "Dr. Bold" — forcing the LLM to navigate the identity-difference spectrum explicitly.

---

## Typical Workflow

```
1. Choose patient scenario      CLINICAL_SEEDS in constants.py
         │
2. Generate intervention        prompt_templates.py → Ollama → 12D vector
         │                      (numeric, diegetic, or contrastive style)
         │
3. Simulate                     simulator.py: RK4, 30 years, 7 state vars
         │
4. Analyze                      analytics.py: 4 pillars (energy/damage/dynamics/intervention)
         │
5. Audit alignment              posiwid_audit.py: intention vs actual
         │
6. Iterate                      Adjust prompt style, patient profile, or intervention
```

---

## Module Pages

### Zimmerman-Informed

- **[`prompt_templates`](prompt_templates.md)** — Three prompt styles for LLM-mediated intervention design
- **[`pds_mapping`](pds_mapping.md)** — PDS → patient parameter mapping via character archetypes
- **[`posiwid_audit`](posiwid_audit.md)** — POSIWID alignment scoring (intention vs actual)
- **[`archetype_matchmaker`](archetype_matchmaker.md)** — Which character archetypes work best for which patients?
- **[`sobol_sensitivity`](sobol_sensitivity.md)** — Global sensitivity via Saltelli sampling

### Core Infrastructure

- **[`simulator`](simulator.md)** — RK4 ODE integrator, 7 state variables, tissue types, time-varying schedules
- **[`analytics`](analytics.md)** — 4-pillar metrics (energy, damage, dynamics, intervention)
- **[`llm_common`](llm_common.md)** — LLM query, response parsing, flattening detection, grid snapping
- **[`constants`](constants.md)** — Biological constants, parameter space definitions, clinical seeds
- **[`schemas`](schemas.md)** — Pydantic validation models for LLM output

### TIQM Pipeline & Utilities

- **[`tiqm_experiment`](tiqm_experiment.md)** — Full TIQM pipeline: offer wave → simulation → confirmation wave
- **[`cliff_mapping`](cliff_mapping.md)** — Heteroplasmy cliff characterization, bisection search, 2D heatmaps
- **[`visualize`](visualize.md)** — Matplotlib trajectory plots, cliff curves, intervention comparisons
- **[`generate_patients`](generate_patients.md)** — Patient population generator (100 normal + 82 edge-case)

### Tier 1 — Pure Simulation Experiments

- **[`causal_surgery`](causal_surgery.md)** — Treatment timing / point of no return (~192 sims)
- **[`dark_matter`](dark_matter.md)** — Futile intervention taxonomy (~700 sims)
- **[`protocol_interpolation`](protocol_interpolation.md)** — Fitness landscape between protocols (~1325 sims)

### Tier 2 — LLM Seed Experiments

- **[`oeis_seed_experiment`](oeis_seed_experiment.md)** — OEIS sequences → intervention vectors (~396 trials)
- **[`character_seed_experiment`](character_seed_experiment.md)** — Fictional characters → protocols (~8000 trials)

### Tier 3 — LLM Meta-Analysis

- **[`fisher_metric`](fisher_metric.md)** — LLM output variance / clinical certainty (400 queries)
- **[`clinical_consensus`](clinical_consensus.md)** — Multi-model agreement (40 queries + 40 sims)

### Tier 4 — Synthesis

- **[`perturbation_probing`](perturbation_probing.md)** — Intervention fragility mapping (~1250 sims)
- **[`categorical_structure`](categorical_structure.md)** — Functor validation: Sem → Vec → Beh
- **[`llm_seeded_evolution`](llm_seeded_evolution.md)** — Hill-climbing from LLM vs random seeds (~4000 sims)

### Tier 5 — Discovery Tools

- **[`interaction_mapper`](interaction_mapper.md)** — D4: Synergy/antagonism between intervention pairs (~2160 sims)
- **[`reachable_set`](reachable_set.md)** — D1: Achievable outcome space + Pareto frontiers (~2400 sims)
- **[`competing_evaluators`](competing_evaluators.md)** — D5: Multi-criteria robust protocol search (~1000 sims)
- **[`temporal_optimizer`](temporal_optimizer.md)** — D2: Optimal intervention timelines via ES (~3000 sims)
- **[`multi_tissue_sim`](multi_tissue_sim.md)** — D3: Coupled brain + muscle + cardiac simulation (~30 sims)

---

## References

- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. Graduate College Dissertations and Theses, 2082.
- Zimmerman, J.W., Hudon, D., Cramer, K., St-Onge, J., Fudolig, M., Trujillo, M.Z., Danforth, C.M., and Dodds, P.S. (2024). "A blind spot for large language models: Supradiegetic linguistic information." *Plutonics*, 17, 107-156.
- Dodds, P.S., Alshaabi, T., Fudolig, M.I., Zimmerman, J.W., et al. (2023). "Ousiometrics and telegnomics: The essence of meaning conforms to a two-dimensional powerful-weak and dangerous-safe framework." *arXiv*.
- Cramer, John G. (forthcoming from Springer Verlag in 2026). *How to Live Much Longer*.
- Saltelli, A. (2002). "Making best use of model evaluations to compute sensitivity indices." *Computer Physics Communications*, 145(2), 280-297.
- Jansen, M.J.W. (1999). "Analysis of variance designs for model output." *Computer Physics Communications*, 117(1-2), 35-43.
- Beer, Stafford (1974). "Designing Freedom." CBC Massey Lectures.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
