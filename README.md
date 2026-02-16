# How to Live Much Longer: The Mitochondrial DNA Connection

A computational simulation of mitochondrial aging dynamics and intervention strategies, based on John G. Cramer's book.

> **Cramer, J.G. (2025). *How to Live Much Longer: The Mitochondrial DNA Connection*. ISBN 979-8-9928220-0-4.**

## Overview

This project adapts the TIQM (Transactional Interpretation of Quantum Mechanics) pipeline from the parent [Evolutionary-Robotics](https://github.com/gardenofcomputation/Evolutionary-Robotics) project for cellular energetics. Instead of LLM → physics simulation → VLM scoring for robot locomotion, we use LLM → mitochondrial ODE simulation → VLM scoring for intervention protocol design.

**Core thesis (Cramer 2025):** Aging is a cellular energy crisis caused by progressive mitochondrial DNA damage. When the fraction of damaged mtDNA (heteroplasmy) exceeds ~70% — the "heteroplasmy cliff" — ATP production collapses nonlinearly, triggering a cascade of cellular dysfunction.

## TIQM Mapping

| TIQM Concept | Robotics Project | This Project |
|---|---|---|
| Offer wave | LLM generates 12D weight+physics vector | LLM generates 12D intervention+patient vector |
| Simulation | PyBullet 3-link robot locomotion | RK4 ODE of 7 mitochondrial state variables |
| Confirmation wave | VLM evaluates locomotion behavior | VLM evaluates cellular trajectory |
| Resonance | Semantic match to character/sequence seed | Clinical match to patient scenario |
| Parameter space | 6 weights + 6 physics params | 6 intervention + 6 patient params |

## 12-Dimensional Parameter Space

### Intervention Parameters (6D)

| Parameter | Range | Grid | Effect |
|---|---|---|---|
| `rapamycin_dose` | 0.0–1.0 | [0, 0.1, 0.25, 0.5, 0.75, 1.0] | mTOR inhibition → enhanced mitophagy |
| `nad_supplement` | 0.0–1.0 | [0, 0.1, 0.25, 0.5, 0.75, 1.0] | NAD+ precursor restoration |
| `senolytic_dose` | 0.0–1.0 | [0, 0.1, 0.25, 0.5, 0.75, 1.0] | Senescent cell clearance |
| `yamanaka_intensity` | 0.0–1.0 | [0, 0.1, 0.25, 0.5, 0.75, 1.0] | Partial reprogramming (costs 3–5 MU!) |
| `transplant_rate` | 0.0–1.0 | [0, 0.1, 0.25, 0.5, 0.75, 1.0] | Healthy mtDNA infusion via mitlets |
| `exercise_level` | 0.0–1.0 | [0, 0.1, 0.25, 0.5, 0.75, 1.0] | Hormetic adaptation |

### Patient Parameters (6D)

| Parameter | Range | Description |
|---|---|---|
| `baseline_age` | 20–90 years | Starting age |
| `baseline_heteroplasmy` | 0.0–0.95 | Fraction of damaged mtDNA |
| `baseline_nad_level` | 0.2–1.0 | NAD+ availability (declines with age) |
| `genetic_vulnerability` | 0.5–2.0 | mtDNA damage susceptibility |
| `metabolic_demand` | 0.5–2.0 | Tissue energy requirement |
| `inflammation_level` | 0.0–1.0 | Chronic inflammation |

## ODE Model: 7 State Variables

The simulator integrates 7 coupled ordinary differential equations using a 4th-order Runge-Kutta method:

| Variable | Symbol | Description |
|---|---|---|
| Healthy mtDNA | N_healthy | Normalized healthy copy count |
| Damaged mtDNA | N_damaged | Normalized damaged copy count |
| ATP production | ATP | Energy output (MU/day) |
| Reactive oxygen species | ROS | Oxidative stress level |
| NAD+ availability | NAD | Cofactor for energy production |
| Senescent fraction | Sen | Fraction of senescent cells |
| Membrane potential | ΔΨ | Mitochondrial membrane potential |

### Key Dynamics

- **ROS-damage vicious cycle:** Damaged mitochondria → excess ROS → more mtDNA damage (Ch. II.H pp.14-15, Appendix 2 pp.152-154)
- **Heteroplasmy cliff:** Steep sigmoid at ~70% — ATP collapses above this threshold (from mitochondrial genetics literature; Cramer's MitoClock uses a different ~25% metric, Ch. V.K p.66)
- **Age-dependent deletion doubling:** 11.8 years (young) → 3.06 years (old), from Va23 nanopore sequencing data (Appendix 2, p.155, Fig. 23). Transition at age 65 (corrected from earlier value of 40 per Cramer's email review).
- **Replication advantage:** Shorter damaged mtDNA replicates faster — book says "at least 21%" for deletions >3kbp (Appendix 2, pp.154-155); simulation conservatively uses 5%
- **Energy units:** 1 MU (Metabolic Unit) = 10^8 ATP energy releases. Normal cell = ~1 MU/day (Ch. VIII.A, Table 3, p.100)
- **Interventions:** Rapamycin → mTOR → mitophagy (Ch. VI.A.1 pp.71-72), NAD+ via NMN/NR gated by CD38 degradation (Ch. VI.A.3 pp.72-73), senolytics D+Q+F (Ch. VII.A.2 p.91), Yamanaka reprogramming at 3-5 MU cost (Ch. VIII.A Table 3 p.100, Ch. VII.B p.95), transplant as primary rejuvenation with competitive displacement of damaged copies (Ch. VIII.G pp.104-107), exercise (hormesis)

### ODE Corrections Applied

**Falsifier audit (2026-02-15)** found 4 critical bugs in the original ODE equations. Full report: [`artifacts/falsifier_report_2026-02-15.md`](artifacts/falsifier_report_2026-02-15.md).

| Fix | Issue | Resolution |
|---|---|---|
| C1 | Cliff was cosmetic — didn't feed back into replication/apoptosis | ATP collapse now halts replication |
| C2 | mtDNA copy number unbounded | Homeostatic regulation toward total ~ 1.0 |
| C3 | NAD supplementation boosted damaged mito equally | NAD now selectively benefits healthy mito (quality control) |
| C4 | No bistability past cliff | Damaged replication advantage creates irreversible collapse |
| M1 | Yamanaka worked without energy | Yamanaka gated by ATP availability |

**Cramer email corrections (2026-02-15):**

| Fix | Change | Source |
|---|---|---|
| C7 | NAD+ boost gated by CD38 survival factor (40% at min dose, 100% at max with apigenin) | Ch. VI.A.3 p.73 |
| C8 | Transplant rate doubled (0.15→0.30), competitive displacement added, headroom raised (1.2→1.5) | Ch. VIII.G pp.104-107 |
| C9 | AGE_TRANSITION restored from 40 to 65 | Appendix 2 p.155 |

## Model Predictions: Slowing and Reversing Aging

Full report: [`artifacts/slowing_vs_reversing_aging_2026-02-15.md`](artifacts/slowing_vs_reversing_aging_2026-02-15.md)

### Slowing Aging

The model identifies **rapamycin** (mTOR inhibition → enhanced mitophagy) as the single most effective intervention for slowing the rate of mitochondrial DNA damage accumulation. At moderate dose, it cuts the 30-year heteroplasmy increase roughly in half. **NAD+ supplementation** (NMN/NR) is the best single intervention for maintaining ATP production — the only monotherapy that preserves or increases energy output over 30 years. Senolytics, exercise, and transplant have minimal standalone effect on heteroplasmy but contribute meaningfully in combination.

### Reversing Aging

**Reversal — actual reduction of heteroplasmy below starting values — requires combination therapy.** The minimum viable reversal cocktail is:

| Intervention | Dose | Role |
|---|---|---|
| Rapamycin | 0.5 | Clears damaged mtDNA via enhanced mitophagy |
| NAD+ (NMN/NR) | 0.75 | Restores cofactor, selectively supports healthy mito |
| Senolytics (D+Q+F) | 0.5 | Frees energy budget from senescent cells |
| Exercise | 0.5 | Biogenesis + antioxidant upregulation |

Adding **mitochondrial transplant** (0.5) significantly accelerates reversal. Adding **Yamanaka reprogramming** (0.25) further accelerates it, but only if the cocktail maintains sufficient ATP to power the 3-5 MU reprogramming cost.

### Cliff Rescue

The model shows **no absolute point of no return**. Even starting at 90% heteroplasmy (deep past the 70% cliff, ATP collapsed to 0.04 MU/day), aggressive combination therapy can restore heteroplasmy to ~0.15 and ATP to ~0.85 MU/day over 30 years. The cocktail+transplant+rapamycin+NAD protocol is the most efficient cliff rescue strategy.

### The Synergy Principle

Each intervention attacks a different node in the ROS-damage vicious cycle: rapamycin *removes* damaged copies, NAD+ *supports* healthy ones, senolytics *free energy*, exercise *creates* new copies and reduces ROS, transplant *adds* copies from outside, and Yamanaka *converts* damaged to healthy. This confirms Cramer's core thesis: aging is a cellular energy crisis requiring multi-angle intervention.

## 4-Pillar Analytics

| Pillar | Metrics |
|---|---|
| **Energy** | ATP trajectory, min ATP, time-to-crisis, reserve ratio, slope, CV |
| **Damage** | Heteroplasmy trajectory, time-to-cliff, acceleration, cliff distance |
| **Dynamics** | ROS FFT frequency, membrane potential CV, NAD slope, ROS-het correlation |
| **Intervention** | Energy cost, het benefit, ATP benefit, benefit-cost ratio, crisis delay |

## Biological Constants (Cramer 2025 Citations)

All biological constants in `constants.py` have been traced to specific chapters and pages in Cramer (2025). Key values:

| Constant | Value | Cramer 2025 Source |
|---|---|---|
| Baseline ATP | 1.0 MU/day | Ch. VIII.A, Table 3, p.100 (1 MU = 10^8 ATP releases) |
| Yamanaka energy cost | 3-5 MU | Ch. VIII.A, Table 3, p.100; Ch. VII.B p.95: "3 to 10 times" (Ci24, Fo18) |
| Deletion doubling (young) | 11.81 yr | Appendix 2, p.155, Fig. 23 (Va23: Vandiver et al., *Aging Cell* 22(6), 2023) |
| Deletion doubling (old) | 3.06 yr | Appendix 2, p.155, Fig. 23 (transition at age 65, corrected from 40) |
| Replication advantage | 1.05x (sim) | Appendix 2, pp.154-155: "at least 21% faster" for >3kbp deletions |
| NAD+ decline | 0.01/yr | Ch. VI.A.3, pp.72-73 (Ca16: Camacho-Pereira et al., 2016) |
| CD38 base survival | 0.4 | Ch. VI.A.3 p.73: CD38 destroys NMN/NR; 40% survives at min dose |
| CD38 suppression gain | 0.6 | Apigenin suppresses CD38, raising survival to 100% at max dose |
| Transplant addition rate | 0.30 | Ch. VIII.G pp.104-107: primary rejuvenation (doubled from original 0.15) |
| Transplant displacement | 0.12 | Competitive displacement of damaged copies by healthy transplants |
| Mitophagy (PINK1/Parkin) | 0.02/yr | Ch. VI.B, p.75: mechanism described, rate is simulation parameter |
| Senescence rate | 0.005/yr | Ch. VII.A, pp.89-92; Ch. VIII.F, p.103: ~2x energy, SASP |
| Membrane potential (ΔΨ) | 1.0 (norm) | Ch. IV, pp.46-47: proton gradient; Ch. VI.B, p.75: low ΔΨ → PINK1 |
| Heteroplasmy cliff | 0.70 | Literature (Rossignol 2003); book uses different MitoClock metric (~25%, Ch. V.K p.66) |

See the citation key at the top of `constants.py` for full details on each constant's provenance.

## Files

### Core Pipeline

| File | Description |
|---|---|
| `constants.py` | Central configuration, 12D parameter space, biological constants, type aliases |
| `schemas.py` | Pydantic validation models (`InterventionVector`, `PatientProfile`, `FullProtocol`) |
| `simulator.py` | RK4 ODE integrator for 7 state variables; supports tissue types, stochastic mode, `InterventionSchedule` |
| `analytics.py` | 4-pillar health analytics computation |
| `llm_common.py` | Shared LLM utilities (Ollama query, response parsing, grid snapping, flattening detection) |
| `prompt_templates.py` | Prompt styles: numeric, diegetic, contrastive (used by `tiqm_experiment.py --style`) |
| `cliff_mapping.py` | Heteroplasmy threshold mapping and cliff characterization |
| `visualize.py` | Matplotlib plots (Agg backend) |
| `tiqm_experiment.py` | Full TIQM pipeline with Ollama LLM integration |
| `protocol_mtdna_synthesis.py` | 9-step mtDNA synthesis and transplant protocol |

### Research Campaign Scripts

Adapted from the parent Evolutionary-Robotics project's research campaign infrastructure. Each script is self-contained and writes JSON artifacts to `artifacts/`.

**Tier 1 — Pure Simulation (no LLM needed)**

| File | Description | Scale |
|---|---|---|
| `causal_surgery.py` | Mid-simulation intervention switching; finds point of no return for starting treatment | ~192 sims, ~30s |
| `dark_matter.py` | Random 12D sampling; classifies futile/paradoxical interventions | ~700 sims, ~2 min |
| `protocol_interpolation.py` | Linear interpolation between champion protocols; finds super-protocols at midpoints | ~1325 sims, ~3 min |

**Tier 2 — LLM Seed Experiments (requires Ollama)**

| File | Description | Scale |
|---|---|---|
| `oeis_seed_experiment.py` | OEIS integer sequences as semantic seeds for intervention design | ~99 seqs × 4 models, ~2 hrs |
| `character_seed_experiment.py` | 2000 fictional characters as seeds for patient+protocol generation | ~8000 trials, ~5 hrs |

**Tier 3 — LLM Meta-Analysis (requires Ollama)**

| File | Description | Scale |
|---|---|---|
| `fisher_metric.py` | Measures LLM output variance to quantify clinical certainty per scenario | 400 queries, ~30-60 min |
| `clinical_consensus.py` | Multi-model agreement analysis across clinical scenarios | 40 queries + 40 sims, ~15-20 min |

**Tier 4 — Synthesis (some need prior experiment data)**

| File | Description | Scale |
|---|---|---|
| `perturbation_probing.py` | Perturbs each of 12 params ±1 grid step to map intervention fragility | ~1250 sims, ~5 min |
| `categorical_structure.py` | Formal functor validation: Sem→Vec→Beh distance correlations | Pure computation, ~5 sec |
| `llm_seeded_evolution.py` | Hill-climbing from LLM-generated vs random starting points | ~4000 sims, ~10 min |

**Zimmerman-Informed Experiments (requires Ollama for some)**

Based on Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont.

| File | Description | Scale |
|---|---|---|
| `prompt_templates.py` | Prompt styles: numeric (original), diegetic (narrative), contrastive (Dr. Cautious vs Dr. Bold) | Library (no sims) |
| `sobol_sensitivity.py` | Saltelli sampling + Sobol first/total-order indices; captures parameter interactions | ~6656 sims, ~3 min |
| `pds_mapping.py` | Maps Zimmerman's Power/Danger/Structure character dimensions to 6D patient space | No sims |
| `posiwid_audit.py` | POSIWID alignment: gap between LLM-stated intentions and actual simulation outcomes | Requires Ollama, ~15-20 min |
| `archetype_matchmaker.py` | Identifies which character archetypes produce best protocols per patient type | Needs character experiment data |

**Tier 5 — Discovery Tools (no LLM needed, exploit simulation for novel biology)**

| File | Description | Scale |
|---|---|---|
| `interaction_mapper.py` | 2D grid sweeps of all 15 intervention pairs; measures synergy/antagonism per patient type | ~2160 sims, ~3 min |
| `reachable_set.py` | Latin hypercube sampling of 6D intervention space; Pareto frontier, reachable region, minimum-intervention paths | ~2400 sims, ~5 min |
| `competing_evaluators.py` | 4 evaluator functions score ~500 candidates; finds "transaction" protocols robust across all criteria | ~1000 sims, ~2 min |
| `temporal_optimizer.py` | (1+lambda) ES over phased intervention schedules; optimal timelines vs constant dosing | ~3000 sims, ~7 min |
| `multi_tissue_sim.py` | Coupled brain+muscle+cardiac simulation with shared NAD, systemic inflammation, cardiac blood flow | ~30 sims, ~2 min |

## Questions from John G. Cramer (2026-02-16)

Full Q&A: [`artifacts/cramer_questions_2026-02-16.md`](artifacts/cramer_questions_2026-02-16.md)

John Cramer reviewed the simulation and raised five questions about its scope and limitations: (1) individual vs. population modeling, (2) tissue-specific cliff thresholds, (3) mutation type granularity, (4) whether the simulation verifies the mtDNA energy hypothesis, and (5) testable predictions. The answers document what the simulator currently does, what it doesn't, and six specific predictions that could be experimentally tested.

## Landscape Characterization

Full analysis: [`artifacts/landscape_characterization.md`](artifacts/landscape_characterization.md)
Cross-project comparison: [`artifacts/cross_project_gnarliness_comparison.md`](artifacts/cross_project_gnarliness_comparison.md)
Project state and next steps: [`artifacts/project_state_2026-02-15.md`](artifacts/project_state_2026-02-15.md) — infrastructure-rich, data-poor; ~27 min of compute would populate the atlas

The 12D parameter space has been partially characterized (~400 simulations). Key findings:

- **Patient-stratified topology**: Healthy patients (het < 0.3) can't fail; near-cliff patients (het > 0.6) mostly fail (68.8%). The landscape is a patchwork of computational classes, not uniformly rough.
- **Intervention hierarchy**: Transplant (0.76) >> rapamycin (0.63) >> NAD (0.38) >> exercise (0.08) >> senolytics (0.04). Strongly anisotropic — 17.4x potency range.
- **Moderate roughness**: CV(ATP) = 0.503. For comparison, the ER project's landscape is fractal everywhere (sign flip rate 0.58-0.72). Hill-climbing works here; it fails there.
- **No timing critical point**: For near-cliff patients, switch-time has r=0.007 with outcome — the bistable attractor has already captured the trajectory.
- **Robustness paradox**: Successful protocols are 7.4x more robust than failing ones.

## Testing

```bash
# Full test suite (85 tests across 4 modules)
pytest tests/ -v

# Standalone self-tests
python simulator.py    # 10 ODE scenarios
python analytics.py    # 4-pillar analytics
```

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mito-aging

# Run the simulator standalone test
python simulator.py

# Run analytics test
python analytics.py

# Run TIQM experiments (requires Ollama running locally)
ollama serve &
python tiqm_experiment.py

# Quick single-scenario test
python tiqm_experiment.py --single

# Zimmerman prompt styles
python tiqm_experiment.py --style diegetic
python tiqm_experiment.py --contrastive

# Print the mtDNA synthesis protocol
python protocol_mtdna_synthesis.py

# Research campaigns (Tier 1 — no Ollama needed)
python causal_surgery.py
python dark_matter.py
python protocol_interpolation.py

# Research campaigns (Tier 2-4 — requires Ollama)
python oeis_seed_experiment.py
python character_seed_experiment.py
python fisher_metric.py
python clinical_consensus.py
python perturbation_probing.py
python categorical_structure.py    # needs seed experiment data
python llm_seeded_evolution.py     # optionally loads seed experiment data

# Zimmerman-informed experiments
python sobol_sensitivity.py        # ~3 min, no Ollama
python pds_mapping.py              # no Ollama
python posiwid_audit.py            # requires Ollama, ~15-20 min
python archetype_matchmaker.py     # needs character experiment data

# Discovery tools (Tier 5 — no Ollama needed)
python interaction_mapper.py       # ~3 min
python reachable_set.py            # ~5 min
python competing_evaluators.py     # ~2 min
python temporal_optimizer.py       # ~7 min
python multi_tissue_sim.py         # ~2 min
```

## Requirements

- Python 3.11+
- NumPy 1.26.4
- Matplotlib (with Agg backend)
- Ollama (local, for TIQM experiments only)

## Parent Project

This project extends the TIQM pipeline from [Evolutionary-Robotics](https://github.com/gardenofcomputation/Evolutionary-Robotics), which uses LLM-driven weight generation and VLM-scored evaluation for emergent robot locomotion in PyBullet.

## References

- Cramer, J.G. (2025). *How to Live Much Longer: The Mitochondrial DNA Connection*. ISBN 979-8-9928220-0-4.
- Cramer, J.G. (1986). "The Transactional Interpretation of Quantum Mechanics." *Reviews of Modern Physics*, 58(3), 647–687.
- Camacho-Pereira, J. et al. (2016). "CD38 dictates age-related NAD decline and mitochondrial dysfunction through an SIRT3-dependent mechanism." *Cell Metabolism*, 23(6), 1127–1139.
- McCully, J.D. et al. (2009). "Injection of isolated mitochondria during early reperfusion for cardioprotection." *American Journal of Physiology*, 296(1), H94–H105.
- Rossignol, R. et al. (2003). "Mitochondrial threshold effects." *Biochemical Journal*, 370(3), 751–762.
- Vandiver, A.R. et al. (2023). "Nanopore sequencing identifies a higher frequency and expanded spectrum of mitochondrial DNA deletion mutations in human aging." *Aging Cell*, 22(6). doi:10.1111/acel.13842.
- Wallace, D.C. (2005). "A mitochondrial paradigm of metabolic and degenerative diseases, aging, and cancer." *Genetics*, 163(4), 1215–1241.
- Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction in Language, as Implemented in Humans and Large Language Models (LLMs)." PhD dissertation, University of Vermont. [PDS mapping, prompt engineering, POSIWID audit]
