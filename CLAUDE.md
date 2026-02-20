# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Action Items

- **REPORT TO JOHN CRAMER:** The falsifier agent found 4 critical bugs in the ODE equations (2026-02-15). Full report: `artifacts/falsifier_report_2026-02-15.md`. Key issues: cliff was cosmetic not dynamical, mtDNA copy number unbounded, NAD supplementation inverted therapeutic sign, universal attractor prevented cliff crossing. Fixes were applied same day — Cramer should review the corrected dynamics for biological plausibility.
- **CRAMER CORRECTIONS APPLIED (2026-02-15):** Per John Cramer's email review:
  - **C7: CD38 degrades NMN/NR** — NAD+ boost now gated by CD38 survival factor (p.73). Low-dose supplementation is largely futile (CD38 destroys most precursor). High doses imply CD38 suppression via apigenin, improving delivery. Coefficient reduced from 0.35 → 0.25 * cd38_survival.
  - **C8: Transplant is primary rejuvenation** — Addition rate doubled (0.15 → 0.30), headroom raised (1.2 → 1.5), and competitive displacement of damaged copies added (0.12 * n_d). Transplant now clearly outperforms NAD supplementation and can rescue near-cliff patients.
- **CRAMER CORRECTION APPLIED (2026-02-15):** Per John Cramer's second email:
  - **C9: AGE_TRANSITION restored to 65** — The mtDNA deletion doubling time transition was incorrectly set to age 40. Book data (Appendix 2 p.155, Va23) places it at age 65. Corrected in constants.py and simulator.py.
- **CRAMER CORRECTION APPLIED (2026-02-15):** Per John Cramer's third email:
  - **C10: AGE_TRANSITION coupled to ATP and mitophagy** — The deletion-damage slope change should not be a fixed age. It is now dynamically coupled to ATP energy level and mitophagy efficiency. High ATP + effective mitophagy → transition shifts later (up to +10 years). Low ATP + poor mitophagy → transition shifts earlier (up to -15 years). Hard cutoff replaced with smooth sigmoid blend (width ~5 years).
  - **C10 calibration (2026-02-19, recalibrated):** `NATURAL_HEALTH_REF = 0.91` recalibrated after NAD coefficient reduction (0.4→0.2) and cliff recalibration (0.70→0.50). Natural aging now yields ATP ≈ 0.908 at age 65; normalizing by 0.91 gives residual shift < 0.01 years (<1 day). See `simulator.py:_deletion_rate()`.
- **CRAMER CORRECTION APPLIED (2026-02-17):** Per John Cramer's email:
  - **C11: Split mutation types** — ROS is NOT the main mtDNA mutation driver (1980s Free Radical Theory is outdated). Two distinct mutation types with different dynamics:
    - **Point mutations** (N_point, state[7]): Linear growth, no replication advantage. Sources: ~67% Pol γ errors + ~33% ROS-induced transitions. Functionally mild.
    - **Deletion mutations** (N_deletion, state[1]): Exponential growth, size-dependent replication advantage (1.10x, book says ≥1.21). These drive the heteroplasmy cliff. Source: Pol γ slippage, NOT ROS.
  - State vector expanded from 7D to 8D. Cliff factor uses deletion heteroplasmy only. ROS→damage coupling weakened to ~33% of previous (point mutations only).
  - Reference: Appendix 2 pp.152-155, Va23 (Vandiver et al. 2023).
- **NAD AUDIT & CLIFF RECALIBRATION (2026-02-19):** Hype literature guard:
  - **NAD coefficients reduced (0.4→0.2):** Two non-Cramer modeling assumptions (ATP production, antioxidant defense) inflated NAD+ efficacy. Reduced from 0.4 to 0.2 each. NAD gain at max dose: +0.100 → +0.066 ATP (35% reduction). See `artifacts/finding_nad_audit_hype_guard_2026-02-19.md`.
  - **HETEROPLASMY_CLIFF recalibrated (0.70→0.50):** C11 split broke cliff dynamics — deletion het maxed at ~0.57, never reaching old 0.70 threshold. Lowered to 0.50 (deletion-only equivalent of literature's 0.70 total het). Cliff now activates properly.
  - **Bistability restored:** ATP-gated mitophagy (autophagy requires energy) + transplant het penalty (hostile environment impairs engraftment). Model now has point of no return at het~0.93-0.95.
  - **REVIEW WITH CRAMER:** Is HETEROPLASMY_CLIFF=0.50 the right deletion threshold? Are NAD coefficients at 0.2 appropriate? Is point of no return at het~0.93 consistent with clinical expectation?
- **ASK DODDS & DANFORTH (LEMURS data):** The precision medicine expansion uses three sleep coefficients attributed to "UVM LEMURS" but not derived from any LEMURS publication. Current values are literature-approximated but could be grounded in actual LEMURS Oura ring data. Specific requests:
  1. **Sleep quality → physiological recovery rate:** LEMURS tracks HRV recovery overnight. What fraction of next-day HRV recovery is lost per unit of sleep quality reduction? This maps to `SLEEP_DISRUPTION_IMPACT` (currently 0.5, literature-approximated). We use this to scale mitophagy/repair efficacy.
  2. **Alcohol → sleep quality dose-response:** LEMURS surveys likely capture both alcohol consumption and Oura sleep scores. What is the quantitative relationship between drinks/day and Oura sleep score reduction? This maps to `ALCOHOL_SLEEP_DISRUPTION` (currently 0.4).
  3. **Sleep disturbance → stress/inflammation proxy:** LEMURS has both Oura data and stress/wellness surveys. Is there a quantitative mapping from Oura sleep score → next-day perceived stress? This maps to our `(1.0 - sleep_quality) * 0.05` inflammation effect.
  - Note: LEMURS is NOT about grief — grief coefficients come from O'Connor (2022/2025) bereavement studies via the grief-simulator project. These are separate research threads that were incorrectly conflated in constants.py (now fixed).
  - See `artifacts/finding_sleep_coefficient_audit_2026-02-19.md` for full audit.
  - Relevant LEMURS papers: Fudolig et al. 2024 (Digital Biomarkers, sleep HR dynamics), Bloomfield et al. 2024 (PLOS Digital Health, stress prediction), Fudolig et al. 2025 (npj Complexity, collective sleep patterns).

## Project Overview

Computational simulation of mitochondrial aging dynamics and intervention strategies, based on John G. Cramer's book *How to Live Much Longer: The Mitochondrial DNA Connection* (2025, ISBN 979-8-9928220-0-4).

Adapts the TIQM (Transactional Interpretation of Quantum Mechanics) pipeline from the parent [Evolutionary-Robotics](../Evolutionary-Robotics/) project. Instead of LLM → physics simulation → VLM scoring for robot locomotion, we use LLM → mitochondrial ODE simulation → VLM scoring for intervention protocol design.

**Core thesis (Cramer 2025):** Aging is a cellular energy crisis caused by progressive mitochondrial DNA damage. When the fraction of deletion-bearing mtDNA exceeds ~50% (deletion heteroplasmy cliff, recalibrated from 70% total het for C11 mutation split) — ATP production collapses nonlinearly.

## Environment Setup

```bash
conda activate mito-aging
# Prefix: see environment.yml (Python 3.11, numpy 1.26.4, matplotlib)
# Minimal dependencies — no pybullet, no fastapi, no pillow

# LLM-mediated experiments require Ollama running locally (http://localhost:11434)
# Scripts that need Ollama: tiqm_experiment.py, oeis_seed_experiment.py,
# character_seed_experiment.py, fisher_metric.py, clinical_consensus.py,
# llm_seeded_evolution.py (optionally)
```

### Local LLMs via Ollama

Same Ollama instance as the parent ER project. Available models:

| Model | Role |
|---|---|
| `qwen3-coder:30b` | Offer wave (primary intervention designer) |
| `llama3.1:latest` | Confirmation wave (trajectory evaluator) |
| `deepseek-r1:8b` | Alternative offer model (reasoning, emits `<think>` tags) |
| `gpt-oss:20b` | Alternative offer model |

Different models are used for offer vs confirmation waves to prevent self-confirmation bias (Cramer's TIQM principle).

## Commands

```bash
# Standalone tests (no Ollama needed)
python simulator.py       # ODE integrator test with 10 scenarios (incl. tissue types, stochastic, phased schedule, Cramer corrections)
python analytics.py       # 4-pillar analytics test
pytest tests/ -v          # Full pytest suite (~439 tests: simulator, analytics, LLM parsing, schemas, zimmerman bridge, resilience, kcramer bridge, grief bridge, expansion modules, scenarios, protocol dictionary pipeline)
python cliff_mapping.py   # Heteroplasmy cliff mapping (~2 min)
python visualize.py       # Generate all plots to output/
python generate_patients.py           # Normal population (100 patients, ~30s)
python generate_patients.py --edge    # Edge-case population (82 patients, ~25s)
python generate_patients.py --both    # Both populations

# TIQM experiments (requires Ollama)
python tiqm_experiment.py                     # All 10 clinical scenarios
python tiqm_experiment.py --single            # Quick single test
python tiqm_experiment.py --style diegetic    # Zimmerman-informed narrative prompts
python tiqm_experiment.py --contrastive       # Dr. Cautious vs Dr. Bold protocols

# Tier 1 — Pure simulation experiments (no Ollama needed)
python causal_surgery.py          # Treatment timing / point of no return (~30s, ~192 sims)
python dark_matter.py             # Futile intervention taxonomy (~2 min, ~700 sims)
python protocol_interpolation.py  # Fitness landscape between protocols (~3 min, ~1325 sims)

# Tier 2 — LLM seed experiments (requires Ollama)
python oeis_seed_experiment.py       # OEIS sequences → interventions (~2 hrs)
python character_seed_experiment.py  # Fictional characters → protocols (~5 hrs)

# Tier 3 — LLM meta-analysis (requires Ollama)
python fisher_metric.py       # Treatment certainty measurement (~30-60 min, 400 queries)
python clinical_consensus.py  # Multi-model agreement (~15-20 min, 40 queries + 40 sims)

# Tier 4 — Synthesis (some require prior experiment data)
python perturbation_probing.py   # Intervention fragility mapping (~5 min, ~1250 sims)
python categorical_structure.py  # Functor validation (needs seed data, ~5 sec)
python llm_seeded_evolution.py   # Hill-climb from LLM seeds (~10 min, ~4000 sims)
python llm_seeded_evolution.py --narrative  # Narrative feedback evolution (requires Ollama)

# Resilience analysis (no Ollama needed)
python resilience_viz.py                                      # Default radiation shock at year 10
python resilience_viz.py --disturbance chemo --magnitude 0.8  # Chemotherapy burst
python resilience_viz.py --disturbance toxin                  # Toxin exposure
python resilience_viz.py --disturbance inflammation            # Inflammation burst

# Grief→Mito integration (requires ~/grief-simulator)
python grief_mito_viz.py                                      # Generate all grief→mito plots
python -m pytest tests/test_grief_bridge.py -v                # Grief bridge tests (41 tests)
python -c "from grief_mito_simulator import GriefMitoSimulator; s = GriefMitoSimulator(); print(s.run({}))"

# Zimmerman-informed experiments (2026-02-15 upgrade)
python sobol_sensitivity.py      # Sobol global sensitivity analysis (~3 min, ~6656 sims)
python pds_mapping.py            # PDS→patient parameter mapping (Zimmerman §4.6.4)
python posiwid_audit.py          # POSIWID alignment audit (requires Ollama, ~15-20 min)
python archetype_matchmaker.py   # Archetype→protocol matchmaker (needs character data)
python ten_types_audit.py        # Ten Types platform audit (deterministic, no Ollama)

# Tier 5 — Discovery Tools (no Ollama needed, standalone)
python interaction_mapper.py     # D4: Synergy/antagonism between intervention pairs (~3 min, ~2160 sims)
python reachable_set.py          # D1: Reachable outcome space + Pareto frontiers (~5 min, ~2400 sims)
python competing_evaluators.py   # D5: Multi-criteria robust protocols (~2 min, ~1000 sims)
python temporal_optimizer.py     # D2: Optimal intervention timelines via ES (~7 min, ~3000 sims)
python multi_tissue_sim.py       # D3: Coupled brain+muscle+cardiac simulation (~2 min, ~30 sims)

# Tier 5 — Literature Search (requires internet, Ollama optional)
python lit_spider.py                                     # Full run, all params (~20 min with LLM)
python lit_spider.py --params heteroplasmy_cliff,base_replication_rate  # Subset
python lit_spider.py --no-llm                            # Keyword-only, no Ollama (~1 min)
python lit_spider.py --max-papers 5                      # Fewer papers per param

# Tier 5 — EA-Toolkit Optimization (no Ollama needed, requires ~/ea-toolkit)
python ea_optimizer.py                             # CMA-ES, budget 500 (~3 min)
python ea_optimizer.py --algo de --budget 1000     # Differential Evolution
python ea_optimizer.py --compare --budget 300      # Head-to-head 5-algo comparison (~8 min)
python ea_optimizer.py --landscape --budget 200    # Landscape analysis (~1 min)
python ea_optimizer.py --patient near_cliff_80     # Different patient profile
python ea_optimizer.py --metric het                # Optimize heteroplasmy reduction

# Tier 6 — Zimmerman Toolkit Integration (no Ollama needed, requires ~/zimmerman-toolkit)
python zimmerman_analysis.py                           # All 14 tools (~15-25 min)
python zimmerman_analysis.py --tools sobol             # Sobol only (~1 min at n_base=32)
python zimmerman_analysis.py --tools sobol --n-base 256  # Full Sobol (~10 min, 6656 sims)
python zimmerman_analysis.py --tools falsifier         # Falsification (~30s, 200 tests)
python zimmerman_analysis.py --tools contrastive       # Contrastive pairs (~2 min)
python zimmerman_analysis.py --tools sobol,falsifier,posiwid  # Multiple tools
python zimmerman_analysis.py --patient near_cliff_80   # Intervention-only mode (6D)
python zimmerman_analysis.py --viz                     # Also generate plots
python zimmerman_viz.py                                # Generate plots from saved reports
```

## Architecture

### Dependency Graph

```
constants.py          ← Central config, type aliases (no dependencies)
    ↓
schemas.py            ← Pydantic validation models (imports nothing from project)
    ↓
simulator.py          ← RK4 ODE integrator + InterventionSchedule (imports constants)
    ↓
analytics.py          ← 4-pillar metrics (imports constants, simulator)
    ↓
llm_common.py         ← Shared LLM utilities (imports constants, schemas)
    ↓
cliff_mapping.py      ← Cliff analysis (imports simulator, constants)
visualize.py          ← Matplotlib plots (imports simulator, analytics, cliff_mapping)
tiqm_experiment.py    ← TIQM pipeline (imports everything + Ollama)

# Research campaign scripts (self-contained, import simulator + analytics)
causal_surgery.py          ← Treatment timing (no LLM)
dark_matter.py             ← Futile intervention taxonomy (no LLM)
protocol_interpolation.py  ← Protocol fitness landscape (no LLM)
oeis_seed_experiment.py    ← OEIS → intervention vectors (Ollama)
character_seed_experiment.py ← Characters → protocols (Ollama)
fisher_metric.py           ← LLM output variance (Ollama)
clinical_consensus.py      ← Multi-model agreement (Ollama)
perturbation_probing.py    ← Intervention fragility (no LLM, uses prior data)
categorical_structure.py   ← Functor validation (no LLM, uses prior data)
llm_seeded_evolution.py    ← LLM seeds vs random for optimization (optional LLM)

# Zimmerman-informed additions (2026-02-15 upgrade)
prompt_templates.py        ← Prompt styles: numeric, diegetic, contrastive (imports constants)
sobol_sensitivity.py       ← Saltelli sampling + Sobol indices (imports simulator)
pds_mapping.py             ← PDS→patient mapping (imports constants, analytics)
posiwid_audit.py           ← POSIWID alignment audit (imports llm_common, simulator)
archetype_matchmaker.py    ← Archetype→protocol matching (imports pds_mapping, analytics)
ten_types_audit.py         ← Ten Types whole-platform audit (deterministic evidence scoring)

# Discovery tools (Tier 5, no LLM, imports simulator + analytics/constants)
interaction_mapper.py      ← D4: Synergy/antagonism 2D grid sweeps (imports simulator, constants)
reachable_set.py           ← D1: Reachable outcome space + Pareto (imports simulator, constants)
competing_evaluators.py    ← D5: Multi-criteria protocol search (imports simulator, analytics)
temporal_optimizer.py      ← D2: Temporal schedule optimization (imports simulator, analytics)
multi_tissue_sim.py        ← D3: Coupled multi-tissue simulation (imports simulator.derivatives, constants)

generate_patients.py       ← Patient population generator + evaluator (imports simulator, analytics, constants)

# Precision Medicine Expansion (2026-02-19)
genetics_module.py         ← APOE4/FOXO3/CD38 + sex modifiers (imports constants)
lifestyle_module.py        ← Alcohol, coffee, diet, fasting (imports constants)
supplement_module.py       ← Hill-function dose-response 11 nutraceuticals (imports constants)
parameter_resolver.py      ← 50D→12D modifier chain (imports genetics, lifestyle, supplements)
downstream_chain.py        ← MEF2, HA, SS, CR, amyloid, tau ODEs + memory_index (imports constants)
scenario_definitions.py    ← InterventionProfile, Scenario, A-D scenarios (imports nothing)
scenario_runner.py         ← Pipeline orchestration (imports parameter_resolver, simulator, downstream_chain)
scenario_analysis.py       ← Milestone extraction (imports numpy)
scenario_plot.py           ← Trajectory/milestone/heatmap plots (imports matplotlib)
run_scenario_comparison.py ← CLI 4-scenario comparison (imports all scenario modules)

lit_spider.py              ← PubMed literature search for parameter validation (imports constants, llm_common)

# Protocol Dictionary Pipeline (2026-02-19, ported from rosetta-motion)
protocol_record.py         ← ProtocolRecord dataclass, fingerprinting, VALID_OUTCOME_CLASSES (imports nothing)
protocol_dictionary.py     ← Persistent catalog with query/dedup/summary (imports protocol_record)
protocol_enrichment.py     ← Complexity, clinical signature, prototype grouping (imports protocol_record)
protocol_classifier.py     ← Multi-labeler: rule + analytics-fit classification (imports nothing from project)
protocol_rewrite_rules.py  ← Declarative 3-layer rule engine with audit trail (imports nothing from project)
protocol_pattern_language.py ← DAG validation + topological orchestrator (imports nothing from project)
protocol_review.py         ← JSONL review queue with resolve/provenance (imports protocol_record)
run_protocol_pipeline.py   ← End-to-end pipeline runner + CLI (imports all protocol modules + simulator + analytics)

ea_optimizer.py            ← EA-toolkit integration: 8 algorithms for intervention optimization (imports simulator, analytics, constants, ea_toolkit)

# Zimmerman-toolkit integration (Tier 6, no LLM, requires ~/zimmerman-toolkit)
zimmerman_bridge.py        ← MitoSimulator: Zimmerman Simulator protocol adapter (imports simulator, analytics, constants, zimmerman.base)
zimmerman_analysis.py      ← Full 14-tool analysis runner + CLI (imports zimmerman_bridge, all 14 zimmerman modules)
zimmerman_viz.py           ← Matplotlib visualizations for Zimmerman reports (imports constants)

# Cramer-toolkit integration (Tier 7, no LLM, requires ~/cramer-toolkit + ~/zimmerman-toolkit)
kcramer_bridge.py          ← Biological stress scenarios + convenience analysis functions (imports kcramer toolkit, zimmerman_bridge)

# Grief→Mito Integration (Phase 2, requires ~/grief-simulator)
grief_bridge.py            ← GriefDisturbance + grief_trajectory() + grief_scenarios() (imports grief-simulator via importlib)
grief_mito_simulator.py    ← GriefMitoSimulator: Zimmerman adapter for 26D combined system
grief_mito_scenarios.py    ← Grief stress scenario bank (16 scenarios) for cramer-toolkit
grief_mito_viz.py          ← Side-by-side grief/mito visualization (trajectory + comparison + overview)

# Resilience suite (agroecology-inspired, no LLM, imports simulator + constants)
disturbances.py            ← 4 disturbance types + simulate_with_disturbances() custom RK4 loop (imports simulator, constants)
resilience_metrics.py      ← Resistance, recovery time, regime retention, elasticity, composite score (imports numpy)
resilience_viz.py          ← 5 visualization functions + CLI (imports disturbances, resilience_metrics, simulator, constants)

archive/orphans/protocol_mtdna_synthesis.py  ← Standalone (archived; no imports from project)
```

### TIQM Pipeline Mapping

| TIQM Concept | Robotics Project | This Project |
|---|---|---|
| Offer wave | LLM generates 12D weight+physics vector | LLM generates 12D intervention+patient vector |
| Simulation | PyBullet 3-link robot locomotion (4000 steps @ 240 Hz) | RK4 ODE of 8 state variables (3000 steps, 30 years) |
| Confirmation wave | VLM evaluates locomotion behavior | VLM evaluates cellular trajectory |
| Resonance | Semantic match to character/sequence seed | Clinical match to patient scenario |
| The "cliff" | Behavioral cliffs in 6D weight space | Heteroplasmy cliff at ~70% damaged mtDNA |

### Core Pipeline

1. **constants.py** — All biological constants (from Cramer 2025), 12D parameter space definitions with discrete grids, Ollama model config, 10 clinical scenario seeds
2. **simulator.py** — RK4 ODE integrator. `simulate(intervention, patient, tissue_type, stochastic, resolver)` returns full 30-year trajectory of 8 state variables. Supports tissue-specific profiles (brain/muscle/cardiac), stochastic Euler-Maruyama mode, and optional `ParameterResolver` for precision medicine expanded inputs
3. **analytics.py** — 4-pillar metrics computed from simulation results. `compute_all(result)` returns energy/damage/dynamics/intervention pillars
4. **llm_common.py** — Shared LLM utilities: `query_ollama()`, `query_ollama_raw()`, `parse_json_response()`, `parse_intervention_vector()`, `detect_flattening()`, `split_vector()`. Handles markdown fence stripping, think-tag removal, flattening detection, grid snapping
5. **tiqm_experiment.py** — Full TIQM pipeline: offer prompt → Ollama → parse 12D vector → snap to grid → simulate → compute analytics → confirmation prompt → parse resonance scores

### Simulator Details

The ODE system models 8 coupled variables integrated via 4th-order Runge-Kutta (dt=0.01 years ~ 3.65 days, 3000 steps over 30 years). Supports tissue-specific profiles (brain/muscle/cardiac/default from `constants.py:TISSUE_PROFILES`), optional stochastic Euler-Maruyama integration for confidence intervals, and time-varying intervention schedules via `InterventionSchedule`:

- **N_healthy** (state[0]): Healthy mtDNA copy count with homeostatic regulation toward total ≈ 1.0
- **N_deletion** (state[1]): Large-deletion mtDNA copies — exponential growth with replication advantage (1.10x), drive the heteroplasmy cliff. Source: Pol γ slippage
- **ATP** (state[2]): Energy production, driven by cliff factor (deletion het only) × NAD × (1 - senescence)
- **ROS** (state[3]): Reactive oxygen species, quadratic in total heteroplasmy; causes only point mutations (~33% of ROS→damage)
- **NAD** (state[4]): Age-declining cofactor, boosted by supplementation, drained by ROS/Yamanaka
- **Senescent_fraction** (state[5]): Senescence driven by ROS + low ATP + age, cleared by senolytics
- **Membrane_potential (ΔΨ)** (state[6]): Slave variable tracking cliff × NAD × (1 - senescence)
- **N_point** (state[7]): Point-mutation mtDNA copies — linear growth, no replication advantage. Sources: ~67% Pol γ errors + ~33% ROS-induced transitions. Functionally mild

Key dynamics post-falsifier fixes (2026-02-15):
- **Cliff feeds back into replication and apoptosis** (fix C1) — ATP collapse halts replication
- **Copy number regulated** (fix C2) — total N_h + N_d homeostatically targets 1.0
- **NAD selectively benefits healthy mitochondria** (fix C3) — quality control, not damaged boost
- **Bistability past cliff** (fix C4) — damaged replication advantage creates irreversible collapse
- **Yamanaka gated by ATP** (fix M1) — no energy → no repair

Cramer email corrections (2026-02-15):
- **CD38 degrades NMN/NR** (fix C7) — NAD+ boost gated by CD38 survival factor; low dose mostly destroyed, high dose includes apigenin CD38 suppression
- **Transplant is primary rejuvenation** (fix C8) — doubled addition rate, competitive displacement of damaged copies, the only method for reversing accumulated mtDNA damage
- **Split mutation types** (fix C11) — N_damaged split into N_deletion (state[1]) and N_point (state[7]). Cliff uses deletion heteroplasmy only. ROS→damage creates point mutations only (~33% of old coupling). Deletions grow via Pol γ slippage with 1.10x replication advantage

### 4-Pillar Analytics

| Pillar | Key Metrics |
|---|---|
| **Energy** | ATP trajectory, min/max/mean/CV, reserve ratio, slope, time-to-crisis, terminal slope |
| **Damage** | Heteroplasmy trajectory, cliff distance, time-to-cliff, acceleration, fraction above cliff |
| **Dynamics** | ROS FFT dominant freq/amplitude, membrane potential CV/slope, NAD slope, ROS-het correlation, ROS-ATP correlation, senescence rate |
| **Intervention** | ATP benefit (terminal + mean) vs baseline, het benefit, energy cost, benefit-cost ratio, crisis delay years |

### Research Campaign Scripts

**Tier 1 — Pure Simulation (no LLM):**
- **causal_surgery.py** — Mid-simulation intervention switching: no-treatment → intervention (and reverse) at various time points. Answers "when is it too late?" 3 patients × 4 interventions × 8 switch times × 2 directions = ~192 sims
- **dark_matter.py** — Random 12D sampling, classifies outcomes as thriving/stable/declining/collapsed/paradoxical. "Paradoxical" = intervention harms patient. ~700 sims
- **protocol_interpolation.py** — Linear interpolation between 5 champion protocols in 6D intervention space. Finds super-protocols at midpoints. Also does radial sweeps and 3D grid through (rapamycin, NAD, exercise) subspace. ~1325 sims

**Tier 2 — LLM Seed Experiments (requires Ollama):**
- **oeis_seed_experiment.py** — OEIS integer sequences as semantic seeds for intervention design via 4 models
- **character_seed_experiment.py** — 2000 fictional characters as seeds for patient+protocol generation across 4 models

**Tier 3 — LLM Meta-Analysis (requires Ollama):**
- **fisher_metric.py** — Queries each model 10× per scenario to measure output variance. High variance = clinical ambiguity. 10 scenarios × 10 repeats × 4 models = 400 queries
- **clinical_consensus.py** — Same scenario across 4 models: pairwise cosine similarity, per-parameter agreement, identifies controversial vs consensus scenarios

**Tier 4 — Synthesis:**
- **perturbation_probing.py** — Perturbs each of 12 params ±1 grid step around probe vectors. Maps intervention fragility. ~1250 sims
- **categorical_structure.py** — Formal functor validation: Sem→Vec→Beh distance correlations, faithfulness, sheaf consistency. Pure computation on prior experiment data
- **llm_seeded_evolution.py** — Hill-climbing from 20 LLM seeds + 20 random seeds × 100 evaluations each. Tests "launchpad vs trap" hypothesis. `--narrative` flag enables trajectory-feedback evolution (requires Ollama)

**Zimmerman-Informed Experiments (2026-02-15 upgrade):**
- **prompt_templates.py** — Prompt styles: numeric (original), diegetic (narrative), contrastive (Dr. Cautious vs Dr. Bold). Used by tiqm_experiment.py via `--style` flag
- **sobol_sensitivity.py** — Saltelli sampling + Sobol first-order (S1) and total-order (ST) indices. Captures parameter interactions missed by one-at-a-time perturbation. N=256 base → 6656 sims
- **pds_mapping.py** — Maps Zimmerman's Power/Danger/Structure dimensions from fictional character archetypes to the 6D patient parameter space. Compares PDS-predicted vs LLM-generated patient parameters
- **posiwid_audit.py** — POSIWID alignment: measures gap between LLM-stated intentions and actual simulation outcomes. Two-phase query (intention + protocol) per scenario. Requires Ollama
- **archetype_matchmaker.py** — Combines PDS mapping with character experiment data. Identifies which character archetypes produce the best protocols for which patient types. Tier 4 (needs prior character experiment data)

**Tier 5 — Discovery Tools (no LLM, exploit simulation for novel biology):**
- **interaction_mapper.py** (D4) — 2D grid sweeps of all 15 intervention pairs × 3 patient types. Measures super-additivity (synergy) and sub-additivity (antagonism). Compares with Sobol interaction indices if available. ~2160 sims
- **reachable_set.py** (D1) — Latin hypercube sampling of 6D intervention space. Maps achievable (het, ATP) region, Pareto frontier, minimum-intervention paths to health targets (maintain_health, significant_reversal, aggressive_reversal, cliff_escape). ~2400 sims
- **competing_evaluators.py** (D5) — 4 evaluator functions (ATP_Guardian, Het_Hunter, Crisis_Delayer, Efficiency_Auditor) score ~500 candidate protocols. Finds "transaction" protocols in top-25% for ALL evaluators. 4D Pareto frontier, evaluator agreement matrix. Optionally ingests Pareto/synergy data from D1/D4. ~1000 sims
- **temporal_optimizer.py** (D2) — (1+lambda) ES over phased InterventionSchedule genotypes (3 phases × 6 params + 2 boundaries = 20D). Compares optimal phased schedule to constant equivalent; timing_importance = relative improvement. ~3000 sims
- **multi_tissue_sim.py** (D3) — Wraps derivatives() to simulate brain+muscle+cardiac coupled by shared NAD+ pool, systemic inflammation, and cardiac blood flow. Tests 5 protocols × 4 allocation strategies + brain allocation sweep. Discovers cardiac cascade, worst-first allocation benefits. ~30 sims

**Resilience Suite (agroecology-inspired, no LLM):**
- **disturbances.py** — 4 disturbance types (IonizingRadiation, ToxinExposure, ChemotherapyBurst, InflammationBurst) with two-channel injection: impulse (one-shot state modification) + ongoing (parameter perturbation during shock window). Custom RK4 loop with 4-phase per-step logic (impulse, param modification, RK4, constraints). Each disturbance calibrated against actual ODE dynamics.
- **resilience_metrics.py** — 4 ecological resilience metrics: resistance (peak deviation during shock), recovery time (sustained return to tolerance, 10-step window ≈ 1 mitochondrial turnover cycle), regime retention (cliff-crossing detection), elasticity (2-year post-shock deviation slope). Composite score: 0.25 resistance + 0.30 recovery + 0.30 regime + 0.15 elasticity. Based on Holling (1973), Pimm (1984), Scheffer (2009).
- **resilience_viz.py** — 5 visualization functions: `plot_shock_response()` (4-panel ATP/het/ROS/ΔΨ with shock window shading), `plot_resilience_comparison()` (multi-trajectory overlay), `plot_resilience_summary()` (bar chart with traffic-light recovery time colors), `plot_recovery_landscape()` (magnitude × timing heatmap), `generate_all_plots()` (10-plot suite). CLI: `python resilience_viz.py [--disturbance radiation --magnitude 0.8]`.
- **tests/test_resilience.py** — 28 tests across 3 classes (disturbance classes, simulate_with_disturbances, resilience metrics). Module-scoped fixtures for baseline, radiation, and chemo results.

**Tier 6 — Zimmerman Toolkit Integration (no LLM, requires ~/zimmerman-toolkit):**
- **zimmerman_bridge.py** — `MitoSimulator` class satisfying the Zimmerman `Simulator` protocol (`run(params) -> dict`, `param_spec() -> bounds`). Accepts flat 12D param dict, splits into intervention + patient, runs 30-year simulation, computes 4-pillar analytics, returns flat dict of ~40 scalar metrics. Supports full 12D mode and intervention-only 6D mode with fixed patient override. Caches baseline simulation. Caps inf values to 999.0.
- **zimmerman_analysis.py** — CLI runner for all 14 Zimmerman tools: Sobol, Falsifier, ContrastiveGenerator, ContrastSetGenerator, PDSMapper, POSIWIDAuditor, PromptBuilder, LocalityProfiler, RelationGraphExtractor, Diegeticizer, TokenExtispicyWorkbench, PromptReceptiveField, SuperdiegeticBenchmark, MeaningConstructionDashboard. Supports `--tools` filter, `--patient` profiles, `--n-base` for Sobol, `--viz` for automatic plot generation. Saves individual JSON reports to `artifacts/zimmerman/` plus a compiled markdown dashboard. Full run: ~15-25 min, ~8000+ sims.
- **zimmerman_viz.py** — Matplotlib visualizations from Zimmerman reports: Sobol horizontal bars (S1/ST, color-coded intervention vs patient), contrastive flip frequency, locality decay curves, causal relation graph (bipartite param→output layout), POSIWID alignment heatmap, dashboard radar chart. Reads from saved JSON reports or accepts dict directly. Outputs to `output/zimmerman/`.

**Tier 7 — Cramer Toolkit Integration (no LLM, requires ~/cramer-toolkit + ~/zimmerman-toolkit):**
- **kcramer_bridge.py** — Domain-specific biological stress scenarios for the cramer-toolkit. Defines 6 scenario banks: INFLAMMATION_SCENARIOS (4), NAD_SCENARIOS (4), VULNERABILITY_SCENARIOS (3), DEMAND_SCENARIOS (3), AGING_SCENARIOS (6), COMBINED_SCENARIOS (5) = 25 total stress scenarios in ALL_STRESS_SCENARIOS. Also defines 5 reference intervention protocols (no_treatment, conservative, moderate, aggressive, transplant_focused). Convenience functions: `run_resilience_analysis()` (full robustness + regret + vulnerability + rankings), `run_vulnerability_analysis()` (sorted impact list), `run_scenario_comparison()` (any analysis function under multiple scenarios).
- **tests/test_kcramer_bridge.py** — 24 tests across 5 classes: scenario bank structure (11), scenario application (4), ScenarioSimulator integration (4), protocol bank (5).

## Precision Medicine Expansion (2026-02-19)

Three-layer architecture expanding the simulator from 8-state/12D to a precision medicine platform (~50D input, 14 downstream state variables) without modifying the Cramer core ODE.

### Architecture

```
~50D Expanded Inputs → [ParameterResolver] → 12D Core → [Cramer ODE] → 8 States → [DownstreamChain] → 6 ODEs + Memory Index
```

1. **ParameterResolver** (`parameter_resolver.py`): Maps genetics, lifestyle, supplements, grief, sleep to effective 12D core inputs via a 10-step modifier chain. Pre-computes time-varying trajectories (grief decay, alcohol taper, gut health) at construction, interpolates at each timestep.
2. **Cramer Core ODE** (`simulator.py`): Unchanged. Accepts `resolver=None` kwarg; when provided, calls `resolver.resolve(t)` instead of `_resolve_intervention()`.
3. **DownstreamChain** (`downstream_chain.py`): Post-processes core outputs into 6 additional ODEs (MEF2, histone acetylation, synaptic strength, cognitive reserve, amyloid-beta, tau) + derived `memory_index` and `resilience`.

### New Modules

| Module | Role | Depends On |
|--------|------|-----------|
| `genetics_module.py` | APOE4/FOXO3/CD38 genotype + sex/menopause modifiers | constants |
| `lifestyle_module.py` | Alcohol, coffee, diet, fasting effects | constants |
| `supplement_module.py` | Hill-function dose-response for 11 nutraceuticals | constants |
| `parameter_resolver.py` | 50D→12D modifier chain with time-varying trajectories | genetics, lifestyle, supplements |
| `downstream_chain.py` | MEF2, HA, SS, CR, amyloid, tau ODEs + memory_index | constants |
| `scenario_definitions.py` | InterventionProfile/Scenario dataclasses, A-D scenarios | — |
| `scenario_runner.py` | Pipeline: resolver → simulate → downstream | parameter_resolver, simulator, downstream_chain |
| `scenario_analysis.py` | Milestone extraction, scenario comparison | — |
| `scenario_plot.py` | Trajectory plots, milestone bars, heatmaps | matplotlib |
| `run_scenario_comparison.py` | CLI script for 4-scenario comparison | all above |

### New Commands

```bash
# 4-scenario comparison (63yo APOE4 female, scenarios A-D)
python run_scenario_comparison.py                    # Print milestones
python run_scenario_comparison.py --save-plots       # Also save plots to output/scenarios/
python run_scenario_comparison.py --years 40         # Custom duration
```

### Expanded Parameter Space

The resolver accepts ~50D input (14 patient + 24 intervention expanded params) and produces the standard 12D core pair. New patient params include `apoe_genotype` (0/1/2), `sex` (M/F), `menopause_status`, `grief_intensity`, `intellectual_engagement`, `education_level`. New intervention params include `sleep_intervention`, `alcohol_intake`, `coffee_intake`, `diet_type`, 11 supplement doses, `probiotic_intensity`, `therapy_intensity`.

### Design Documents

- `artifacts/design_doc_time_varying_parameter_resolver_2026-02-19.md`
- `artifacts/handoff_batch{1,2,3,4}_*_2026-02-19.md`
- `docs/plans/2026-02-19-precision-medicine-expansion.md`

## Protocol Dictionary Pipeline (2026-02-19)

Ported from rosetta-motion's pattern language, rewrite rules, multi-labeler, and review governance patterns. Provides a unified protocol curation system: ingest → enrich → classify → rewrite → review → save.

### Architecture

```
Sources (dark_matter, EA, LLM seeds) → [Ingest] → ProtocolRecord → [Enrich] → [Classify] → [Rewrite Rules] → [Review Queue] → ProtocolDictionary
```

### New Modules

| Module | Role | Rosetta-Motion Source |
|--------|------|---------------------|
| `protocol_record.py` | ProtocolRecord dataclass + fingerprinting | `DiscoveryRecord` |
| `protocol_dictionary.py` | Persistent catalog with query/dedup/summary | `MotionDiscovery` |
| `protocol_enrichment.py` | Complexity, clinical signature, prototype group | `controller_simplicity()`, `sensory_signature()` |
| `protocol_classifier.py` | Rule-based + analytics-fit multi-labeler | `discover_label()` |
| `protocol_rewrite_rules.py` | Declarative 3-layer rule engine with audit | `rewrite_rules.py` |
| `protocol_pattern_language.py` | DAG validation, topological sort, orchestration | `pattern_language.py`, `pattern_orchestrator.py` |
| `protocol_review.py` | JSONL review queue with resolve/provenance | `review_queue_append()` |
| `run_protocol_pipeline.py` | End-to-end pipeline runner + ingest adapters | `run_pipeline.py` |

### Pattern Files

- `patterns/protocol_pattern_language.v1.json` — 7-stage DAG (global → ingest → analytics → robustness → classify → review → report)
- `patterns/protocol_rewrite_rules.v1.json` — 4 default rules (Yamanaka energy warning, transplant-dominant flag, low-confidence routing, paradoxical investigation)

### New Commands

```bash
# Single protocol through pipeline
python run_protocol_pipeline.py --intervention '{"rapamycin_dose":0.5,"nad_supplement":0.75}' --output artifacts/protocol_pipeline

# Ingest from dark_matter artifact
python run_protocol_pipeline.py --source dark_matter --artifact artifacts/dark_matter.json

# Protocol pipeline tests (~68 tests across 9 files)
pytest tests/test_protocol_record.py tests/test_protocol_dictionary.py tests/test_protocol_enrichment.py tests/test_protocol_classifier.py tests/test_protocol_rewrite_rules.py tests/test_protocol_pattern_language.py tests/test_protocol_review.py tests/test_protocol_pipeline.py tests/test_protocol_pipeline_integration.py -v
```

### Design Documents

- `docs/plans/2026-02-19-protocol-dictionary-pipeline.md`

## 12D Parameter Space

The LLM generates a 12-dimensional vector for each clinical scenario:

**6 Intervention params** (all 0.0–1.0, snapped to grid [0, 0.1, 0.25, 0.5, 0.75, 1.0]):
- `rapamycin_dose` — mTOR inhibition → enhanced mitophagy
- `nad_supplement` — NAD+ precursor (NMN/NR) restoration
- `senolytic_dose` — Senescent cell clearance (dasatinib+quercetin)
- `yamanaka_intensity` — Partial reprogramming (OSKM). **WARNING: costs 3-5 MU of ATP**
- `transplant_rate` — Healthy mtDNA infusion via platelet-derived mitlets
- `exercise_level` — Hormetic adaptation (moderate ROS → antioxidant upregulation)

**6 Patient params** (ranges vary per parameter):
- `baseline_age` — 20–90 years
- `baseline_heteroplasmy` — 0.0–0.95 (cliff at 0.7)
- `baseline_nad_level` — 0.2–1.0
- `genetic_vulnerability` — 0.5–2.0 (haplogroup-dependent)
- `metabolic_demand` — 0.5–2.0 (brain=high, skin=low)
- `inflammation_level` — 0.0–1.0 (inflammaging)

All values snapped to discrete grids via `snap_param()` / `snap_all()` in `constants.py`.

## Key Biological Constants

| Constant | Value | Source |
|---|---|---|
| Heteroplasmy cliff (deletion het) | 0.50 | Recalibrated for C11 deletion-only het. Literature value 0.70 (Rossignol 2003) was total het; with deletions ~60% of total, equivalent deletion cliff is ~0.42-0.56. Uses 0.50 as conservative estimate |
| Cliff steepness (sigmoid) | 15.0 | Simulation calibration (not from book) |
| Deletion doubling (young) | 11.8 years | Cramer Appendix 2 p.155, Fig. 23 (Va23 data); also Ch. II.H p.15 |
| Deletion doubling (old) | 3.06 years | Cramer Appendix 2 p.155, Fig. 23 (Va23 data); transition at age 65 (corrected from 40) |
| Deletion replication advantage | 1.10x | Cramer Appendix 2 pp.154-155: book says "at least 21% faster" (Va23); code uses 10% (size-dependent advantage for large deletions) |
| Point mutation replication advantage | 1.00x | Point mutations have no replication advantage (no size reduction) |
| Yamanaka energy cost | 3-5 MU/day | Cramer Ch. VIII.A Table 3 p.100; Ch. VII.B p.95 says 3-10x (Ci24, Fo18) |
| Baseline ATP | 1.0 MU/day | Cramer Ch. VIII.A Table 3 p.100 (1 MU = 10^8 ATP releases) |
| Baseline mitophagy rate | 0.02/year | Mechanism: Cramer Ch. VI.B p.75 (PINK1/Parkin); rate is sim param |
| NAD decline rate | 0.01/year | Cramer Ch. VI.A.3 pp.72-73 (Ca16 = Camacho-Pereira 2016, Ch. VI refs p.87) |
| Senescence rate | 0.005/year | Cramer Ch. VII.A pp.89-92, Ch. VIII.F p.103; rate is sim param |
| Membrane potential | 1.0 (norm) | Cramer Ch. IV pp.46-47 (~180mV healthy); Ch. VI.B p.75 (low = PINK1 trigger) |
| CD38 base survival | 0.4 | Cramer Ch. VI.A.3 p.73: CD38 destroys NMN/NR; 40% survives at min dose |
| CD38 suppression gain | 0.6 | Apigenin suppresses CD38, raising survival to 100% at max dose |
| Transplant addition rate | 0.30 | Cramer Ch. VIII.G pp.104-107: primary rejuvenation (doubled from 0.15) |
| Transplant displacement | 0.12 | Competitive displacement of damaged copies by transplanted healthy mitos |
| Transplant headroom | 1.5 | Max total copies with transplant (raised from 1.2) |

## Common Code Patterns

### Running a quick simulation from Python

```python
from simulator import simulate
from analytics import compute_all

# No treatment, default 70-year-old patient
result = simulate()
print(f"Final het: {result['heteroplasmy'][-1]:.4f}")
print(f"Final ATP: {result['states'][-1, 2]:.4f}")

# With intervention cocktail
cocktail = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5}
treated = simulate(intervention=cocktail)
baseline = simulate()  # no-treatment for comparison
analytics = compute_all(treated, baseline)
```

### Querying Ollama for intervention vectors

```python
from llm_common import query_ollama, split_vector
vector, raw = query_ollama("qwen3-coder:30b", prompt)
if vector:
    intervention, patient = split_vector(vector)
    result = simulate(intervention=intervention, patient=patient)
```

### Grid snapping

```python
from constants import snap_param, snap_all
snap_param("rapamycin_dose", 0.37)  # → 0.25 (nearest grid point)
snap_all({"rapamycin_dose": 0.37, "baseline_age": 67})  # snaps all params
```

### Tissue-specific simulation

```python
from simulator import simulate
# Brain tissue: high demand, high ROS sensitivity, low biogenesis
brain_result = simulate(tissue_type="brain")
# Stochastic mode: 100 Euler-Maruyama trajectories
stoch = simulate(stochastic=True, n_trajectories=100, noise_scale=0.02)
# stoch["trajectories"] has shape (100, n_steps, 8)
```

### Prompt style selection

```python
from prompt_templates import PROMPT_STYLES, get_prompt
offer = get_prompt("diegetic", "offer")   # Zimmerman-informed narrative style
offer = get_prompt("contrastive", "offer")  # Dr. Cautious vs Dr. Bold
```

### Time-varying intervention schedules

```python
from simulator import simulate, phased_schedule, pulsed_schedule
from constants import DEFAULT_INTERVENTION

no_treatment = dict(DEFAULT_INTERVENTION)
cocktail = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
            "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0, "exercise_level": 0.5}

# Phased: no treatment years 0-10, then full cocktail years 10-30
schedule = phased_schedule([(0, no_treatment), (10, cocktail)])
result = simulate(intervention=schedule, sim_years=30)

# Pulsed: 5-year on/off cycles
schedule = pulsed_schedule(cocktail, no_treatment, period=5.0, duty_cycle=0.5)
result = simulate(intervention=schedule, sim_years=30)
```

### Loading and using sample patients

```python
import json
from constants import PATIENT_NAMES
from simulator import simulate
from analytics import compute_all

# Load normal population
with open("artifacts/sample_patients_100.json") as f:
    data = json.load(f)
patients = data["patients"]

# Simulate a patient
p = patients[0]
patient_dict = {k: p[k] for k in PATIENT_NAMES}
result = simulate(patient=patient_dict)
print(f"Patient {p['_id']}: ATP={result['states'][-1, 2]:.3f}")

# Load edge cases and filter by category
with open("artifacts/sample_patients_edge.json") as f:
    edge_data = json.load(f)
cliff_patients = [p for p in edge_data["patients"] if p.get("_category") == "cliff_boundary"]
```

### Generating fresh patient populations

```python
from generate_patients import generate_patients, generate_edge_patients

# Custom population with different seed/size
patients = generate_patients(n=200, seed=99)

# Edge cases (fixed set, no randomness)
edge = generate_edge_patients()
```

### Schema validation

```python
from schemas import FullProtocol, InterventionVector
from llm_common import validate_llm_output

# Direct validation
protocol = FullProtocol(rapamycin_dose=0.5, baseline_age=70)

# Validate raw LLM output dict
snapped, warnings = validate_llm_output(raw_dict)
```

### Resilience analysis

```python
from disturbances import IonizingRadiation, ChemotherapyBurst, simulate_with_disturbances
from resilience_metrics import compute_resilience
from simulator import simulate

# Single shock
shock = IonizingRadiation(start_year=10.0, magnitude=0.8)
result = simulate_with_disturbances(disturbances=[shock])
baseline = simulate()
metrics = compute_resilience(result, baseline)
print(f"Resilience: {metrics['summary_score']:.3f}")

# Multiple shocks
shocks = [
    IonizingRadiation(start_year=5.0, magnitude=0.5),
    ChemotherapyBurst(start_year=15.0, magnitude=0.6),
]
result = simulate_with_disturbances(disturbances=shocks)
```

### Zimmerman toolkit analysis

```python
from zimmerman_bridge import MitoSimulator
from zimmerman.sobol import sobol_sensitivity
from zimmerman.falsifier import Falsifier

# Full 12D mode (intervention + patient)
sim = MitoSimulator()
sobol = sobol_sensitivity(sim, n_base=32, seed=42)  # 448 sims

# Intervention-only mode (fixed patient)
sim_iv = MitoSimulator(intervention_only=True,
                        patient_override={"baseline_age": 80.0, ...})
falsifier = Falsifier(sim_iv)
report = falsifier.falsify(n_random=100, n_boundary=50)
```

### Cramer toolkit resilience analysis

```python
from kcramer_bridge import (
    MitoSimulator, ALL_STRESS_SCENARIOS, PROTOCOLS,
    run_resilience_analysis, run_vulnerability_analysis,
    run_scenario_comparison,
)
from kcramer import ScenarioSimulator, scenario_aware

# Full resilience analysis (robustness + regret + vulnerability + rankings)
sim = MitoSimulator()
report = run_resilience_analysis(sim, output_key="final_atp")
print(f"Most robust: {report['summary']['most_robust']['protocol']}")
print(f"Score: {report['summary']['most_robust']['score']:.3f}")

# Which scenarios hurt the moderate protocol most?
vuln = run_vulnerability_analysis(sim, protocol=PROTOCOLS["moderate"])
print(f"Worst scenario: {vuln[0]['scenario']}")

# Make any Zimmerman tool scenario-conditioned
from zimmerman.sobol import sobol_sensitivity
from kcramer_bridge import INFLAMMATION_SCENARIOS
results = scenario_aware(sobol_sensitivity, sim, INFLAMMATION_SCENARIOS, n_base=32)
# results["baseline"], results["mild_inflammaging"], etc.
```

### Grief->Mito integration

```python
from grief_bridge import GriefDisturbance, grief_trajectory, grief_scenarios
from disturbances import simulate_with_disturbances

# Single bereaved scenario
d = GriefDisturbance(
    grief_patient={"B": 0.8, "M": 0.8, "age": 65.0, "E_ctx": 0.6},
    grief_intervention={"act_int": 0.7, "slp_int": 0.8},
)
result = simulate_with_disturbances(disturbances=[d])

# All 8 clinical seeds x 2 intervention levels
for s in grief_scenarios():
    result = simulate_with_disturbances(disturbances=[s])

# Zimmerman adapter (26D combined system)
from grief_mito_simulator import GriefMitoSimulator
sim = GriefMitoSimulator()
result = sim.run({"grief_B": 0.9, "baseline_age": 70.0})

# Scenario bank for cramer-toolkit
from grief_mito_scenarios import GRIEF_STRESS_SCENARIOS, GRIEF_PROTOCOLS
for s in GRIEF_STRESS_SCENARIOS:
    r = simulate_with_disturbances(disturbances=[s["disturbance"]])
```

## Key Data Files

- `output/tiqm_*.json` — Per-scenario TIQM experiment results (offer + confirmation + analytics)
- `output/tiqm_summary.json` — Combined results for all 10 clinical scenarios
- `output/*.png` — Trajectory plots, cliff curves, intervention comparisons
- `artifacts/causal_surgery.json` — Treatment timing / point of no return results
- `artifacts/perturbation_probing.json` — Intervention fragility data
- `artifacts/falsifier_report_2026-02-15.md` — Critical ODE bug report and fixes
- `artifacts/upgrade_plan_jgc_2026-02-15.md` — Zimmerman-informed upgrade plan (6 phases)
- `artifacts/posiwid_audit.json` — POSIWID alignment audit results
- `artifacts/pds_mapping.json` — PDS→patient mapping results
- `artifacts/archetype_matchmaker.json` — Archetype→protocol matching results
- `artifacts/sample_patients_100.json` — 100 biologically correlated patients + evaluation metrics
- `artifacts/sample_patients_edge.json` — 82 edge-case patients for robustness testing + evaluation
- `artifacts/zimmerman/sobol_report.json` — Sobol global sensitivity indices (S1, ST per param per output)
- `artifacts/zimmerman/falsifier_report.json` — Falsification results (200 tests, boundary + adversarial)
- `artifacts/zimmerman/contrastive_report.json` — Contrastive pairs (minimal parameter flips across cliff)
- `artifacts/zimmerman/contrast_sets_report.json` — Ordered edit sequences with tipping points
- `artifacts/zimmerman/pds_report.json` — Power/Danger/Structure dimension mapping audit
- `artifacts/zimmerman/posiwid_report.json` — POSIWID alignment (intended vs actual outcomes)
- `artifacts/zimmerman/locality_report.json` — Locality profiles (perturbation decay curves)
- `artifacts/zimmerman/relation_graph_report.json` — Causal relation graph (504 edges)
- `artifacts/zimmerman/diegeticizer_report.json` — Narrative roundtrip fidelity
- `artifacts/zimmerman/token_extispicy_report.json` — Tokenization fragmentation hazard
- `artifacts/zimmerman/receptive_field_report.json` — Segment importance (pharmacological > biological > demographics > vulnerability)
- `artifacts/zimmerman/supradiegetic_benchmark_report.json` — Form vs meaning benchmark
- `artifacts/zimmerman/dashboard.json` — Unified dashboard aggregating all tools
- `artifacts/zimmerman/dashboard.md` — Markdown summary report
- `output/zimmerman/*.png` — Sobol bars, contrastive sensitivity, locality curves, relation graph, POSIWID alignment, dashboard radar
- `output/resilience_*.png` — Shock response, comparison, summary, recovery landscape plots
- `output/grief_mito/*.png` — Grief→mito trajectory panels, intervention comparison, scenario overview

## Patient Population Generator (`generate_patients.py`)

Generates two complementary patient populations for testing and experiment seeding.

### Usage

```python
from generate_patients import generate_patients, generate_edge_patients
from generate_patients import evaluate_population, evaluate_edge_population

# Normal population: biologically correlated, realistic distribution
patients = generate_patients(n=100, seed=42)
evaluation = evaluate_population(patients)

# Edge-case population: boundary conditions, stress tests
edge_patients = generate_edge_patients()
edge_evaluation = evaluate_edge_population(edge_patients)
```

### Normal Population (`generate_patients`, 100 patients)

Biology-informed correlation structure ensures realistic co-occurrence of parameter values:

| Parameter | Generation Method | Expected Range |
|---|---|---|
| `baseline_age` | Uniform 20–90 | Full lifespan |
| `baseline_heteroplasmy` | Age-driven: `0.05 + 0.45 * age_frac^1.3` + noise | Young ~0.05, old ~0.50 |
| `baseline_nad_level` | Age-driven: `0.95 - 0.50 * age_frac` + noise | Young ~0.95, old ~0.45 |
| `genetic_vulnerability` | Lognormal(0, 0.25), clipped [0.5, 2.0] | Most ~1.0, rare outliers |
| `metabolic_demand` | Weighted discrete choice | Most 1.0, rare 0.5 or 2.0 |
| `inflammation_level` | Age-driven: `0.05 + 0.40 * age_frac` + noise | Young ~0.05, old ~0.45 |

All values snapped to grid after generation. Key correlations verified:

| Pair | Expected | Measured | Biology |
|---|---|---|---|
| age ↔ het | positive | r = +0.80 | Older → more mtDNA damage |
| age ↔ NAD | negative | r = -0.92 | Older → lower NAD+ (Ca16) |
| age ↔ inflammation | positive | r = +0.66 | Inflammaging |
| het ↔ NAD | negative | r = -0.70 | Damage → worse NAD state |
| genetic_vuln ↔ metabolic_demand | ~zero | r = +0.05 | Independent (haplogroup vs tissue) |

Population quality score: **0.95/1.00** (grid coverage 90%, correlation plausibility 100%, outcome diversity 91%, clinical plausibility 100%).

Outcome distribution under no-treatment simulation: 9% collapsed, 19% declining, 39% stable, 33% healthy. 17% cross the heteroplasmy cliff.

### Edge-Case Population (`generate_edge_patients`, 82 patients)

Systematically constructed to test simulator robustness at boundary conditions. Eight categories:

| Category | N | Purpose | Examples |
|---|---|---|---|
| `single_extreme` | 12 | One param at min or max, rest default | `baseline_age_min` (20), `inflammation_level_max` (1.0) |
| `corner` | 10 | Multiple params at extremes simultaneously | `all_min` (healthiest), `all_max` (sickest), `checkerboard_A/B` (alternating) |
| `cliff_boundary` | 14 | Het sweep across the 0.70 cliff threshold | het = 0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90 + young/old/vulnerable/resilient at cliff |
| `contradictory` | 12 | Biologically unlikely but must not crash | `young_melas` (20yo, het=0.80), `old_superager` (90yo, het=0.05), `zero_het`, `max_het`, `nad_vs_infl` (competing signals) |
| `max_stress` | 10 | Worst-case parameter combinations | `triple_threat` (old+damaged+inflamed), `brain_on_fire`, `post_cliff_collapse`, `young_max_stress` |
| `tissue_vuln_cross` | 8 | 2×2×2 factorial: demand × vulnerability × age | All combinations of {low,high} demand × {resilient,fragile} × {young,old} |
| `age_at_cliff` | 8 | Every decade at het=0.70 | Ages 20, 30, 40, 50, 60, 70, 80, 90 all starting at the cliff |
| `near_limits` | 8 | Values at or near hard parameter boundaries | `het_epsilon` (0.01), `het_near_max` (0.94), `nad_barely_alive` (0.21), `exact_cliff` (0.70) |

Each patient has `_label` (human-readable name) and `_category` tags for filtering.

Robustness score: **1.00/1.00** — 0 crashes, 0 NaN/Inf, 0 negative states, 0 out-of-range heteroplasmy across all 82 edge cases.

Outcome distribution: 56% collapsed, 10% declining, 26% stable, 9% healthy (skewed toward collapse because edge cases are deliberately pathological). 59% cross the cliff.

Notable extremes:
- Lowest final ATP: 0.0061 (`all_max` — 90yo, het=0.95, NAD=0.2, vuln=2.0, demand=2.0, infl=1.0)
- Highest final ATP: 0.9157 (`all_min` — 20yo, het=0.02, NAD=1.0, vuln=0.5, demand=0.5, infl=0.0)
- Highest final het: 0.9997 (`all_max`)
- Lowest final het: 0.1164 (`all_min`)

### Quality Evaluation Metrics

`evaluate_population()` computes 4 scores for the normal population:
- **grid_coverage**: fraction of all grid points (across 6 params) that appear in the population
- **correlation_plausibility**: fraction of 5 expected biological correlations with correct sign and magnitude
- **outcome_diversity**: entropy of the 4 outcome categories (collapsed/declining/stable/healthy) normalized to [0,1]
- **clinical_plausibility**: penalized by implausible combinations (young+high_het, old+perfect_NAD, etc.)

`evaluate_edge_population()` computes 3 robustness scores:
- **crash_rate**: 1.0 - (crashes / total patients)
- **issue_rate**: 1.0 - (patients with NaN/Inf/negative/out-of-range / successful patients)
- **state_validity**: clean runs (no issues at all) / successful runs

## Conventions

- All state variables are non-negative (enforced post-step via `np.maximum(state, 0.0)`)
- Senescent fraction capped at 1.0
- Total heteroplasmy = (N_deletion + N_point) / (N_healthy + N_deletion + N_point); deletion heteroplasmy = N_deletion / (N_healthy + N_deletion + N_point). Cliff factor uses deletion heteroplasmy only. Returns 1.0 if total < 1e-12
- JSON output uses `NumpyEncoder` from `analytics.py` (rounds floats to 6 decimal places)
- Matplotlib uses Agg backend (headless, non-interactive)
- LLM responses parsed with markdown fence stripping and `<think>` tag removal (reasoning models)
- Different models for offer vs confirmation wave to prevent self-confirmation bias
- Output files: plots → `output/`, experiment JSON → `output/` or `artifacts/`
- Analytics pipeline is numpy-only (no scipy, no sklearn), matching parent ER project constraint
- All LLM query/parse utilities consolidated in `llm_common.py`: `query_ollama()`, `query_ollama_raw()`, `parse_json_response()`, `parse_intervention_vector()`, `detect_flattening()`, `validate_llm_output()`
- LLM outputs validated via pydantic schemas (`schemas.py`) before grid snapping — `InterventionVector`, `PatientProfile`, `FullProtocol` models with per-field range constraints
- Type annotations on all public functions in core modules (`constants.py`, `simulator.py`, `analytics.py`, `llm_common.py`, `schemas.py`); type aliases: `ParamDict`, `InterventionDict`, `PatientDict` in `constants.py`
- Time-varying interventions via `InterventionSchedule` class in `simulator.py`; convenience constructors `phased_schedule()` and `pulsed_schedule()`; plain dicts still work (backwards compatible)
- Prompt templates include 2 few-shot examples (young prevention + near-cliff emergency) in OFFER_NUMERIC and OFFER_DIEGETIC to reduce LLM flattening and key omission
- Formal test suite: `pytest tests/ -v` runs ~439 tests across 33 modules (test_simulator, test_analytics, test_llm_parsing, test_schemas, test_zimmerman_bridge, test_resilience, test_kcramer_bridge, test_grief_bridge, test_expansion_constants, test_genetics_module, test_lifestyle_module, test_supplement_module, test_parameter_resolver, test_resolver_integration, test_downstream_chain, test_scenario_framework, test_integration_scenarios, test_protocol_record, test_protocol_dictionary, test_protocol_enrichment, test_protocol_classifier, test_protocol_rewrite_rules, test_protocol_pattern_language, test_protocol_review, test_protocol_pipeline, test_protocol_pipeline_integration)
- 10 clinical scenario seeds are hardcoded in `constants.py:CLINICAL_SEEDS`

## Agents (.claude/agents/)

| Agent | Model | Role |
|---|---|---|
| `cliff-cartographer` | sonnet | Maps heteroplasmy cliff landscape, recommends simulation budget allocation |
| `intervention-surgeon` | sonnet | Designs minimal intervention modifications to test causal hypotheses |
| `trajectory-analyst` | sonnet | Analyzes trajectories across 4 pillars, compares patients/interventions |
| `clinical-matchmaker` | sonnet | Matches patient descriptions to intervention protocols from existing runs |
| `falsifier` | opus | Adversarial reviewer — attacks claims, checks model validity |
| `ollama-delegator` | sonnet | Composes Ollama prompts, parses responses, manages offer/confirmation waves |
| `preflight-validator` | haiku | Environment and config validation before running simulations |
| `paper-drafter` | opus | Drafts academic paper sections from findings |
| `wolfram-engine` | sonnet | Symbolic ODE analysis via wolframscript (equilibria, bifurcations, Jacobian) |
| `cross-project-weaver` | opus | Structural parallels with parent Evolutionary-Robotics project |
| `protocol-auditor` | opus | Reviews 9-step mtDNA protocol for safety, plausibility, costs |
| `patient-generator` | sonnet | Synthesizes realistic patient profiles from clinical correlations |
| `llm-panel` | sonnet | Multi-model consensus from local Ollama LLMs |
| `cloud-llm-panel` | sonnet | Frontier cloud LLM panel for hard questions |

## Ollama Models

| Model | Role | Notes |
|---|---|---|
| `qwen3-coder:30b` | Offer wave (primary) | Reasoning model, emits `<think>` tags, max_tokens boosted to 3000 |
| `llama3.1:latest` | Confirmation wave | Lighter/faster, used for trajectory evaluation |
| `deepseek-r1:8b` | Alternative offer model | Reasoning model, emits `<think>` tags |
| `gpt-oss:20b` | Alternative offer model | General-purpose |

## Relationship to Parent Project

This project mirrors the Evolutionary-Robotics project's architecture point-for-point:

| Component | ER Project | This Project |
|---|---|---|
| Parameter space | 6D weights + 6D physics = 12D | 6D intervention + 6D patient = 12D |
| Simulation | PyBullet rigid-body physics | RK4 ODE of 8 mitochondrial state variables |
| Analytics | Beer-framework 4 pillars | Health-framework 4 pillars |
| Shared LLM utils | `structured_random_common.py` | `llm_common.py` |
| Cliff analysis | `atlas_cliffiness.py` | `cliff_mapping.py` |
| Treatment timing | `causal_surgery.py` (brain transplants) | `causal_surgery.py` (intervention switching) |
| Dead/futile analysis | `analyze_dark_matter.py` (dead gaits) | `dark_matter.py` (futile interventions) |
| Interpolation | `gait_interpolation.py` (weight space) | `protocol_interpolation.py` (intervention space) |
| Perturbation | `perturbation_probing.py` | `perturbation_probing.py` |
| Categorical validation | `categorical_structure.py` | `categorical_structure.py` |
| Fisher metric | `fisher_metric.py` | `fisher_metric.py` |
| LLM-seeded evolution | `llm_seeded_evolution.py` | `llm_seeded_evolution.py` |
