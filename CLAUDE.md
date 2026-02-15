# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Action Items

- **REPORT TO JOHN CRAMER:** The falsifier agent found 4 critical bugs in the ODE equations (2026-02-15). Full report: `artifacts/falsifier_report_2026-02-15.md`. Key issues: cliff was cosmetic not dynamical, mtDNA copy number unbounded, NAD supplementation inverted therapeutic sign, universal attractor prevented cliff crossing. Fixes were applied same day — Cramer should review the corrected dynamics for biological plausibility.

## Project Overview

Computational simulation of mitochondrial aging dynamics and intervention strategies, based on John G. Cramer's book *How to Live Much Longer: The Mitochondrial DNA Connection* (2025, ISBN 979-8-9928220-0-4).

Adapts the TIQM (Transactional Interpretation of Quantum Mechanics) pipeline from the parent [Evolutionary-Robotics](../Evolutionary-Robotics/) project. Instead of LLM → physics simulation → VLM scoring for robot locomotion, we use LLM → mitochondrial ODE simulation → VLM scoring for intervention protocol design.

**Core thesis (Cramer 2025):** Aging is a cellular energy crisis caused by progressive mitochondrial DNA damage. When the fraction of damaged mtDNA (heteroplasmy) exceeds ~70% — the "heteroplasmy cliff" — ATP production collapses nonlinearly.

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
python simulator.py       # ODE integrator test with 8 scenarios (incl. tissue types + stochastic)
python analytics.py       # 4-pillar analytics test
python cliff_mapping.py   # Heteroplasmy cliff mapping (~2 min)
python visualize.py       # Generate all plots to output/

# TIQM experiments (requires Ollama)
python tiqm_experiment.py                     # All 10 clinical scenarios
python tiqm_experiment.py --single            # Quick single test
python tiqm_experiment.py --style diegetic    # Zimmerman-informed narrative prompts
python tiqm_experiment.py --contrastive       # Dr. Cautious vs Dr. Bold protocols

# Protocol
python protocol_mtdna_synthesis.py  # Print 9-step mtDNA synthesis protocol

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

# Zimmerman-informed experiments (2026-02-15 upgrade)
python sobol_sensitivity.py      # Sobol global sensitivity analysis (~3 min, ~6656 sims)
python pds_mapping.py            # PDS→patient parameter mapping (Zimmerman Ch. 4)
python posiwid_audit.py          # POSIWID alignment audit (requires Ollama, ~15-20 min)
python archetype_matchmaker.py   # Archetype→protocol matchmaker (needs character data)
```

## Architecture

### Dependency Graph

```
constants.py          ← Central config (no dependencies)
    ↓
simulator.py          ← RK4 ODE integrator (imports constants)
    ↓
analytics.py          ← 4-pillar metrics (imports constants, simulator)
    ↓
llm_common.py         ← Shared LLM utilities (imports constants)
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

protocol_mtdna_synthesis.py  ← Standalone (no imports from project)
```

### TIQM Pipeline Mapping

| TIQM Concept | Robotics Project | This Project |
|---|---|---|
| Offer wave | LLM generates 12D weight+physics vector | LLM generates 12D intervention+patient vector |
| Simulation | PyBullet 3-link robot locomotion (4000 steps @ 240 Hz) | RK4 ODE of 7 state variables (3000 steps, 30 years) |
| Confirmation wave | VLM evaluates locomotion behavior | VLM evaluates cellular trajectory |
| Resonance | Semantic match to character/sequence seed | Clinical match to patient scenario |
| The "cliff" | Behavioral cliffs in 6D weight space | Heteroplasmy cliff at ~70% damaged mtDNA |

### Core Pipeline

1. **constants.py** — All biological constants (from Cramer 2025), 12D parameter space definitions with discrete grids, Ollama model config, 10 clinical scenario seeds
2. **simulator.py** — RK4 ODE integrator. `simulate(intervention, patient, tissue_type, stochastic)` returns full 30-year trajectory of 7 state variables. Supports tissue-specific profiles (brain/muscle/cardiac) and stochastic Euler-Maruyama mode
3. **analytics.py** — 4-pillar metrics computed from simulation results. `compute_all(result)` returns energy/damage/dynamics/intervention pillars
4. **llm_common.py** — Shared LLM utilities: `query_ollama()`, `query_ollama_raw()`, `parse_json_response()`, `parse_intervention_vector()`, `detect_flattening()`, `split_vector()`. Handles markdown fence stripping, think-tag removal, flattening detection, grid snapping
5. **tiqm_experiment.py** — Full TIQM pipeline: offer prompt → Ollama → parse 12D vector → snap to grid → simulate → compute analytics → confirmation prompt → parse resonance scores

### Simulator Details

The ODE system models 7 coupled variables integrated via 4th-order Runge-Kutta (dt=0.01 years ~ 3.65 days, 3000 steps over 30 years). Supports tissue-specific profiles (brain/muscle/cardiac/default from `constants.py:TISSUE_PROFILES`) and optional stochastic Euler-Maruyama integration for confidence intervals:

- **N_healthy / N_damaged**: mtDNA copy counts with homeostatic regulation toward total ≈ 1.0
- **ATP**: Energy production, driven by cliff factor × NAD × (1 - senescence)
- **ROS**: Reactive oxygen species, quadratic in heteroplasmy (vicious cycle)
- **NAD**: Age-declining cofactor, boosted by supplementation, drained by ROS/Yamanaka
- **Senescent_fraction**: Senescence driven by ROS + low ATP + age, cleared by senolytics
- **Membrane_potential (ΔΨ)**: Slave variable tracking cliff × NAD × (1 - senescence)

Key dynamics post-falsifier fixes (2026-02-15):
- **Cliff feeds back into replication and apoptosis** (fix C1) — ATP collapse halts replication
- **Copy number regulated** (fix C2) — total N_h + N_d homeostatically targets 1.0
- **NAD selectively benefits healthy mitochondria** (fix C3) — quality control, not damaged boost
- **Bistability past cliff** (fix C4) — damaged replication advantage creates irreversible collapse
- **Yamanaka gated by ATP** (fix M1) — no energy → no repair

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
| Heteroplasmy cliff | 0.70 | Mitochondrial genetics literature (Rossignol 2003); Cramer discusses different metric (MitoClock ~25%, Ch. V.K p.66) |
| Cliff steepness (sigmoid) | 15.0 | Simulation calibration (not from book) |
| Deletion doubling (young) | 11.8 years | Cramer Appendix 2 p.155, Fig. 23 (Va23 data); also Ch. II.H p.15 |
| Deletion doubling (old) | 3.06 years | Cramer Appendix 2 p.155, Fig. 23 (Va23 data); book says age 65 transition, sim uses 40 |
| Damaged replication advantage | 1.05x | Cramer Appendix 2 pp.154-155: book says "at least 21% faster" (Va23); code uses conservative 5% |
| Yamanaka energy cost | 3-5 MU/day | Cramer Ch. VIII.A Table 3 p.100; Ch. VII.B p.95 says 3-10x (Ci24, Fo18) |
| Baseline ATP | 1.0 MU/day | Cramer Ch. VIII.A Table 3 p.100 (1 MU = 10^8 ATP releases) |
| Baseline mitophagy rate | 0.02/year | Mechanism: Cramer Ch. VI.B p.75 (PINK1/Parkin); rate is sim param |
| NAD decline rate | 0.01/year | Cramer Ch. VI.A.3 pp.72-73 (Ca16 = Camacho-Pereira 2016, Ch. VI refs p.87) |
| Senescence rate | 0.005/year | Cramer Ch. VII.A pp.89-92, Ch. VIII.F p.103; rate is sim param |
| Membrane potential | 1.0 (norm) | Cramer Ch. IV pp.46-47 (~180mV healthy); Ch. VI.B p.75 (low = PINK1 trigger) |

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
# stoch["trajectories"] has shape (100, n_steps, 7)
```

### Prompt style selection

```python
from prompt_templates import PROMPT_STYLES, get_prompt
offer = get_prompt("diegetic", "offer")   # Zimmerman-informed narrative style
offer = get_prompt("contrastive", "offer")  # Dr. Cautious vs Dr. Bold
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

## Conventions

- All state variables are non-negative (enforced post-step via `np.maximum(state, 0.0)`)
- Senescent fraction capped at 1.0
- Heteroplasmy = N_damaged / (N_healthy + N_damaged); returns 1.0 if total < 1e-12
- JSON output uses `NumpyEncoder` from `analytics.py` (rounds floats to 6 decimal places)
- Matplotlib uses Agg backend (headless, non-interactive)
- LLM responses parsed with markdown fence stripping and `<think>` tag removal (reasoning models)
- Different models for offer vs confirmation wave to prevent self-confirmation bias
- Output files: plots → `output/`, experiment JSON → `output/` or `artifacts/`
- Analytics pipeline is numpy-only (no scipy, no sklearn), matching parent ER project constraint
- All LLM query/parse utilities consolidated in `llm_common.py`: `query_ollama()`, `query_ollama_raw()`, `parse_json_response()`, `parse_intervention_vector()`, `detect_flattening()`
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
| Simulation | PyBullet rigid-body physics | RK4 ODE of 7 mitochondrial state variables |
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
