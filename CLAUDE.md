# CLAUDE.md — Claude Code Guidance

## Commands

```bash
# Standalone tests (no Ollama needed)
python simulator.py       # ODE integrator test with 4 scenarios
python analytics.py       # 4-pillar analytics test
python cliff_mapping.py   # Heteroplasmy cliff mapping (~2 min)
python visualize.py       # Generate all plots to output/

# TIQM experiments (requires Ollama)
python tiqm_experiment.py           # All 10 clinical scenarios
python tiqm_experiment.py --single  # Quick single test

# Protocol
python protocol_mtdna_synthesis.py  # Print 9-step protocol
```

## Architecture

```
constants.py          ← Central config (no dependencies)
    ↓
simulator.py          ← RK4 ODE integrator (imports constants)
    ↓
analytics.py          ← 4-pillar metrics (imports constants, simulator)
    ↓
cliff_mapping.py      ← Cliff analysis (imports simulator, constants)
visualize.py          ← Matplotlib plots (imports simulator, analytics, cliff_mapping)
tiqm_experiment.py    ← TIQM pipeline (imports everything + Ollama)

protocol_mtdna_synthesis.py  ← Standalone (no imports from project)
```

## 12D Parameter Space

The LLM generates a 12-dimensional vector for each clinical scenario:
- 6 intervention params: rapamycin_dose, nad_supplement, senolytic_dose, yamanaka_intensity, transplant_rate, exercise_level
- 6 patient params: baseline_age, baseline_heteroplasmy, baseline_nad_level, genetic_vulnerability, metabolic_demand, inflammation_level

All values are snapped to discrete grids defined in `constants.py` via `snap_param()` / `snap_all()`.

## 7 ODE State Variables

`[N_healthy, N_damaged, ATP, ROS, NAD, Senescent_fraction, Membrane_potential]`

Integrated via RK4 with dt=0.01 years over 30 years (3000 steps).

## Key Biological Constants

- Heteroplasmy cliff: 0.7 (sigmoid steepness: 15.0)
- Deletion doubling time: 11.8yr (young), 3.06yr (>40yo)
- Yamanaka energy cost: 3-5 MU/day
- Damaged mtDNA replication advantage: 1.05×

## Conventions

- All state variables are non-negative (enforced post-step)
- Senescent fraction capped at 1.0
- Heteroplasmy = N_damaged / (N_healthy + N_damaged)
- JSON output uses NumpyEncoder (6 decimal places)
- Matplotlib uses Agg backend (headless)
- LLM responses parsed with markdown fence stripping and think-tag removal
- Different models for offer vs confirmation wave (anti-self-confirmation)
- Output files go to `output/` directory

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

- Offer wave: `qwen3-coder:30b` (or any model in constants.OFFER_MODEL)
- Confirmation wave: `llama3.1:latest` (lighter, faster)
- Reasoning models (emit `<think>` tags): deepseek-r1:8b, qwen3-coder:30b
