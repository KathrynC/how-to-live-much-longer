# GEMINI Context: How to Live Much Longer

**This is the primary repository to start with.**

This project is a computational simulation of mitochondrial aging dynamics and intervention strategies, based on John G. Cramer's forthcoming book, *How to Live Much Longer: The Mitochondrial DNA Connection* (Springer, 2026).

## Project Overview

- **Core Thesis:** Aging is a cellular energy crisis caused by mitochondrial DNA (mtDNA) damage. When heteroplasmy (damaged mtDNA fraction) exceeds ~70% (the "heteroplasmy cliff"), ATP production collapses.
- **Technology Stack:** Python (NumPy, Matplotlib, Pydantic), RK4 ODE solvers, Semantic Cellular Automaton (CA) bridge.
- **Key Modules:**
  - `simulator.py`: 8-variable ODE model of cellular energetics.
  - `parameter_resolver.py`: Maps ~50D precision medicine inputs to 12D core simulation parameters.
  - `ca_simulator.py`: Semantic CA bridge for clinical interpretability and high-throughput screening.
  - `downstream_chain.py`: Models cognitive reserve, amyloid/tau accumulation, and memory index.

## Operational Commands

```bash
# Setup
conda env create -f environment.yml
conda activate mito-aging

# Core Simulation & Analytics
python simulator.py          # Run baseline ODE test
python run_scenario_comparison.py  # Run 4-scenario precision medicine comparison

# Resilience & Stress Testing
python resilience_viz.py     # Disturbance shocks (radiation, chemo, etc.)
python kcramer_tools_runner.py --mode resilience # Scenario-based robustness

# Semantic CA Bridge
python ca_visualize.py       # Discretized state transitions and rule analysis
python hyper_sobol.py        # 45-D sensitivity analysis of CA rules
```

## Critical Conventions

- **Cramer Ground Truth:** All biological constants in `constants.py` are traced to specific chapters/pages in the Cramer (2026) manuscript.
- **Slowing vs. Reversing:** The model distinguishes between rate-reduction (slowing) and state-restoration (reversing). Reversal typically requires multi-angle "cocktail" interventions.
- **12D Parameter Space:** Interventions (6D) and Patient Profiles (6D) form the core simulation vector.
- **TIQM Pipeline:** This project adapts the Transactional Interpretation of Quantum Mechanics (TIQM) pipeline for intervention protocol design.

## Project Structure

- `artifacts/`: JSON results, plots, and analysis reports.
- `docs/`: Technical documentation and optimization references.
- `tests/`: Extensive pytest suite (~385 tests) covering book conformance and precision medicine logic.
- `scripts/`: Utility scripts, including Zotero indexing for literature grounding.

**Note:** Ensure `Ollama` is running locally if executing Tier 2+ research campaigns (`tiqm_experiment.py`).
