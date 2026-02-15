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

- **ROS-damage vicious cycle:** Damaged mitochondria → excess ROS → more mtDNA damage
- **Heteroplasmy cliff:** Steep sigmoid at ~70% — ATP collapses above this threshold
- **Age-dependent deletion doubling:** 11.8 years (young) → 3.06 years (>40yo)
- **Replication advantage:** Shorter damaged mtDNA replicates ~5% faster
- **Interventions:** Rapamycin (mitophagy), NAD+ (cofactor), senolytics (clearance), Yamanaka (repair, high cost), transplant (healthy copies), exercise (hormesis)

## 4-Pillar Analytics

| Pillar | Metrics |
|---|---|
| **Energy** | ATP trajectory, min ATP, time-to-crisis, reserve ratio, slope, CV |
| **Damage** | Heteroplasmy trajectory, time-to-cliff, acceleration, cliff distance |
| **Dynamics** | ROS FFT frequency, membrane potential CV, NAD slope, ROS-het correlation |
| **Intervention** | Energy cost, het benefit, ATP benefit, benefit-cost ratio, crisis delay |

## Files

| File | Description |
|---|---|
| `constants.py` | Central configuration, 12D parameter space, biological constants |
| `simulator.py` | RK4 ODE integrator for 7 state variables |
| `analytics.py` | 4-pillar health analytics computation |
| `cliff_mapping.py` | Heteroplasmy threshold mapping and cliff characterization |
| `visualize.py` | Matplotlib plots (Agg backend) |
| `tiqm_experiment.py` | Full TIQM pipeline with Ollama LLM integration |
| `protocol_mtdna_synthesis.py` | 9-step mtDNA synthesis and transplant protocol |

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mito-aging

# Run the simulator standalone test
python simulator.py

# Run analytics test
python analytics.py

# Run cliff mapping
python cliff_mapping.py

# Generate visualizations
python visualize.py

# Run TIQM experiments (requires Ollama running locally)
ollama serve &
python tiqm_experiment.py

# Quick single-scenario test
python tiqm_experiment.py --single

# Print the mtDNA synthesis protocol
python protocol_mtdna_synthesis.py
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
- McCully, J.D. et al. (2009). "Injection of isolated mitochondria during early reperfusion for cardioprotection." *American Journal of Physiology*, 296(1), H94–H105.
- Wallace, D.C. (2005). "A mitochondrial paradigm of metabolic and degenerative diseases, aging, and cancer." *Genetics*, 163(4), 1215–1241.
