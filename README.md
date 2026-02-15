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
- **Age-dependent deletion doubling:** 11.8 years (young) → 3.06 years (old), from Va23 nanopore sequencing data (Appendix 2, p.155, Fig. 23). Book transition age is 65; simulation uses 40.
- **Replication advantage:** Shorter damaged mtDNA replicates faster — book says "at least 21%" for deletions >3kbp (Appendix 2, pp.154-155); simulation conservatively uses 5%
- **Energy units:** 1 MU (Metabolic Unit) = 10^8 ATP energy releases. Normal cell = ~1 MU/day (Ch. VIII.A, Table 3, p.100)
- **Interventions:** Rapamycin → mTOR → mitophagy (Ch. VI.A.1 pp.71-72), NAD+ via NMN/NR (Ch. VI.A.3 pp.72-73), senolytics D+Q+F (Ch. VII.A.2 p.91), Yamanaka reprogramming at 3-5 MU cost (Ch. VIII.A Table 3 p.100, Ch. VII.B p.95), transplant via mitlets (Ch. VIII.G pp.104-107), exercise (hormesis)

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
| Deletion doubling (old) | 3.06 yr | Appendix 2, p.155, Fig. 23 (book transition at age 65; sim uses 40) |
| Replication advantage | 1.05x (sim) | Appendix 2, pp.154-155: "at least 21% faster" for >3kbp deletions |
| NAD+ decline | 0.01/yr | Ch. VI.A.3, pp.72-73 (Ca16: Camacho-Pereira et al., 2016) |
| Mitophagy (PINK1/Parkin) | 0.02/yr | Ch. VI.B, p.75: mechanism described, rate is simulation parameter |
| Senescence rate | 0.005/yr | Ch. VII.A, pp.89-92; Ch. VIII.F, p.103: ~2x energy, SASP |
| Membrane potential (ΔΨ) | 1.0 (norm) | Ch. IV, pp.46-47: proton gradient; Ch. VI.B, p.75: low ΔΨ → PINK1 |
| Heteroplasmy cliff | 0.70 | Literature (Rossignol 2003); book uses different MitoClock metric (~25%, Ch. V.K p.66) |

See the citation key at the top of `constants.py` for full details on each constant's provenance.

## Files

### Core Pipeline

| File | Description |
|---|---|
| `constants.py` | Central configuration, 12D parameter space, biological constants |
| `simulator.py` | RK4 ODE integrator for 7 state variables |
| `analytics.py` | 4-pillar health analytics computation |
| `llm_common.py` | Shared LLM utilities (Ollama query, response parsing, grid snapping) |
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
