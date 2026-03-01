# ODD Protocol: How to Live Much Longer — Mitochondrial Aging Simulator

**Following Grimm et al. (2020), "The ODD Protocol for Describing Agent-Based and Other Simulation Models: A Second Update."**

**Model version:** `C11 Layout (Point Mutation Split)`
**Date:** 2026-02-28
**Authors:** John G. Cramer, Kathryn Cramer, Claude Opus 4.6
**Repository:** `how-to-live-much-longer/`

---

## I. OVERVIEW

### 1. Purpose and Patterns

#### 1.1 Purpose
The purpose of this model is to simulate the 8-variable coupled dynamics of mitochondrial DNA (mtDNA) aging and evaluate the efficacy of various pharmacological and lifestyle interventions. It specifically targets the discovery of protocols that can slow or reverse the "heteroplasmy cliff"—the nonlinear collapse of cellular energy (ATP).

#### 1.2 Patterns
The model reproduces and is validated against the following empirical biological patterns:
*   **The Heteroplasmy Cliff:** ATP production remains stable until ~70% heteroplasmy, then collapses nonlinearly (Rossignol et al., 2003).
*   **Clonal Expansion of Deletions:** mtDNA deletions expand exponentially, doubling faster in older age due to reduced mitochondrial turnover.
*   **ROS-Damage Vicious Cycle:** The mutual reinforcement of oxidative stress and genomic damage observed in aging tissues.
*   **Rejuvenation Thresholds:** The observation that single-angle interventions (e.g., only NAD+) are insufficient to reverse established heteroplasmy without "Mitlet" transplantation.

### 2. Entities, State Variables, and Scales

#### 2.1 Entities and State Variables
*   **Cellular Metabolic Unit:**
    *   *N_healthy:* Healthy wild-type mtDNA copies.
    *   *N_deletion:* mtDNA with deletions (clonal expansion).
    *   *N_point:* Point-mutated mtDNA (linear drift).
    *   *ATP:* Current production rate (normalized to MU/day).
    *   *ROS:* Oxidative stress level.
    *   *NAD:* Cofactor availability.
    *   *Senescent_fraction:* Fraction of unit in arrest.
    *   *Membrane_potential:* Mitochondrial inner membrane ΔΨ.

#### 2.2 Scales
*   **Temporal:** dt = 0.01 years (~3.65 days). Default horizon: 30–100 years.
*   **Spatial:** Non-spatial (mean-field approximation of cellular state).

### 3. Process Overview and Scheduling
The simulation uses a 4th-order Runge-Kutta (RK4) scheme:
1.  **Calculate Derivatives:** Evaluate coupled ODEs based on current state and intervention levels.
2.  **RK4 Step:** Solver updates the 8D state vector.
3.  **Homeostatic Adjustment:** Adjust replication rates to maintain target copy number (N_total).
4.  **Downstream Chain Update:** Propagate energy state to cognitive reserve and amyloid/tau accumulation models.

---
## II. DESIGN CONCEPTS

### 4. Design Concepts

#### 4.1 Basic Principles (ODD+D)
*   **Theoretical Background:** The model is grounded in the **Mitochondrial Theory of Aging (MTA)** and the **Energy-Resilience Hypothesis**. It posits that aging is a solvable energy crisis.
*   **Decision-Making Objectives:** The agent (patient/clinician) objective is to maximize **Metabolic Stability** (ATP output) over a 30-year horizon while minimizing the metabolic cost of interventions.
*   **Decision Rules:** Interventions are selected based on **Threshold Logic**: if heteroplasmy > 0.5, activate state-restoration (Transplantation); if ROS > 1.2, activate rate-reduction (Senolytics).

#### 4.2 Individual Decision-Making
*   **Information:** The system has perfect information regarding its 8 state variables (though the patient/agent may only sense ATP/Energy).
*   **Perception:** The model simulates the lag in perceiving mitochondrial damage until the "Energy Crisis" (ATP collapse) occurs.
*   **Prediction:** The model uses its internal ODE system to predict the 30-year outcome of a given intervention cocktail.

#### 4.3 Emergence
...

*   **The Energy Crash:** Emerges from the failure of respiratory chain complex assembly as deletion heteroplasmy crosses the threshold.
*   **Bistability:** The system tends to "lock in" to a low-energy state once the cliff is passed.

#### 4.3 Adaptation
Agents (cells) adapt via **Mitophagy** (selective removal of damaged copies) and **Biogenesis** (increased replication under energy stress).

#### 4.4 Objectives
The system strives for **Metabolic Homeostasis** (maintaining ATP=1.0 and N_total=1.0).

#### 4.5 Learning
Not applicable (physiological model).

#### 4.6 Prediction
The model generates decadal projections of survival probability and "Biological Age."

#### 4.7 Sensing
The biogenesis engine senses the "Energy Gap" (target_atp - current_atp) and "Copy Pressure."

#### 4.8 Interaction
None (non-spatial model).

#### 4.9 Stochasticity
Individual variability is introduced via the "Patient Profile" parameters (initial damage, genetic vulnerability).

#### 4.10 Collectives
None.

#### 4.11 Observation
Trajectories are tracked via `simulator.py`, producing plots of Heteroplasmy vs. ATP and survival probability.

---

## III. DETAILS

### 5. Initialization
Initialized with real-world patient data. Base case represents a healthy 20-year-old; clinical cases represent older individuals with accumulated damage.

### 6. Input Data
Ground truth constants from Cramer (2026) *How to Live Much Longer*.

### 7. Submodels
*   **Mitochondrial Transplantation:** Models the "Mitlet" protocol (displacement of damaged copies by healthy donors).
*   **Yamanaka Reprogramming:** Models partial epigenetic reset gated by ATP availability.
*   **Downstream Chain:** Models the impact of energy failure on cognitive function and amyloid accumulation.
