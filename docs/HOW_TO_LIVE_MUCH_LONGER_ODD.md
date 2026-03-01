# ODD Protocol: How to Live Much Longer — Mitochondrial Aging Simulator

**Following Grimm et al. (2020), "The ODD Protocol for Describing Agent-Based and Other Simulation Models: A Second Update."**

**Model version:** `C11 Layout (Point Mutation Split)`
**Date:** 2026-02-28
**Authors:** John G. Cramer, Kathryn Cramer, Claude Opus 4.6
**Repository:** `how-to-live-much-longer/`

---

## I. OVERVIEW

### 1. Purpose
The purpose of this model is to simulate the 8-variable coupled dynamics of mitochondrial DNA (mtDNA) aging and evaluate the efficacy of various pharmacological and lifestyle interventions. It specifically targets the discovery of protocols that can slow or reverse the "heteroplasmy cliff"—the nonlinear collapse of cellular energy (ATP) that occurs when damaged mtDNA exceeds a critical threshold.

### 2. Entities, State Variables, and Scales
The primary entity is a **Cellular Metabolic Unit**.

**State Variables (8D):**
1.  **N_healthy:** Healthy wild-type mtDNA copies (normalized to 1.0).
2.  **N_deletion:** mtDNA with large-scale deletions (exponential growth, drives the cliff).
3.  **ATP:** Adenosine triphosphate production rate (MU/day).
4.  **ROS:** Reactive Oxygen Species level (normalized).
5.  **NAD:** Nicotinamide Adenine Dinucleotide cofactor availability (normalized).
6.  **Senescent_fraction:** Fraction of cells in growth arrest.
7.  **Membrane_potential:** Mitochondrial inner membrane ΔΨ (normalized).
8.  **N_point:** Point-mutated mtDNA (linear growth, functionally mild).

**Scales:**
*   **Temporal:** dt = 0.01 years (~3.65 days). Default horizon: 30–100 years.
*   **Spatial:** Non-spatial (mean-field approximation of cellular state).

### 3. Process Overview and Scheduling
The model uses a 4th-order Runge-Kutta (RK4) integration scheme. In each time step:
1.  **Compute Derivatives:** The rates of change for all 8 variables are calculated based on coupled ODEs.
2.  **Intervention Application:** Pharmacological effects (Rapamycin, NMN, Senolytics, Yamanaka factors, Transplantation) modify the derivative functions.
3.  **Update State:** The RK4 solver updates the 8D vector.
4.  **Homeostasis Check:** Copy number homeostasis logic adjusts replication rates to maintain total N.

---

## II. DESIGN CONCEPTS

### 4. Basic Principles
*   **The Heteroplasmy Cliff:** The central catastrophe where ATP production collapses nonlinearly once deletion heteroplasmy exceeds ~70%.
*   **ROS-Damage Vicious Cycle:** mtDNA damage increases ROS, which in turn accelerates mtDNA damage.
*   **Clonal Expansion:** Deletions replicate faster than wild-type mtDNA due to their smaller size (replication advantage).

### 5. Emergence
*   **Energy Failure:** ATP collapse emerges from the coupling of N_deletion expansion and respiratory chain assembly failure.
*   **Bistability:** Past the cliff, the system enters a stable "low-energy" state that is difficult to reverse without state-restoration (transplantation).

### 6. Adaptation
*   **Mitophagy:** The cell selectively removes damaged mitochondria when ATP is sufficient.
*   **Biogenesis:** Total mtDNA replication increases in response to low copy number or high energy demand (exercise).

### 7. Objectives
*   The system does not have explicit agent goals but strives for **Metabolic Homeostasis**.
*   The *optimizer* uses a fitness function: `Fitness = Avg_ATP - (Final_Heteroplasmy * 10)`.

---

## III. DETAILS

### 8. Initialization
Individuals are initialized based on a "Patient Profile" (Age, current heteroplasmy, ROS baseline). Defaults are based on Chapter II of *How to Live Much Longer* (Cramer, 2026).

### 9. Input Data
The model is grounded in biological constants derived from the Cramer (2026) manuscript, including doubling times, ROS production rates, and ETC assembly thresholds.

### 10. Submodels
*   **Mitochondrial Transplantation:** Models the "Mitlet" protocol (displacement of damaged copies by healthy donors).
*   **Yamanaka Reprogramming:** Models partial epigenetic reset gated by ATP availability.
*   **CD38 Suppression:** Models the degradation of NMN/NR supplements by aging-related enzymes.
