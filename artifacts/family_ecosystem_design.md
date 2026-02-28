# Design Doc: Family Ecosystem Reporting Module (FERM)

**Status:** Draft / Technical Specification  
**Project:** How to Live Much Longer - Mitochondrial Simulation Suite  
**Date:** February 2026

---

## 1. Overview
The Family Ecosystem Reporting Module (FERM) is a specialized analytics layer designed to track, compare, and optimize the biological trajectories of seven individuals across three interconnected research households: **Westport**, **Seattle**, and **Champlain**.

The module moves beyond individual simulation to model the family as a **Shared Biological Information Ecosystem**, quantifying the impact of environment, genetics (EDS Gradient), and legacy interventions (Mitrix Rescue).

## 2. Core Architecture

### 2.1 The Global Identity Registry
FERM uses a unified naming registry to ensure consistent tracking across local documents and simulation runs:
- **Westport:** Kathryn (64), Peter (28)
- **Seattle:** John Jr. (91), John III (62), Selena Shea (26)
- **Champlain:** Ratio (23), Jasper (24)

### 2.2 The "M-Age" Calculation Engine
FERM implements a standardized **Mitochondrial Age (M-Age)** metric. This provides a "Headline Number" for family health by mapping current heteroplasmy and ATP levels back to a reference aging curve.
- **EDS Weighting:** Automatically applies the clinical drag gradient (1.15x to 1.45x) to the M-Age velocity.

### 2.3 Environmental ROI Engine
Formalizes the "Full Farm" and "Artist Ecosystem" findings:
- **EE Multiplier:** Applies the 1.5x biogenesis and 0.15 mitophagy boosts found in the Enriched Environment literature.
- **Resilience Buffer:** Quantifies the "Additional Years" added to the mitochondrial runway by the household environment.

## 3. Key Reporting Modules

### 3.1 The Family Maturity Matrix
A comparative table showing:
- **Chrono Age vs. M-Age** (Highlighting the 30-year "EDS Maturity Gap").
- **Energy Reserve (ATP)** (Color-coded Green/Yellow/Red for crisis monitoring).
- **Cognitive Buffer (CR)** (Tracking the Artist/Scholar "Software" resilience).

### 3.2 The Legacy Rescue Dashboard
A dedicated view for the Seattle Rescue project:
- **Donor Purity Tracking:** Monitors Selena and Ratio as potential mitochondrial donors.
- **Transplant Efficacy:** Predicts the % reduction in John Jr.'s heteroplasmy.
- **ATP Kickstart Monitor:** Tracks the post-transplant "Energy Engine" status.

### 3.3 The "Metabolic Storm" Early Warning
An automated stress-test reporter that predicts how each family member will handle:
- **Viral Shocks** (e.g., Preschool Spike).
- **Sleep Crisis** (Seizure threshold monitoring for Ratio).
- **Creative Exhaustion** (The "Gallery Crunch").

## 4. Phase 10: Real-Time Telemetry Bridge
FERM v2.0 introduces "Closed-Loop" simulation by piping live data from wearables and medical devices directly into the ODE solver. This eliminates the "Estimation Gap" and allows for proactive interventions.

### 4.1 Data-to-Parameter Mapping
- **HRV (Heart Rate Variability):** Maps to `inflammation_level`. High HRV = Low Inflammation.
- **Deep Sleep (min):** Maps to `glymphatic_clearance_factor`. Quantifies the nightly ROS washout.
- **Respiratory Rate:** Maps to `lung_o2` and `metabolic_demand`.
- **CGM (Glucose):** Maps to `insulin_sensitivity` and `fructose_penalty`.

### 4.2 Candidate Device Ecosystem
#### Category A: Allostatic Tracking (Daily Load)
- **Oura Ring Gen 3:** Optimal for sleep architecture and recovery indexing.
- **Whoop 4.0:** High-frequency strain and autonomic balance.
- **Apple Watch Ultra:** VO2 Max and lifestyle integration.

#### Category B: Metabolic & Seizure Monitoring
- **Dexcom G7 (CGM):** Continuous glucose/insulin-response tracking.
- **Embrace2 (Empatica):** Clinical-grade seizure detection (Essential for Ratio).
- **Lumen:** Real-time respiratory exchange ratio (Metabolic flexibility).

#### Category C: Mitochondrial Ground Truth
- **InsideTracker:** Bi-monthly blood biomarkers (hs-CRP, Cortisol, Vitamin D).
- **Viome:** Microbiome sequencing to validate "Full Farm" synergy.
- **Core Temp Sensor:** Real-time monitoring of heat-shock protein (HSP) induction during saunas.

## 5. Technical Implementation
- **Implementation Language:** Python (utilizing `unified_brain_model.py`).
- **Data Ingestion:** REST API connectors for Oura/Apple Health/Dexcom.
- **Dose-Response Correction:** The simulator uses telemetry to self-calibrate internal constants based on real-world recovery velocity.
- **Security:** Fully git-ignored; zero exposure of private familial data to public repositories.

---

## 5. Success Metrics
- **Quantifiable Runway:** FERM should accurately predict the gain/loss of "Mitochondrial Years" for every intervention change.
- **Inter-Household Synergy:** Identifies "Social protocols" (e.g., teaching or collaborative art) that provide the highest ROI across the group.
