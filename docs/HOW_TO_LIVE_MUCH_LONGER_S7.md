# Simulation Experiments (Supplement S7): How to Live Much Longer

**Standardized documentation of simulation experiments for the Mitochondrial Aging Simulator.**

---

## 1. Experimental Goals
The goal of these experiments was to validate the "Heteroplasmy Cliff" hypothesis and test the efficacy of the six intervention mechanisms (Rapamycin, NMN, Senolytics, Yamanaka factors, Exercise, and Transplantation) in reversing mitochondrial decay.

## 2. Design of Experiments (DoE)

### 2.1 Factors and Parameters
*   **Intervention Doses (6):** Continuous levels [0.0, 1.0].
*   **Patient Profile (6):** Age, initial heteroplasmy, ROS production rate, genetic vulnerability, bio-age acceleration, and exercise intensity.

### 2.2 Sampling Strategy
We used **Sobol Sensitivity Analysis** (via `hyper_sobol.py`) to explore the 45-dimensional CA rule space and identify the most influential biological constants. Policy testing was conducted using a 4-scenario comparison (Baseline, Standard Rejuvenation, High-Intensity, and Full 'Cramer' Protocol).

## 3. Performance Measures (Outputs)
*   **ATP Stability:** Duration of time production remains above 0.8 MU/day.
*   **Heteroplasmy Reset:** Final heteroplasmy level at Year 30 vs. Baseline.
*   **Survival Probability:** Calculated based on ATP collapse and heteroplasmy threshold crossings.

## 4. Replication and Analysis
Each scenario was run through the RK4 solver with 100 Monte Carlo variants to assess robustness under parameter uncertainty. Results were analyzed using the **KCramer Precision Medicine Protocol** to generate patient-specific recommendations.
