# The TIQM Pipeline: Transactional Interpretation of Mitochondrial Simulation

The TIQM pipeline in this repository implements the **Transactional Interpretation of Quantum Mechanics** framework, adapted from the parent **Evolutionary-Robotics** project. It is used to design and evaluate mitochondrial aging interventions using Large Language Models (LLMs) and Vision-Language Models (VLMs).

The core experiment loop is implemented in `tiqm_experiment.py`.

---

## Three-Phase Pipeline

```text
Clinical Scenario Seed
    ↓
[1] Offer Wave         → LLM generates a 12D intervention + patient vector (temp=0.7)
    ↓
[2] Simulation         → RK4 ODE solver runs a 30-year trajectory + 4-pillar analytics
    ↓
[3] Confirmation Wave  → A DIFFERENT LLM/VLM rates resonance (temp=0.3)
    ↓
Artifact: intervention, analytics, resonance scores
```

### Phase 1: The Offer Wave
An LLM (via local Ollama) receives a clinical scenario seed (e.g., *"70-year-old with cognitive decline"*) and generates a **12-dimensional parameter vector**:
*   **6 Intervention Parameters:** Rapamycin dose, NAD+ supplementation, senolytic dose, Yamanaka intensity, transplant rate, exercise level.
*   **6 Patient Parameters:** Age, baseline heteroplasmy, baseline NAD+, genetic vulnerability, inflammation, metabolic demand.

The raw LLM output is parsed from JSON and snapped to a discrete grid via `snap_all()`.

### Phase 2: Simulation & Analytics
The snapped parameters are fed into the **RK4 ODE simulator** (`simulator.py`), which models 8 mitochondrial state variables over 30 simulated years. The output trajectory is analyzed through the **4-pillar analytics framework** (`analytics.py`):
1.  **Energy:** ATP production and crisis proximity.
2.  **Damage:** Heteroplasmy progression and ROS-correlation.
3.  **Dynamics:** Membrane potential stability and CV.
4.  **Intervention:** Benefit-cost ratios and benefit vs. no-treatment.

### Phase 3: The Confirmation Wave
A different model (to prevent self-confirmation bias) evaluates the results and rates **"Resonance"**—how well the generated protocol matches the clinical scenario. 

**Resonance Metrics (0.0 to 1.0):**
*   **`resonance_behavior`**: Does the protocol match the clinical scenario?
*   **`resonance_trajectory`**: Is the trajectory physiologically plausible?
*   **`resonance_symmathesy`**: Quality of mutual learning/adaptation between intervention and patient.

---

## TIQM Concept Mapping (Robotics → Mitochondrial)

| TIQM Concept | Robotics Project | This Project |
| :--- | :--- | :--- |
| **Offer Wave** | LLM generates 12D weight+physics vector | LLM generates 12D intervention+patient vector |
| **Simulation** | PyBullet 3-link robot locomotion | RK4 ODE of 8 mitochondrial state variables |
| **Confirmation Wave** | VLM evaluates locomotion behavior | VLM evaluates cellular trajectory |
| **Resonance** | Semantic match to character seed | Clinical match to patient scenario |
| **The "Cliff"** | Behavioral cliffs in weight space | Heteroplasmy cliff at ~70% damaged mtDNA |

---

## CLI Usage & Run Modes

The pipeline supports multiple prompt styles (selected via `--style`):
*   **`numeric`**: Direct parameter specification (default).
*   **`diegetic`**: Zimmerman-informed narrative prompts.
*   **`contrastive`**: "Dr. Cautious" vs. "Dr. Bold" dual protocols.

**Execution Flags:**
*   `(none)`: Runs all 10 clinical scenarios.
*   `--single`: Runs the first scenario only (quick test).
*   `--contrastive`: Generates cautious and bold protocols for each scenario.

---

## Output Artifacts
*   `output/tiqm_{seed_id}.json`: Per-scenario results (offer + confirmation + analytics).
*   `output/tiqm_summary.json`: Combined results with resonance statistics across all scenarios.
