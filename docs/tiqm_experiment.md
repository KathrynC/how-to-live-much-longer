# tiqm_experiment

LLM-driven mitochondrial intervention design via the TIQM pipeline.

---

## Overview

Implements the Transactional Interpretation of Quantum Mechanics (TIQM) pipeline adapted from the parent Evolutionary-Robotics project. Three-phase cycle: offer wave (LLM generates 12D vector from clinical scenario), simulation (30-year ODE), confirmation wave (different LLM evaluates trajectory). Different models for offer vs confirmation prevent self-confirmation bias (Cramer's TIQM principle).

---

## Pipeline

```
Clinical Scenario Seed
    ↓
[1] Offer wave: LLM → 12D intervention + patient vector (temp=0.7)
    ↓
[2] Simulation: RK4 ODE → 30-year trajectory + 4-pillar analytics
    ↓
[3] Confirmation wave: different LLM rates resonance (temp=0.3)
    ↓
Artifact: intervention, analytics, resonance scores
```

---

## Key Functions

### `run_experiment(seed, offer_model, confirm_model, verbose) → dict|None`

Run a single TIQM experiment for one clinical scenario seed. Returns artifact dict with intervention, patient, analytics, and confirmation resonance scores.

### `run_all_experiments(seeds, output_dir) → list[dict]`

Run TIQM experiments for all clinical scenario seeds (default: 10 from `CLINICAL_SEEDS`). Saves per-scenario JSON artifacts and a combined `tiqm_summary.json`. Prints summary resonance statistics.

---

## Prompt Styles

Selected at runtime via `--style` flag. Imported from `prompt_templates.py`.

| Style | Description |
|-------|-------------|
| `numeric` | Original direct parameter specification (default) |
| `diegetic` | Zimmerman-informed narrative prompts (§2.2.3) |
| `contrastive` | Dr. Cautious vs Dr. Bold dual protocols (§4.7.6) |

---

## Confirmation Wave

The confirmation prompt presents simulation results (ATP trajectory, heteroplasmy, cliff timing, intervention details) and asks the confirming LLM to rate:
- **resonance_behavior** (0.0–1.0): Does the protocol match the clinical scenario?
- **resonance_trajectory** (0.0–1.0): Is the trajectory physiologically plausible?

---

## CLI Modes

| Flag | Description | Scale |
|------|-------------|-------|
| (none) | All 10 clinical scenarios | ~10 trials |
| `--single` | First scenario only (quick test) | 1 trial |
| `--style diegetic` | Use narrative prompts | same |
| `--contrastive` | Generate cautious + bold protocols per scenario | 3 scenarios × 2 protocols |

---

## Output

- `output/tiqm_{seed_id}.json` — Per-scenario artifact
- `output/tiqm_summary.json` — Combined results with statistics

---

## Reference

Cramer, J.G. (forthcoming 2026). *How to Live Much Longer: The Mitochondrial DNA Connection*. Springer. ISBN 978-3-032-17740-7.

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
