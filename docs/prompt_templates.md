# prompt_templates

Diegetic prompt builder for LLM-mediated mitochondrial intervention design.

---

## Overview

Provides three prompt styles for asking local LLMs (via Ollama) to generate 12D intervention+patient parameter vectors. Each style presents the same information — parameter names, valid ranges, clinical scenario — but frames it differently to test Zimmerman's (2025) findings on LLM meaning construction.

The three styles allow A/B testing of how prompt framing affects parameter quality:

| Style | How Parameters Are Presented | Thesis Basis |
|-------|------------------------------|-------------|
| **Numeric** (`OFFER_NUMERIC`) | Raw parameter names with (min, max) ranges | Supradiegetic baseline — the form LLMs process worst |
| **Diegetic** (`OFFER_DIEGETIC`) | Clinical narrative ("How aggressively should we clear damaged mitochondria?") | §2.2.3: LLMs construct meaning from diegetic content |
| **Contrastive** (`OFFER_CONTRASTIVE`) | Two opposing doctors bracket the problem | §4.7.6: TALOT/OTTITT meaning-from-contrast |

All three styles include anti-flattening measures: few-shot examples that demonstrate the different scales of patient parameters (age 20–90 vs. dose 0.0–1.0), reducing the collapse of qualitative distinctions that Zimmerman identifies in §3.5.3.

---

## Prompt Templates

### `OFFER_NUMERIC`

Straightforward parameter request listing all 12 parameters with their ranges. Includes two few-shot examples (young prevention case + near-cliff emergency) to anchor the LLM's output format and reduce flattening.

### `OFFER_DIEGETIC`

Narrative prompt that frames each parameter as a clinical judgment call:
- "How aggressively should we clear damaged mitochondria?" instead of "rapamycin_dose: (0, 1)"
- Options given as qualitative levels: "none / minimal / moderate / substantial / aggressive / maximum"
- Mapping table at the end converts words back to numbers

Includes the same two few-shot examples as the numeric prompt, with clinical reasoning added.

### `OFFER_CONTRASTIVE`

Two-expert debate format:
- **Dr. Cautious**: "Prioritizes safety, minimal intervention, preserving energy reserves."
- **Dr. Bold**: "Believes aging is an emergency requiring maximum intervention."

Produces TWO 12D vectors that bracket the solution space. The patient characterization is shared (both doctors assess the same patient).

### `CHARACTER_NUMERIC` / `CHARACTER_DIEGETIC`

Specialized prompts for the character seed experiment. Given a fictional character (e.g., "Luke Skywalker from Star Wars"), design a protocol that maps from the character's traits to a clinical profile.

### `OEIS_NUMERIC` / `OEIS_DIEGETIC`

Specialized prompts for the OEIS seed experiment. Given a mathematical integer sequence, let its growth pattern inspire an intervention protocol.

### `CONFIRMATION_PROMPT`

Shared across all styles. Given simulation results (heteroplasmy trajectory, ATP trajectory, intervention used), the confirmation-wave LLM evaluates clinical resonance and trajectory plausibility.

---

## Usage

```python
from prompt_templates import PROMPT_STYLES, get_prompt

# Select a style
offer = get_prompt("diegetic", "offer")
prompt = offer.format(scenario="65-year-old with early cognitive decline")

# Or access directly
from prompt_templates import OFFER_CONTRASTIVE
prompt = OFFER_CONTRASTIVE.format(scenario="...")
```

The `--style` flag in `tiqm_experiment.py` selects which prompt style to use at runtime.

---

## Biological Constants in Prompts

The prompts embed key biological constraints from Cramer (2026):

- Heteroplasmy cliff at ~70% (Rossignol 2003; Cramer Ch. V.K p.66)
- Yamanaka costs 3-5 MU ATP (Cramer Ch. VIII.A Table 3 p.100)
- CD38 destroys low-dose NMN/NR (Cramer Ch. VI.A.3 p.73)
- Transplant is the only true rejuvenation (Cramer Ch. VIII.G pp.104-107)

These are stated explicitly in the prompts so the LLM can reason about treatment tradeoffs.

---

## Reference

Zimmerman, J.W. (2025). PhD dissertation, University of Vermont. §2.2.3 (diegetic/supradiegetic), §3.5.3 (flattening), §4.7.6 (TALOT/OTTITT).

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
