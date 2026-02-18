# kcramer_bridge

Domain-specific K-Cramer Toolkit integration for mitochondrial resilience analysis.

---

## Overview

`kcramer_bridge.py` maps the generic scenario primitives from the
`kcramer` namespace (provided by `cramer-toolkit`) to biologically meaningful
stress conditions for this model.
It enables protocol robustness/regret/vulnerability analysis under 25 stress
scenarios spanning inflammation, NAD depletion, vulnerability, demand, aging,
and combined multi-factor stress.

---

## Key Objects

### `INFLAMMATION_SCENARIOS`, `NAD_SCENARIOS`, `VULNERABILITY_SCENARIOS`, `DEMAND_SCENARIOS`, `AGING_SCENARIOS`, `COMBINED_SCENARIOS`

Named `ScenarioSet` banks representing biological stress families.

### `ALL_STRESS_SCENARIOS`

Concatenation of all stress banks (25 total scenarios).

### `PROTOCOLS`

Reference intervention protocol bank:

- `no_treatment`
- `conservative`
- `moderate`
- `aggressive`
- `transplant_focused`

---

## Key Functions

### `run_resilience_analysis(sim, protocols, scenarios, output_key) -> dict`

Runs protocol cross-product under selected scenarios and returns consolidated
resilience report (`robustness`, `regret`, `vulnerability`, `rankings`).

### `run_vulnerability_analysis(sim, protocol, scenarios, output_key) -> list[dict]`

Runs one protocol across scenarios and returns sorted vulnerability impacts
(worst first).

### `run_scenario_comparison(analysis_fn, sim, scenarios, extract, **kwargs) -> dict`

Applies arbitrary analysis function under each scenario via
`ScenarioSimulator`/`scenario_compare`.

---

## CLI Runner

`kcramer_tools_runner.py` provides an operational CLI over these integrations:

```bash
python kcramer_tools_runner.py --mode resilience
python kcramer_tools_runner.py --mode vulnerability --protocol moderate
python kcramer_tools_runner.py --mode compare --output-key final_atp
```

Each run writes a timestamped artifact JSON in `artifacts/`.

---

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
