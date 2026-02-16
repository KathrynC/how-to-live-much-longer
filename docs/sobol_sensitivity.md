# sobol_sensitivity

Global sensitivity analysis via Saltelli sampling and Sobol indices.

---

## Overview

Identifies which of the 12 parameters (6 intervention + 6 patient) actually drive simulation outcomes, and which parameter *interactions* matter. Uses Saltelli's quasi-random sampling scheme with Jansen estimators for first-order (S1) and total-order (ST) Sobol indices.

Unlike one-at-a-time perturbation (`perturbation_probing.py`), Sobol analysis captures the full interaction structure of the parameter space. A parameter with low S1 but high ST is one that matters only through its interactions with other parameters.

---

## Method

### Saltelli Sampling

For D = 12 parameters and N base samples:

1. Generate two independent quasi-random matrices A and B, each N × D
2. For each parameter i, create matrix AB_i by replacing column i of A with column i of B
3. Total samples: N × (2D + 2) = N × 26

With N = 256 (default), this produces **6,656 simulations** (~2-3 min).

### Sobol Indices

For each output variable (heteroplasmy_final, ATP_final):

| Index | Formula | Meaning |
|-------|---------|---------|
| **S1** (first-order) | Var[E(Y\|Xi)] / Var(Y) | Fraction of output variance explained by parameter i alone |
| **ST** (total-order) | 1 - Var[E(Y\|X~i)] / Var(Y) | Fraction due to parameter i and ALL its interactions |
| **ST - S1** | (computed) | Interaction contribution — variance explained only through joint effects |

### Interpretation

- **S1 ≈ ST:** Parameter acts independently (no significant interactions)
- **S1 << ST:** Parameter matters mostly through interactions with others
- **S1 ≈ 0, ST ≈ 0:** Parameter doesn't matter (safe to fix at any value)
- **ΣS1 << 1:** Strong interaction effects dominate the system

---

## Usage

```python
# As a script
python sobol_sensitivity.py
# Outputs: artifacts/sobol_sensitivity.json

# Programmatically
from sobol_sensitivity import run_sobol_analysis
result = run_sobol_analysis(n_base=256, seed=42)

print(result["het_final"]["S1"])
# {"rapamycin_dose": 0.08, "baseline_heteroplasmy": 0.45, ...}

print(result["het_final"]["ST"])
# {"baseline_heteroplasmy": 0.52, "genetic_vulnerability": 0.15, ...}

print(result["rankings"])
# Parameters sorted by total-order influence
```

---

## Output

```json
{
  "n_base": 256,
  "n_total_sims": 6656,
  "het_final": {
    "S1": {"baseline_heteroplasmy": 0.45, "rapamycin_dose": 0.08, ...},
    "ST": {"baseline_heteroplasmy": 0.52, "genetic_vulnerability": 0.15, ...}
  },
  "atp_final": {
    "S1": {"baseline_heteroplasmy": 0.38, ...},
    "ST": {...}
  },
  "rankings": {
    "het_final": ["baseline_heteroplasmy", "genetic_vulnerability", ...],
    "atp_final": [...]
  }
}
```

---

## Implementation

Pure numpy — no scipy, no SALib. The Saltelli sampling and Jansen estimators are implemented directly, matching the parent project's numpy-only constraint.

---

## Reference

- Saltelli, A. (2002). "Making best use of model evaluations to compute sensitivity indices." *Computer Physics Communications*, 145(2), 280-297.
- Jansen, M.J.W. (1999). "Analysis of variance designs for model output." *Computer Physics Communications*, 117(1-2), 35-43.
