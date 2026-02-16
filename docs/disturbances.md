# disturbances

Disturbance framework for mitochondrial resilience analysis.

---

## Overview

Models acute biological stressors that perturb the 7-variable ODE system, enabling measurement of cellular resilience. Four concrete disturbance types — ionizing radiation, toxin exposure, chemotherapy, and inflammation — implement a two-hook pattern (`modify_state` for one-time impulse, `modify_params` for sustained parameter modification) on the abstract `Disturbance` base class. The framework treats the mitochondrial network as an ecosystem subject to shocks, following agroecology principles where resilience (not just steady-state health) is the key indicator.

---

## Disturbance Architecture

### `Disturbance` (abstract base class)

All disturbances share four attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable disturbance name |
| `start_year` | `float` | Time (years from sim start) when disturbance begins |
| `duration` | `float` | How long the disturbance is active (years) |
| `magnitude` | `float` | Severity on [0, 1] scale (clipped) |

Two abstract hooks define the perturbation:

- **`modify_state(state, t) -> ndarray`** — One-time impulse applied to the 7-element state vector at onset. Called once when `t` first enters the active window. Directly alters `N_healthy`, `N_damaged`, `ATP`, `ROS`, `NAD`, `Senescent_fraction`, and/or `Membrane_potential`.

- **`modify_params(intervention, patient, t) -> (dict, dict)`** — Sustained parameter modification during the active window. Called at every timestep while `is_active(t)` returns `True`. Receives copies of intervention and patient dicts; returns modified copies.

---

## Disturbance Types

### `IonizingRadiation`

Ionizing radiation burst: damages mtDNA directly, produces ROS.

**Biological basis:** Cramer Ch. II.H p.14 — ROS-damage vicious cycle; radiation directly causes double-strand breaks in mtDNA.

**Defaults:** `start_year=5.0`, `duration=1.0`, `magnitude=0.5`

**State modifications at onset (`modify_state`):**

| State Variable | Modification | Effect at magnitude=1.0 |
|----------------|-------------|--------------------------|
| `N_healthy` (0) | `-= 0.05 * magnitude * N_healthy` | Up to 5% of healthy pool converted |
| `N_damaged` (1) | `+= 0.05 * magnitude * N_healthy` | Receives transferred copies |
| `ROS` (3) | `+= 0.15 * magnitude` | Acute ROS burst (+0.15) |

**Parameter modifications during active window (`modify_params`):**

| Parameter | Modification | Effect at magnitude=1.0 |
|-----------|-------------|--------------------------|
| `genetic_vulnerability` | `*= (1.0 + 0.5 * magnitude)` | Up to 1.5x vulnerability |
| `inflammation_level` | `+= 0.2 * magnitude` (capped at 1.0) | Elevated inflammation |

---

### `ToxinExposure`

Environmental toxin exposure: damages membrane potential, boosts ROS.

**Biological basis:** Cramer Ch. IV pp.46-47 (membrane potential), Ch. VI.B p.75 (low membrane potential triggers mitophagy via PINK1/Parkin).

**Defaults:** `start_year=5.0`, `duration=2.0`, `magnitude=0.5`

**State modifications at onset (`modify_state`):**

| State Variable | Modification | Effect at magnitude=1.0 |
|----------------|-------------|--------------------------|
| `Membrane_potential` (6) | `*= (1.0 - 0.2 * magnitude)` | Up to 20% drop |
| `NAD` (4) | `*= (1.0 - 0.1 * magnitude)` | Up to 10% depletion (PARPs activated) |
| `ROS` (3) | `+= 0.1 * magnitude` | Mitochondrial stress ROS (+0.1) |

**Parameter modifications during active window (`modify_params`):**

| Parameter | Modification | Effect at magnitude=1.0 |
|-----------|-------------|--------------------------|
| `inflammation_level` | `+= 0.3 * magnitude` (capped at 1.0) | Elevated inflammation |
| `metabolic_demand` | `+= 0.2 * magnitude` (capped at 2.0) | Detoxification energy cost |

---

### `ChemotherapyBurst`

Cytotoxic chemotherapy: massive ROS, NAD depletion, mtDNA damage. The most severe disturbance type — collaterally damages mitochondria along with cancer cells.

**Biological basis:** Cramer Ch. VII (cellular damage mechanisms); clinical scenario `post_chemo_55` in `constants.py`.

**Defaults:** `start_year=5.0`, `duration=0.5`, `magnitude=0.8`

**State modifications at onset (`modify_state`):**

| State Variable | Modification | Effect at magnitude=1.0 |
|----------------|-------------|--------------------------|
| `N_healthy` (0) | `-= 0.1 * magnitude * N_healthy` | Up to 10% of healthy pool converted |
| `N_damaged` (1) | `+= 0.1 * magnitude * N_healthy` | Receives transferred copies |
| `ROS` (3) | `+= 0.3 * magnitude` | Massive ROS burst (+0.3) |
| `NAD` (4) | `*= (1.0 - 0.25 * magnitude)` | Up to 25% NAD crash |
| `Membrane_potential` (6) | `*= (1.0 - 0.15 * magnitude)` | Up to 15% collapse |

**Parameter modifications during active window (`modify_params`):**

| Parameter | Modification | Effect at magnitude=1.0 |
|-----------|-------------|--------------------------|
| `genetic_vulnerability` | `*= (1.0 + 0.8 * magnitude)` | Up to 1.8x vulnerability |
| `inflammation_level` | `+= 0.4 * magnitude` (capped at 1.0) | Severe inflammation |
| `metabolic_demand` | `+= 0.3 * magnitude` (capped at 2.0) | High metabolic burden |

---

### `InflammationBurst`

Acute systemic inflammation: infection, fever, or autoimmune flare.

**Biological basis:** Cramer Ch. VII.A pp.89-90 (SASP — senescence-associated secretory phenotype), Ch. VIII.F p.103 (senescent cells use approximately 2x energy).

**Defaults:** `start_year=5.0`, `duration=0.5`, `magnitude=0.5`

**State modifications at onset (`modify_state`):**

| State Variable | Modification | Effect at magnitude=1.0 |
|----------------|-------------|--------------------------|
| `ROS` (3) | `+= 0.1 * magnitude` | ROS elevation (+0.1) |
| `Senescent_fraction` (5) | `+= 0.02 * magnitude` (capped at 1.0) | Accelerated senescence (+0.02) |

**Parameter modifications during active window (`modify_params`):**

| Parameter | Modification | Effect at magnitude=1.0 |
|-----------|-------------|--------------------------|
| `inflammation_level` | `+= 0.5 * magnitude` (capped at 1.0) | Major inflammation spike |
| `metabolic_demand` | `+= 0.15 * magnitude` (capped at 2.0) | Immune response energy cost |

---

## Key Functions

### `simulate_with_disturbances(intervention, patient, disturbances, sim_years, dt) -> dict`

Run the mitochondrial aging simulation with disturbance events injected at the appropriate times. Wraps the RK4 integration loop from `simulator.py`, applying state impulses at onset and parameter modifications at each timestep during the active window.

**Args:**
- `intervention` — Dict of 6 intervention params (defaults to no treatment)
- `patient` — Dict of 6 patient params (defaults to typical 70-year-old)
- `disturbances` — List of `Disturbance` objects to apply
- `sim_years` — Override simulation horizon (default: `constants.SIM_YEARS`)
- `dt` — Override timestep (default: `constants.DT`)

**Returns dict with:**

| Key | Type | Description |
|-----|------|-------------|
| `time` | `ndarray` | Time points (n_steps+1,) |
| `states` | `ndarray` | State trajectories (n_steps+1, 7) |
| `heteroplasmy` | `ndarray` | Heteroplasmy at each step |
| `intervention` | `dict` | Intervention dict used |
| `patient` | `dict` | Patient dict used |
| `disturbances` | `list[dict]` | Disturbance event metadata (name, start, duration, magnitude) |
| `shock_times` | `list[tuple]` | `(start, end)` tuples for each disturbance |

**Integration loop details:**
1. State impulses are applied once per disturbance via `_applied_impulse` flag
2. State is clamped non-negative after impulse; senescent fraction capped at 1.0
3. Parameter modifications are applied at every timestep during active window
4. Multiple disturbances compose: all active modifications stack
5. RK4 integration uses `derivatives()` from `simulator.py`

---

## Usage

```python
from disturbances import (
    IonizingRadiation, ToxinExposure, ChemotherapyBurst,
    InflammationBurst, simulate_with_disturbances,
)
from simulator import simulate

# Single shock
shock = IonizingRadiation(start_year=10.0, magnitude=0.8)
result = simulate_with_disturbances(disturbances=[shock])

# Multi-shock scenario
shocks = [
    IonizingRadiation(start_year=5.0, magnitude=0.6),
    ChemotherapyBurst(start_year=15.0, magnitude=0.7),
]
result = simulate_with_disturbances(disturbances=shocks)

# Compare with baseline
baseline = simulate()
delta_atp = result["states"][-1, 2] - baseline["states"][-1, 2]
```

---

## Reference

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag. Ch. II.H p.14, Ch. IV pp.46-47, Ch. VI.B p.75, Ch. VII pp.89-92, Ch. VIII.F p.103.
