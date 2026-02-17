# resilience_metrics

Ecological resilience indicators for mitochondrial disturbance response.

---

## Overview

Computes four core resilience metrics from shocked vs baseline simulation trajectories: resistance, recovery time, regime retention, and elasticity. Inspired by agroecology — a healthy ecosystem resists perturbation (low peak deviation), recovers quickly, and returns to its original regime rather than flipping to a degraded state. The heteroplasmy cliff at 0.70 serves as the regime boundary: crossing it is an irreversible regime shift analogous to an ecological tipping point (Scheffer 2009).

---

## Resilience Metrics

### Resistance

How much the system deviates from baseline under shock.

Measured over a window from shock onset to shock end + 5 years. Three sub-metrics:

| Metric | Description |
|--------|-------------|
| `peak_deviation` | Maximum absolute deviation from baseline in the window |
| `mean_deviation` | Average absolute deviation from baseline in the window |
| `relative_peak_deviation` | Peak deviation divided by baseline mean (dimensionless) |

Lower values indicate greater resistance. A system that barely deflects under a severe shock is highly resistant.

### Recovery Time

How quickly the shocked trajectory returns within epsilon of the baseline.

Recovery is defined as `|shocked - baseline| < epsilon * |baseline|` for at least 10 consecutive timesteps (sustained recovery, not a transient crossing). Default `epsilon = 0.05` (within 5% of baseline). Returns `np.inf` if the system never recovers within the simulation horizon.

The 10-step sustained-recovery check prevents false positives from oscillatory trajectories that briefly touch the baseline.

### Regime Retention

Whether the system returns to the same dynamical regime after shock.

The heteroplasmy cliff (`HETEROPLASMY_CLIFF = 0.70`) is the regime boundary. The metric compares the final regime (mean of last 10% of simulation) of shocked vs baseline trajectories:

| Indicator | Description |
|-----------|-------------|
| `regime_retained` | `True` if shocked and baseline end on the same side of the cliff |
| `final_het_shocked` | Mean heteroplasmy in the tail of the shocked trajectory |
| `final_het_baseline` | Mean heteroplasmy in the tail of the baseline trajectory |
| `het_gap` | `final_het_shocked - final_het_baseline` (positive = worse) |
| `ever_crossed_cliff` | `True` if the shocked trajectory crossed the cliff at any point post-shock, starting from below |

Regime shift (retained = `False`) is the most critical failure mode: it indicates the disturbance pushed the system past the heteroplasmy cliff into irreversible ATP collapse.

### Elasticity

Normalized rate of recovery in the first 2 years after shock ends.

Computed as the negative slope of `|shocked - baseline|` over the 2-year post-shock window (linear fit via `np.polyfit`). Positive elasticity means the deviation is decreasing (recovering); negative means the system is diverging further from baseline.

---

## Composite Resilience Score

### `compute_resilience(shocked_result, baseline_result, state_idx, epsilon) -> dict`

Computes all four metrics and combines them into a single `summary_score` on [0, 1] (0 = fragile, 1 = maximally resilient).

**Component score transformations:**

| Component | Transformation | Range |
|-----------|---------------|-------|
| `resistance_score` | `max(0, 1 - relative_peak_deviation)` | [0, 1] |
| `recovery_score` | `exp(-recovery_time / 5.0)` | (0, 1]; 0 if never recovered |
| `regime_score` | 1 if retained, 0 if shifted | {0, 1} |
| `elasticity_score` | `1 / (1 + exp(-10 * elasticity))` | (0, 1) |

**Composite score weights:**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Resistance | 0.25 | Important but less critical than recovery |
| Recovery | 0.30 | Speed of return is clinically actionable |
| Regime retention | 0.30 | Regime shift is catastrophic (irreversible) |
| Elasticity | 0.15 | Early recovery rate is informative but noisy |

`summary_score = 0.25 * resistance + 0.30 * recovery + 0.30 * regime + 0.15 * elasticity`

**Returns dict with:**

| Key | Type | Description |
|-----|------|-------------|
| `resistance` | `dict` | Peak, mean, and relative peak deviation |
| `recovery_time_years` | `float` | Years to recover (999.0 if never) |
| `regime` | `dict` | Retention indicators and heteroplasmy gaps |
| `elasticity` | `float` | Recovery slope (positive = recovering) |
| `summary_score` | `float` | Composite score [0, 1] |
| `component_scores` | `dict` | Individual transformed scores (resistance, recovery, regime, elasticity) |

---

## Key Functions

### `compute_resistance(shocked_states, baseline_states, time, shock_start, shock_end, state_idx) -> dict`

Measure peak and mean deviation from baseline during and after shock. Default `state_idx=2` (ATP). Analysis window: shock onset to shock end + 5 years.

### `compute_recovery_time(shocked_states, baseline_states, time, shock_end, state_idx, epsilon) -> float`

Time in years for shocked trajectory to return within epsilon of baseline. Returns `np.inf` if never recovered. Default `epsilon=0.05`.

### `compute_regime_retention(shocked_het, baseline_het, time, shock_start) -> dict`

Determine whether the system crosses the heteroplasmy cliff regime boundary. Compares pre-shock regime, final regime, and tracks any transient cliff crossing.

### `compute_elasticity(shocked_states, baseline_states, time, shock_end, state_idx) -> float`

Initial rate of recovery (negative slope of deviation) in the first 2 years after shock ends. Positive = recovering, negative = diverging.

### `compute_resilience(shocked_result, baseline_result, state_idx, epsilon) -> dict`

Compute all four metrics plus composite score from simulation result dicts. Accepts output of `simulate_with_disturbances()` and `simulate()`.

### `compute_resilience_sweep(disturbance_class, magnitudes, intervention, patient, start_year, state_idx) -> list[dict]`

Sweep disturbance magnitude and compute resilience at each level. Instantiates the given `Disturbance` subclass at each magnitude, runs `simulate_with_disturbances()`, and returns a list of resilience metric dicts augmented with `magnitude` and `disturbance` fields.

---

## Usage

```python
from simulator import simulate
from disturbances import simulate_with_disturbances, IonizingRadiation
from resilience_metrics import compute_resilience, compute_resilience_sweep

# Single resilience assessment
baseline = simulate()
shocked = simulate_with_disturbances(
    disturbances=[IonizingRadiation(start_year=10.0, magnitude=0.8)]
)
metrics = compute_resilience(shocked, baseline)
print(f"Resilience score: {metrics['summary_score']:.3f}")
print(f"Regime retained:  {metrics['regime']['regime_retained']}")

# Magnitude sweep
sweep = compute_resilience_sweep(
    IonizingRadiation, [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)
for s in sweep:
    print(f"mag={s['magnitude']:.1f}: score={s['summary_score']:.3f}")
```

---

## Reference

Cramer, J.G. (forthcoming from Springer Verlag in 2026). *How to Live Much Longer*. Ch. V.K p.66 (heteroplasmy cliff).

Scheffer, M. (2009). *Critical Transitions in Nature and Society*. Princeton University Press (regime shift theory).

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer Verlag in 2026**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
