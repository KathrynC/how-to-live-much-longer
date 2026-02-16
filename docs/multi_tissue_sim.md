# multi_tissue_sim

D3: Coupled multi-tissue simulator — brain + muscle + cardiac.

---

## Overview

Wraps the existing `derivatives()` function to simulate 3 tissues coupled by shared NAD+ pool, systemic inflammation, and cardiac blood flow. State vector: 7 × 3 = 21 states. Does NOT modify `simulator.py`.

---

## Tissue Model

| Tissue | `metabolic_demand` | `ros_sensitivity` | `biogenesis_rate` | Key Property |
|--------|-------------------|--------------------|-------------------|--------------|
| Brain | 2.0 | 1.5 | 0.7 | 20% of body's O2 for 2% of mass (Ch. IV p.46) |
| Muscle | 1.5 | 1.0 | 1.3 | Most exercise-responsive (Ch. VI.B p.76) |
| Cardiac | 1.8 | 1.2 | 0.9 | Pump for all intervention delivery |

---

## Coupling Mechanisms

### NAD+ Sharing

Global pool distributed by metabolic demand weighting. Brain draws 2× due to extreme oxidative metabolism.

### Systemic Inflammation

SASP from any tissue raises inflammation for all others. Brain SASP weighted 1.5× (neuroinflammation cascades via blood-brain barrier breakdown).

### Cardiac Blood Flow

`cardiac_ATP / BASELINE_ATP` modulates intervention delivery to ALL tissues. If heart fails, drugs can't reach targets — creating a catastrophic cascade.

---

## Experiments

### 1. Protocol Comparison (5 protocols × 4 allocations)

| Allocation Strategy | Description |
|--------------------|-------------|
| `equal` | Same doses to all tissues |
| `brain_priority` | Brain gets 1.5× dose, others proportionally reduced |
| `cardiac_first` | Cardiac gets priority to maintain blood flow |
| `worst_first` | Dynamic: prioritize whichever tissue is weakest |

### 2. Brain Allocation Sweep

10-point sweep of brain allocation fraction to find optimal brain vs systemic trade-off.

---

## Discovery Potential

- **Cardiac cascade**: Heart failure → blood flow reduction → systemic collapse
- **Worst-first benefit**: Dynamic allocation outperforms static
- **NAD+ competition**: Brain depleting shared pool for other tissues
- **Neuroinflammation coupling**: Cross-tissue SASP effects

---

## Scale

5 protocols × 4 allocations + 10 sweep = ~30 sims (each 3× computation). Estimated time: ~2 minutes.

---

## Output

- `artifacts/multi_tissue_sim.json` — Per-tissue trajectories, allocation comparison, cardiac cascade detection

---

## Reference

Cramer, J.G. (2026, forthcoming). *How to Live Much Longer*. Springer Verlag. Ch. IV pp.46-47, Ch. VI.B p.76, Ch. VII.A pp.89-92.
