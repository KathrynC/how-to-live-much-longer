# Cross-Project Gnarliness Comparison: ER vs Mitochondrial Aging

## Context

Both projects share a 12D parameter space → simulation → fitness evaluation pipeline
(the TIQM architecture). This analysis compares the topology of the two fitness
landscapes using Rudy Rucker's "gnarliness" framework, which maps to Wolfram's
four computational classes: Class 1 (fixed point), Class 2 (periodic), Class 3
(chaotic), and Class 4 (complex — the interesting edge between order and chaos).

**ER project**: 6D neural weights + 6D physics → PyBullet rigid-body simulation → DX fitness
**JGC project**: 6D intervention doses + 6D patient params → RK4 ODE of 7 state variables → ATP/heteroplasmy fitness

---

## The ER Landscape: Class 3 (Proven Fractal Chaos)

Characterized across ~15,000 simulations (atlas_cliffiness, dark_matter, gait_interpolation, perturbation_probing):

| Property | Value | Implication |
|---|---|---|
| Champion-to-champion roughness / random roughness | **0.999** | No privileged paths exist |
| Sign flip rate between adjacent samples | **0.58-0.72** | DX changes direction >half the time |
| Perpendicular / transect roughness ratio | **1.01** | Isotropic — no direction is special |
| Points with >5m cliffs (r=0.05 probe) | **95.8%** | Cliffs everywhere |
| Cliffiness correlation with behavior | **r < 0.14** | Roughness uncorrelated with quality |
| Dead zone (|DX| < 1m) | **11.8%** | But 0% frozen — all oscillate |
| Dead gait behavioral categories | **5 types** | Circlers, cancellers, spinners, rockers, others |

The ER landscape is fractal **everywhere, at every scale, in every direction, including
between champions**. Smooth weight interpolation produces chaotic DX. Contact dynamics
between rigid bodies and the ground plane — the physical source of the chaos — destroy
all gradient structure.

This is pure Class 3: maximal entropy, zero exploitable structure.

---

## The Mitochondrial Landscape: Class 2-4 Hybrid

Partially characterized from causal_surgery.json (192 switching experiments) and
perturbation_probing.json (8 probes x 24 perturbations):

| Property | Value | Implication |
|---|---|---|
| ATP coefficient of variation | **0.503** | Moderate roughness (ER would be >>1.0) |
| Dead zones (ATP < 0.2) | **19.8%** | But only for near-cliff patients |
| Healthy patient crisis rate | **0/128** | Class 2 zone: outcome predetermined |
| Near-cliff patient crisis rate | **44/64 = 68.8%** | Bistable attractor dominates |
| Intervention potency range | **17.4x** (0.04 to 0.76) | Clear hierarchy, exploitable gradient |
| Switch-time correlation (near-cliff) | **r = 0.007** | Timing irrelevant — attractor dominates |
| Robustness ratio (healthy/sick) | **7.4x** | Healthy = stable, sick = fragile |
| Hill-climbing success | **Works** | Local gradient structure exists |

---

## The Fundamental Architectural Difference

**Why the ER landscape is fractal and the mito landscape isn't:**

The ER simulator uses PyBullet contact dynamics — rigid bodies colliding with a ground
plane at 240 Hz. Contact is inherently discontinuous: a foot either touches the ground
or it doesn't. Tiny changes in joint angles can change *which timestep* contact
initiates, creating cascading phase shifts in the entire gait cycle. This is analogous
to the three-body problem: deterministic but computationally irreducible.

The mito simulator uses continuous ODEs integrated by 4th-order Runge-Kutta. The
derivatives are smooth functions (polynomials, sigmoids, products). The *only* source
of sharp nonlinearity is the heteroplasmy cliff — a sigmoid with steepness 15.0
centered at het=0.70. Away from the cliff, the landscape is smooth and differentiable.
*Near* the cliff, bistability creates a sharp boundary, but it's a 1D boundary in 12D
space, not the everywhere-fractal surface of the ER landscape.

---

## Comparative Gnarliness Map

```
Class 1          Class 2          Class 4          Class 3
(dead)           (periodic)       (complex/gnarly)  (chaotic)
  |                |                |                |
  |     Mito:      |   Mito:        |                |
  |   healthy      | near-cliff     |                |  ER: entire
  |   patients     | with bistable  |                |  6D weight
  |   (always      | cliff + some   |                |  space is
  |    survive)    | interventions  |                |  fractal
  |                |                |                |  everywhere
  |                |    Mito:       |                |
  |                |  moderate pts  |                |
  |                |  with graded   |                |
  |                |  intervention  |                |
  |                |  response      |                |
```

The mito landscape is a **patchwork of computational classes**:

- **Class 1-2 zone** (healthy patients, baseline het < 0.3): Outcome is predetermined.
  Interventions barely matter. The ODE flows to a healthy attractor regardless. ~40% of
  patient space — a vast boring plateau.

- **Class 2-4 zone** (moderate patients, het 0.3-0.6): Interventions have graded,
  monotonic effects. Transplant > rapamycin > NAD > exercise. Hill-climbing works. This
  is the "navigable terrain" where the most interesting clinical optimization lives.

- **Class 3-ish zone** (near-cliff patients, het > 0.6): Bistability creates a sharp
  boundary. 68.8% of interventions fail. Timing doesn't matter (r=0.007). The few
  interventions that work (heavy transplant) must overcome the damaged-replication
  advantage. Not truly fractal, but shares Class 3's key feature: unpredictability
  from small perturbations near the boundary.

---

## Using Rucker's Term Precisely

The ER landscape is **maximally gnarly in the wrong way** — it's Class 3 chaos that
*looks* like gnarliness but is actually just noise. There's no computation happening in
the landscape; it's a fractal hash function from weights to behavior. The gnarliness has
no depth — zoom in and it's the same fractal at every scale, carrying no more
information at high resolution than at low.

The mito landscape is **genuinely gnarly in the Ruckerian sense** — it has *structure at
the edge of predictability*. The heteroplasmy cliff creates a phase transition. The CD38
gating of NAD creates a threshold nonlinearity. The Yamanaka energy cost creates a
risk/reward tradeoff. The multi-tissue coupling (D3) creates resource competition. These
are all the ingredients of Class 4 complexity: local rules that generate unpredictable
but structured global behavior.

**The irony: the simpler simulator produces the more interesting landscape.** Seven
coupled ODEs with one sigmoid cliff generate richer computational structure than a full
rigid-body physics engine with thousands of contact calculations per second. The ODE's
smoothness *enables* structure; PyBullet's contact discontinuities *destroy* it.

---

## Head-to-Head Summary

| Dimension | ER Atlas | Mito Landscape |
|---|---|---|
| Wolfram class | **3** (everywhere fractal) | **2-4 hybrid** (patchwork) |
| Gradient structure | None (hill-climbing fails) | Yes (hill-climbing works) |
| Dead zone | 11.8% (dynamically rich) | 19.8% (patient-stratified) |
| Roughness | Universal (ratio 0.999) | Moderate (CV 0.503) |
| Isotropy | Perfect (ratio 1.01) | Strongly anisotropic (17.4x potency range) |
| Chaos source | Contact dynamics (discontinuous) | Heteroplasmy cliff (one sigmoid) |
| Rucker gnarliness | High entropy, low depth | Lower entropy, higher depth |

The ER landscape is a Jackson Pollock — fractal splatter with self-similar structure at
every scale but no narrative. The mito landscape is a topographic map — smooth valleys,
one sharp cliff, navigable with a compass but with a region near the cliff edge where
the compass spins.

---

## What's Missing (Mito Side)

The 5 discovery tools are written but not yet run. Once executed (~19 min, ~7,800 sims),
they will provide:

- **Sobol indices**: Whether interactions dominate main effects (high ST-S1 = Class 4; low = Class 2)
- **Interaction mapper**: Whether synergy patterns reverse between patient types (reversal = gnarly)
- **Reachable set**: Pareto frontier shape (sharp elbow = phase transition; smooth = well-behaved)
- **Temporal optimizer**: Timing importance score (high = temporal structure matters)
- **Multi-tissue sim**: Whether the cardiac cascade adds a new dimension of gnarliness

---

*Generated 2026-02-15. Data: ER atlas (~15,000 sims across 4 analysis scripts),
JGC landscape (causal_surgery.json 192 sims, perturbation_probing.json 8x24 probes).*
