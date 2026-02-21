# JGC Project State: 2026-02-15

## Summary

The project is **architecturally complete but empirically thin**. The simulation engine
is solid, the analysis tools are comprehensive, but most of them haven't been exercised.
It's a well-equipped lab with almost no experimental data.

---

## What's Built

30 Python files, ~12,870 lines. The architecture is complete:

- **Core engine**: ODE simulator with 7 state variables, 4th-order RK4, tissue types,
  stochastic mode, phased intervention schedules. Corrected through a falsifier audit
  (C1-C4, M1) and two rounds of Cramer email review (C7-C9).
- **Analysis stack**: 4-pillar analytics, 85 pytest tests, pydantic validation, grid
  snapping, LLM parsing with flattening detection.
- **Research campaign**: 5 tiers of scripts mirroring the ER project's atlas
  infrastructure — from pure simulation through LLM-seeded experiments to discovery tools.
- **Documentation**: Every biological constant cited to Cramer, forthcoming 2026 chapter/page. Every
  coefficient in every tool justified with biological rationale.

## What's Actually Been Run

Only **3 artifact files** exist:

| Artifact | Sims | Status |
|---|---|---|
| `causal_surgery.json` | 192 | Complete |
| `perturbation_probing.json` | ~200 | Complete |
| `pest_control_report.json` | — | Complete |

Out of ~15 scripts that produce artifacts, 3 have been executed.

## What Hasn't Run

**Could run right now (no Ollama, ~27 min total, ~17,000 sims):**

| Script | Sims | Time | What it reveals |
|---|---|---|---|
| `dark_matter.py` | ~700 | ~2 min | Futile intervention taxonomy |
| `protocol_interpolation.py` | ~1325 | ~3 min | Super-protocols at midpoints |
| `cliff_mapping.py` | — | ~2 min | Cliff threshold characterization |
| `sobol_sensitivity.py` | ~6656 | ~3 min | Parameter interactions (ST-S1) |
| `interaction_mapper.py` | ~2160 | ~3 min | Synergy/antagonism pairs |
| `reachable_set.py` | ~2400 | ~5 min | Pareto frontiers per patient |
| `competing_evaluators.py` | ~1000 | ~2 min | Transaction protocol rarity |
| `temporal_optimizer.py` | ~3000 | ~7 min | Phased vs constant dosing |
| `multi_tissue_sim.py` | ~30 | ~2 min | Cardiac cascade, tissue coupling |

**Requires Ollama (hours):**
- `tiqm_experiment.py` — core TIQM pipeline, never run
- `oeis_seed_experiment.py` — ~2 hrs
- `character_seed_experiment.py` — ~5 hrs
- `fisher_metric.py` — ~30-60 min
- `clinical_consensus.py` — ~15-20 min
- `posiwid_audit.py` — ~15-20 min

**Requires prior data:**
- `archetype_matchmaker.py` — needs character experiment
- `categorical_structure.py` — needs seed data
- `llm_seeded_evolution.py` — benefits from seed data

## Key Findings So Far

From the ~400 sims that have run:

1. **Transplant dominates** — sensitivity 0.76, confirming Cramer's recommendation.
   The C8 correction (doubling rate, adding competitive displacement) made transplant
   the clear primary intervention.

2. **Exercise and senolytics are nearly inert** — sensitivity < 0.1. Either these
   mechanisms aren't bottleneck-breaking in the current ODE, or the model is missing
   biogenesis feedback and SASP coupling dynamics.

3. **Age has zero perturbation sensitivity** — suspicious. AGE_TRANSITION at 65 should
   create a sharp change but perturbation probes don't span the boundary. The moderate
   patient (age 60) never crosses 65 during a 30-year sim; the near-cliff patient (75)
   is already past it.

4. **Robustness paradox** — successful protocols are 7.4x more robust to perturbation
   than failing protocols. The landscape around good protocols is a broad mesa; around
   failing protocols, a knife-edge.

5. **No point of no return within the window** — for near-cliff patients, the bistable
   attractor captures the trajectory before any tested switch time. By het=0.60 at age
   75, you're already committed unless you hit the system with maximum transplant.

6. **Moderate roughness** — CV(ATP) = 0.503. Navigable landscape with gradient
   structure. Hill-climbing works. This is fundamentally different from the ER project's
   proven-fractal Class 3 chaos (roughness ratio 0.999, sign flip rate 0.58-0.72).

## The Asymmetry with the ER Project

The ER project has a rich empirical atlas: ~15,000 simulations, quantitative proof of
universal fractality, dark matter taxonomy with 5 behavioral categories, and the central
result that champion transects are exactly as rough as random directions. The landscape
is *proven* fractal — an actual scientific finding.

The JGC project's characterization is tentative. The ~27 minutes of Tier 1 + Tier 5
tools would produce ~17,000 sims and transform observations into a proper atlas.

## The Key Advantage Not Yet Exploited

The mito landscape is **navigable** — gradients exist, hill-climbing works, the
landscape has structure. The ER project proved its landscape is a fractal hash where
the only strategy is random global sampling. The JGC project hasn't yet demonstrated
what *can* be done with a smooth landscape.

The discovery tools (D1-D5) were designed for this: Pareto frontiers, synergy maps,
temporal optimization, multi-tissue coupling. None have produced data yet.

## Next Steps

1. **Immediate**: Build a hill climber in the ER project, then port it here. The mito
   landscape's gradient structure means hill-climbing should dramatically outperform
   random search — in direct contrast to the ER project where it provably cannot work.
   This is the cleanest cross-project comparison: same algorithm, fundamentally
   different landscapes, opposite outcomes.

2. **~27 minutes of compute**: Run all Tier 1 + Tier 5 tools to populate the atlas.

3. **LLM experiments**: Run the TIQM pipeline, character seeds, and Zimmerman
   experiments once Ollama time is available.

---

*Generated 2026-02-15.*
