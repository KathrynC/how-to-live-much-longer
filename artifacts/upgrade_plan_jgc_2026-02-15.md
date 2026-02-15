# JGC Upgrade Plan: how-to-live-much-longer

**Date:** 2026-02-15
**Informed by:** Zimmerman (2025) dissertation on LLM meaning construction, pest control audit, slowing/reversing aging analysis

## Context

The Zimmerman dissertation revealed systematic LLM blind spots (flattening, frequency-truth conflation, supradiegetic blindness) and tools (PDS dimensions, extispicy, contrastive prompting) that can directly improve how this project uses LLMs. The pest control agent found 55 unused imports and 1 off-grid default. This plan upgrades the project in 6 phases.

---

## Phase A: Code Health

### A1. Fix pest control findings
- Remove 55 unused imports across 17 files (all severity=info, safe to remove)
- Fix `DEFAULT_PATIENT['inflammation_level'] = 0.3` — snap to 0.25 (nearest grid point) or add 0.3 to grid
- Verification: re-run `python pest_control.py` — expect 0 issues

### A2. Consolidate duplicate LLM utilities
- `tiqm_experiment.py` has its own `query_ollama()` and `parse_response()` that duplicate `llm_common.py`
- Refactor to import from `llm_common.py` instead
- Files: `tiqm_experiment.py`, `llm_common.py`

---

## Phase B: Zimmerman-Informed Prompt Engineering

### B1. Diegetic prompt redesign (Zimmerman Ch. 2-3: supradiegetic blindness)
- Current prompts list parameters as bare numbers with range annotations
- LLMs handle diegetic (semantic/narrative) content better than supradiegetic (structural/numerical)
- Redesign: embed parameter choices in clinical narrative context, not as a numeric form to fill
- Example: Instead of `rapamycin_dose: 0-1`, use "How aggressively should we pursue mitophagy enhancement? (conservative/moderate/aggressive/maximum)"
- Create `prompt_templates.py` with both old (numeric) and new (diegetic) prompt styles for A/B comparison
- Files: new `prompt_templates.py`, modify `tiqm_experiment.py`, `character_seed_experiment.py`, `oeis_seed_experiment.py`

### B2. Contrastive prompting (Zimmerman Ch. 5: TALOT/OTTITT)
- Generate TWO protocols per scenario — one aggressive, one conservative — and ask the LLM to explain tradeoffs
- Exploits "meaning arises from contrast" principle
- Add `--contrastive` flag to `tiqm_experiment.py`

### B3. Flattening audit (Zimmerman Ch. 3: qualitative collapse)
- LLMs treat `rapamycin_dose: 0.5` and `baseline_age: 0.5` identically
- Audit all prompts for flattening risks
- Fix: use natural-language ranges, add domain-specific validation to `parse_intervention_vector()`

---

## Phase C: Semantic Bridge Experiments

### C1. PDS-to-patient mapping (Zimmerman Ch. 4: Power/Danger/Structure)
- Map PDS dimensions to patient parameters: Power -> metabolic_demand, Danger -> inflammation + vulnerability, Structure -> NAD level
- New `pds_mapping.py` that loads character archetypes and computes expected patient vectors
- Compare PDS-predicted vs LLM-generated patient parameters

### C2. POSIWID alignment audit (Zimmerman Ch. 5: purpose = what it does)
- Ask LLM "what outcome do you intend?" BEFORE generating parameters, then compare intention vs simulation
- New `posiwid_audit.py` (Tier 2 experiment, requires Ollama)

### C3. Frequency-truth calibration (Zimmerman Ch. 3)
- Check if LLMs over-recommend popular interventions (rapamycin, NMN) vs less-discussed ones (mitlet transplant)
- Extend `fisher_metric.py` output analysis

---

## Phase D: Simulation Enhancements

### D1. Multi-tissue ODE extension
- Add `tissue_type` parameter with 4 profiles: brain, muscle, cardiac, default
- Same ODE, different coupling constants per tissue
- Files: `constants.py`, `simulator.py`

### D2. Sobol sensitivity analysis
- Global sensitivity analysis using Saltelli sampling (numpy-only)
- Identifies parameter interactions, not just individual sensitivities
- New `sobol_sensitivity.py`

### D3. Stochastic ODE mode
- Optional Euler-Maruyama integration for noise in ROS/damage
- N=100 trajectories per scenario for confidence intervals
- Files: `simulator.py` (add `stochastic=True` kwarg)

---

## Phase E: New Research Campaign Scripts

### E1. Archetype-to-protocol matchmaker
- Combine PDS mapping with character experiment data
- Identify which character archetypes produce best protocols for which patient types
- New `archetype_matchmaker.py`

### E2. Protocol evolution with narrative feedback
- Feed LLM the trajectory of previous attempts instead of blind hill-climbing
- "Your last protocol ended at het=0.28. The ATP dipped at year 15. How would you adjust?"
- Modify `llm_seeded_evolution.py` with `--narrative` flag

---

## Phase F: Documentation & Integration

- Update CLAUDE.md with new files and experiments
- Update README.md with new capabilities
- Run pest_control.py — target 0 issues
- Git commit per phase

---

## Implementation Order

```
Phase A (code health)     <- Do first, unblocks everything
Phase B (prompts)         <- Core Zimmerman applications
Phase C (semantic bridge) <- New experiments (depends on B1)
Phase D (simulation)      <- Independent of B/C, can run in parallel
Phase E (campaigns)       <- Requires B+C+D
Phase F (docs)            <- Last
```
