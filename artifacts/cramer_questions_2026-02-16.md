# Questions from John G. Cramer about the Simulation (2026-02-16)

Questions submitted by John Cramer, author of *How to Live Much Longer: The Mitochondrial DNA Connection* (2025). Answers based on close reading of `simulator.py`, `constants.py`, and the research campaign scripts.

---

## (1) Is human aging being modeled as many individual human bodies with a range of aging parameters, reporting averages, or as the behavior and response of one human body?

**One body at a time.** The simulator models a single "cell" (really a representative tissue compartment) with 7 coupled state variables tracked over 30 years via RK4 integration. Each run uses one patient profile (age, starting heteroplasmy, NAD level, genetic vulnerability, metabolic demand, inflammation level).

However, there is **population-level analysis** layered on top:

- A patient generator (`generate_patients.py`) creates populations of 100 biologically correlated patients (age drives heteroplasmy, NAD, inflammation with realistic correlations) and 82 edge cases
- Each patient is simulated independently and results are aggregated for statistics
- There is a stochastic mode (Euler-Maruyama with additive noise on ROS and damage accumulation) that runs multiple trajectories for confidence intervals -- but these are still the *same* patient with random perturbations, not different bodies

**What it doesn't do:** There is no inter-cellular communication, no organ-level emergent behavior, and no whole-body integration across organs. The `multi_tissue_sim.py` experiment is a step toward this -- it runs brain, muscle, and cardiac tissue simultaneously coupled by a shared NAD+ pool, systemic inflammation, and cardiac blood flow -- but it is still three independent ODE systems with coupling terms, not a true multi-organ model.

---

## (2) Is the "mitochondrial heteroplasmy cliff", which is modeled as the mutation fraction of mtDNA at which cellular failure occurs, different for the different cell types and organs?

**No -- the cliff threshold (70%) is a single global constant.** `HETEROPLASMY_CLIFF = 0.7` in `constants.py` is used by the same sigmoid function (`_cliff_factor` in `simulator.py`) for all tissue types.

What *does* vary by tissue type are three modifiers (defined in `constants.py:TISSUE_PROFILES`):

| Tissue | Metabolic Demand | ROS Sensitivity | Biogenesis Rate |
|--------|-----------------|-----------------|-----------------|
| Default | 1.0 | 1.0 | 1.0 |
| Brain | 2.0 | 1.5x | 0.3x |
| Muscle | 1.5 | 0.8x | 1.5x |
| Cardiac | 1.8 | 1.2x | 0.5x |

So brain tissue *reaches* the cliff faster (higher ROS sensitivity, higher metabolic demand, almost no biogenesis from exercise) even though the cliff threshold itself is the same. This is a known simplification -- in reality, the threshold likely varies by tissue because different tissues have different electron transport chain complex compositions and different capacities for compensatory upregulation.

**This would be a good area to refine** if there is literature on tissue-specific thresholds.

---

## (3) Are the mutation types (point transition, point transversion, variable-length deletion) and their rates and functional time dependences (linear vs. exponential) considered in the simulation?

**No -- the simulation does not distinguish mutation types.** All mtDNA damage is lumped into a single `N_damaged` state variable. There is no distinction between:

- Point transitions
- Point transversions
- Variable-length deletions

The deletion-rate function (`_deletion_rate()` in `simulator.py`) uses the Va23 doubling-time data (11.8 years before age 65, 3.06 years after) and is coupled to ATP level and mitophagy efficiency (per the C10 correction). But this represents the *aggregate* rate of all damage types, not individual mutation classes.

The `DAMAGED_REPLICATION_ADVANTAGE = 1.05` constant specifically references deletion mutations (>3kbp deletions replicate "at least 21% faster" per Va23/Appendix 2 pp.154-155), but it is applied uniformly to all `N_damaged` copies regardless of actual damage type. The code uses a conservative 5% advantage vs. the book's >=21%.

**What this misses:** If point mutations and deletions have different functional consequences (e.g., point mutations may impair but not destroy ETC function, while large deletions completely eliminate it), lumping them together means the cliff behavior may be less sharp than modeled. A multi-class damage model would need separate state variables for each mutation type with distinct rates and functional impacts.

---

## (4) Cramer's hypothesis is that accumulating mtDNA failure produces a body-wide energy shortage that is the root cause of human aging. To what extent is this hypothesis verified by the simulation?

**The simulation *embodies* the hypothesis rather than independently verifying it.** The core thesis -- that accumulating mtDNA damage causes an energy crisis -- is built directly into the ODE equations:

- The cliff factor (`_cliff_factor`) creates the nonlinear ATP collapse at 70% heteroplasmy
- ATP production depends on `cliff * NAD * (1 - senescence)`
- Low ATP feeds back into halted replication (C1 fix), increased apoptosis, and accelerated senescence
- The ROS-damage vicious cycle is quadratic in heteroplasmy

What the simulation *does* show is that **the hypothesis is internally consistent and self-reinforcing:**

- Natural aging (no intervention) of a 70-year-old drives heteroplasmy from 30% toward the cliff over 30 years
- Past the cliff, bistability locks in collapse -- damaged copies' replication advantage prevents recovery (C4 fix)
- Interventions that address the root cause (transplant displacing damaged copies) work far better than symptom management (NAD supplementation)

What it **cannot** do is verify the hypothesis against *alternative* explanations for aging (telomere shortening, epigenetic drift, protein aggregation, etc.) because those mechanisms are not modeled. The simulation shows that *if* the mtDNA hypothesis is correct, the dynamics play out as described in the book -- but it does not rule out competing theories.

The `causal_surgery.py` experiment does provide one useful finding: there is a "point of no return" -- if intervention starts too late (past the cliff), most treatments fail. This is a testable prediction (see question 5).

---

## (5) Are there predictions made by the simulation that can be tested to support its validity?

Yes, the simulation makes several predictions that are in principle testable:

### 1. Point of no return

`causal_surgery.py` finds that intervention switching from no-treatment to full cocktail becomes ineffective past a certain age/heteroplasmy combination. **Prediction:** Clinical trials starting treatment in elderly patients past the cliff will show much weaker responses than trials in younger patients with the same protocol.

### 2. NAD supplementation alone is insufficient at low doses

The CD38 degradation model (C7 correction, Ch. VI.A.3 p.73) predicts that low-dose NMN/NR without CD38 suppression (apigenin) will show minimal benefit. The simulation shows >2x nonlinear dose-response due to CD38 gating. **Prediction:** Dose-escalation trials of NMN +/- apigenin will show a threshold effect where low doses are near-futile but high doses (with CD38 suppression) are significantly effective.

### 3. Transplant outperforms all pharmaceutical interventions

The model predicts mitochondrial transplant is the only intervention that can actually reverse accumulated damage (reduce heteroplasmy), while drugs can only slow accumulation. **Prediction:** Transplant-based therapies will show heteroplasmy *reversal*, not just stabilization, in treated tissues.

### 4. Tissue-specific vulnerability ordering

Brain fails first (high demand, high ROS sensitivity, low biogenesis), then cardiac, then muscle. The `multi_tissue_sim.py` experiment shows a "cardiac cascade" where heart failure reduces blood flow, accelerating brain and muscle decline. **Prediction:** In longitudinal aging studies, mitochondrial dysfunction markers should appear earliest in neural tissue, then cardiac, then skeletal muscle.

### 5. Intervention synergy patterns

`interaction_mapper.py` maps synergy and antagonism between intervention pairs. **Predictions:**

- Rapamycin + exercise is synergistic (both boost mitophagy through different pathways)
- Yamanaka + any high-energy intervention is antagonistic (Yamanaka drains the ATP that other interventions need)
- Transplant + rapamycin is the most synergistic pair (transplant adds healthy copies, rapamycin clears damaged ones)

### 6. The deletion rate transition is health-dependent, not purely age-dependent

The C10 correction predicts that the Va23 age-65 transition in deletion doubling time should correlate with cellular health markers (ATP, mitophagy activity), not just chronological age. **Prediction:** Healthy 70-year-olds should show "young" doubling times; unhealthy 55-year-olds should show "old" rates. The transition age should be earlier in metabolically stressed tissues.

### Important caveat

These are predictions of the *model*, which has many free parameters (coupling strengths, rates) that were calibrated rather than measured. The model's predictive power depends on how well those parameters reflect actual biology. The constants that are grounded in empirical data (doubling times from Va23, CD38 degradation, transplant biology) generate the most trustworthy predictions; the coupling strengths (ROS-damage feedback, cliff steepness) are simulation parameters whose values would benefit from experimental calibration.
