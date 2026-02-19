# Design Document: Time-Varying Parameter Resolver

**Date:** 2026-02-19
**Status:** Design only — not yet implemented
**Author:** Claude Code session (handoff integration planning)

## Problem

The handoff expansion adds ~38 new patient/intervention parameters (genetics, grief, lifestyle, supplements, neuroplasticity, etc.) that need to flow into the frozen Cramer core ODE without modifying its 8-state equations or book-referenced constants. Many of these new parameters are **time-varying**: grief intensity declines over years, supplement adherence may change, cognitive engagement ramps up, alcohol intake may be phased out. The core needs to see smoothly evolving effective 12D inputs, not just static values.

## Existing Time-Varying Mechanisms

The codebase already has **three** independent time-varying mechanisms. The design must unify with them, not create a fourth.

### 1. InterventionSchedule (simulator.py, lines 112–211)

**Piecewise-constant** intervention switching. `InterventionSchedule.at(t)` returns a full 6-param dict for whichever phase is active. Resolved per-step via `_resolve_intervention()` inside the main integration loop (line 1280). Supports `phased_schedule()` and `pulsed_schedule()` constructors.

**Limitation:** Only handles the 6 intervention params. Patient params are static for the entire simulation. Step-function interpolation (no blending between phases).

### 2. Disturbance.modify_params() (disturbances.py, lines 105–120)

**Per-timestep parameter overlay.** Each active `Disturbance` receives copies of the current intervention and patient dicts and returns modified copies. Effects stack across multiple simultaneous disturbances. Applied inside `simulate_with_disturbances()`'s custom RK4 loop (lines 562–574).

**Key pattern:** Patient params become time-varying via `dict(patient)` copy + modification, without permanently mutating the originals. The modifications are computed from the disturbance's internal state (magnitude, time window), not from the ODE state vector.

**Limitation:** Only available in `simulate_with_disturbances()`, not in the main `simulate()`. Requires the full custom integration loop (cannot use the standard `simulate()` path).

### 3. GriefDisturbance (grief_bridge.py, lines 151–204)

**Pre-computed trajectory interpolation.** Runs the external grief simulator at construction time, stores 5 time-series curves (inflammation, cortisol, SNS, sleep, CVD), then interpolates at each mito timestep via `np.interp()`. Maps interpolated grief signals to mito patient params inside `modify_params()`.

**Key pattern:** An external ODE (grief) is solved independently, its trajectory frozen, then injected into the mito ODE as a time-varying parameter modulation. The grief ODE doesn't see mito outputs — it's one-directional.

## Design

### Core Concept: ParameterResolver

A `ParameterResolver` sits between the expanded 50D input space and the frozen 12D core. It is called at each timestep with the current simulation time `t` and (optionally) the current core state vector, and returns effective 6-param intervention + 6-param patient dicts.

```python
class ParameterResolver:
    """Resolves expanded parameters to effective 12D core inputs at each timestep."""

    def __init__(
        self,
        patient_expanded: dict,      # ~20 patient params (genetics, sex, grief, intellectual, ...)
        intervention_expanded: dict,  # ~30 intervention params (supplements, lifestyle, therapy, ...)
        schedules: dict[str, Any] | None = None,  # optional time-varying overrides
    ) -> None: ...

    def resolve(
        self,
        t: float,
        core_state: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return (effective_intervention, effective_patient) for the core ODE at time t."""
        ...
```

### How It Computes Effective Parameters

The resolver applies a **chain of modifiers** in a fixed order. Each modifier reads from the expanded params and/or time, and adjusts the effective 12D values.

```
Step 1: BASELINE
    Start with the 6 core patient defaults derived from expanded params
    (age, het, NAD, vulnerability, demand, inflammation)
    Start with the 6 core intervention defaults (all 0.0)

Step 2: GENETICS (static)
    apoe_genotype  → genetic_vulnerability multiplier (1.0 / 1.3 / 1.6)
    foxo3          → genetic_vulnerability multiplier (* 0.9 if protective)
    cd38_risk      → stored for NAD gating (affects supplement effectiveness)

Step 3: SEX (static, but menopause_status could be time-dependent)
    sex + menopause_status → inflammation_level modifier, het modifier
    estrogen_therapy → partial reversal of menopause effects

Step 4: GRIEF (time-varying)
    If grief module active:
        grief_intensity(t) → inflammation_level additive, metabolic_demand additive
        Uses pre-computed grief trajectory (same pattern as GriefDisturbance)
        OR uses a simple exponential decay: G(t) = G0 * exp(-decay_rate * t)
            where decay_rate depends on therapy_intensity + support_group + coping

Step 5: SLEEP (time-varying if Oura-driven, otherwise static)
    sleep_quality(t) → inflammation_level modifier
    Poor sleep amplifies ROS via inflammation pathway

Step 6: LIFESTYLE (can be phased via schedules)
    alcohol_intake(t) → inflammation_level additive, NAD modifier
        × apoe_sensitivity from Step 2
    diet_type → metabolic_demand modifier (keto reduces effective demand)
    fasting_regimen → effective mitophagy boost (maps to rapamycin-like effect)
    coffee_intake → NAD boost (trigonelline), inflammation reduction (chlorogenic acid)
        × preparation method × timing × genotype multipliers

Step 7: SUPPLEMENTS (can be phased via schedules)
    For each of 11 supplements:
        Hill-function dose-response: effect = max_effect * dose / (dose + half_max)
    Aggregate effects:
        NAD-boosting supplements (NR, B complex) → effective nad_supplement
        Anti-inflammatory (DHA, ALA, vitamin D, zinc, selenium) → inflammation_level reduction
        Mitophagy-boosting (resveratrol, PQQ) → effective rapamycin_dose additive
        ETC support (CoQ10) → metabolic_demand reduction (better efficiency)
        Sleep/cognitive (magnesium) → sleep_quality improvement → inflammation
    Gut health modulates NAD supplement effectiveness:
        effective_nad = raw_nad * (0.7 + 0.3 * gut_health(t))

Step 8: PROBIOTICS (time-varying — gut_health evolves)
    gut_health(t) computed via simple ODE or pre-computed trajectory
    Affects NAD conversion efficiency (Step 7 feedback)

Step 9: CORE INTERVENTION SCHEDULE
    Apply any InterventionSchedule for the 6 core params
    (rapamycin, NAD, senolytics, Yamanaka, transplant, exercise)
    These override/add to the values computed above

Step 10: CLAMP
    Ensure all 12 effective params are within their valid ranges
```

### Time-Varying Strategies

Different parameters vary on different timescales. The resolver supports three strategies, matching existing patterns:

#### Strategy A: Pre-computed trajectory (like GriefDisturbance)

For parameters whose time evolution is known at simulation start and doesn't depend on the core ODE state:

```python
# At construction time:
grief_curve = precompute_grief_decay(G0=0.9, therapy=0.5, support=0.3, years=30)
alcohol_curve = precompute_alcohol_taper(start=0.8, end=0.0, taper_years=2)

# At resolve time:
grief_intensity = np.interp(t, grief_curve.times, grief_curve.values)
alcohol_intake = np.interp(t, alcohol_curve.times, alcohol_curve.values)
```

**Used for:** grief intensity, alcohol tapering, supplement adherence curves, intellectual engagement ramp-up, sleep improvement trajectory.

**Precedent:** `GriefDisturbance._interp()` already does exactly this.

#### Strategy B: Phased schedule (like InterventionSchedule)

For parameters that change at discrete timepoints (start supplements at year 2, add rapamycin at year 5):

```python
supplement_schedule = {
    0: {'nr_dose': 0.0, 'dha_dose': 0.0},
    2: {'nr_dose': 0.8, 'dha_dose': 1.0},  # start OTC
    5: {'nr_dose': 1.0, 'dha_dose': 1.0},  # increase NR
}
```

**Used for:** intervention escalation, diet changes, adding prescriptions.

**Precedent:** `InterventionSchedule.at()` and `phased_schedule()`.

#### Strategy C: Simple internal ODE (like gut_health)

For parameters that evolve continuously and are driven by other expanded params but NOT by the core state:

```python
# Gut health: driven by probiotics, diet, alcohol — not by core het/ATP/ROS
dgut/dt = (probiotic * 0.1 + diet_effect) * (1 - gut) - gut * 0.02 - alcohol * 0.125

# Pre-integrate at construction time (these don't depend on core state):
gut_trajectory = integrate_gut_ode(probiotic_schedule, diet_schedule, alcohol_schedule, years=30)
```

Since gut health, grief decay, and any other "auxiliary ODEs" don't read the core state vector, they can be **pre-integrated at construction time** and then treated as Strategy A (pre-computed trajectory). This avoids coupling auxiliary ODEs into the core integration loop.

**Key insight:** If an auxiliary variable doesn't depend on core outputs (het, ATP, ROS, NAD, etc.), it can always be pre-computed. The one-directional coupling guarantee (auxiliary → core, never core → auxiliary) makes this possible.

### What About State-Dependent Resolution?

The `resolve()` method accepts `core_state` as an optional argument. This is for future use if any expanded parameter genuinely needs to see the core state (e.g., "if ATP drops below 0.5, automatically increase exercise"). However:

- **The initial implementation should NOT use this.** All proposed handoff parameters can be pre-computed or scheduled.
- If state-dependent resolution is ever needed, it would require the resolver to be called inside the integration loop (like `_resolve_intervention` already is), which is straightforward since the architecture supports it.
- State-dependent resolution breaks the pre-computation guarantee and would make the resolver non-cacheable.

### Integration with Existing Code Paths

#### Path 1: simulate() (main simulator)

Currently resolves interventions via `_resolve_intervention(intervention, t)` at line 1280. Patient params are static.

**Change:** Add an optional `resolver: ParameterResolver` argument to `simulate()`. If provided, `_resolve_intervention` is replaced by `resolver.resolve(t)` which returns both intervention and patient dicts. If not provided, behavior is unchanged (backwards compatible).

```python
def simulate(
    intervention=None,
    patient=None,
    resolver=None,        # NEW: ParameterResolver instance
    sim_years=None,
    dt=None,
    tissue_type=None,
    stochastic=False,
    ...
) -> dict:
    ...
    for i in range(n_steps):
        t = i * dt
        if resolver is not None:
            current_intervention, current_patient = resolver.resolve(t, state)
        else:
            current_intervention = _resolve_intervention(intervention, t)
            current_patient = patient
        ...
```

This is a **minimal change** to the core integration loop — one `if/else` added, no ODE modifications.

#### Path 2: simulate_with_disturbances() (disturbances.py)

Currently applies `Disturbance.modify_params()` on top of resolved intervention + patient. The resolver would slot in **before** disturbance modification:

```python
for i in range(n_steps):
    t = i * dt
    # ... impulse injection ...

    # Resolve expanded params to effective 12D
    if resolver is not None:
        current_intervention, current_patient = resolver.resolve(t, state)
    else:
        current_intervention = _resolve_intervention(intervention, t)
        current_patient = dict(patient)

    # Disturbances overlay on top of resolved params (unchanged)
    for d in disturbances:
        current_intervention, current_patient = d.modify_params(
            current_intervention, current_patient, t)

    # RK4 step with effective params (unchanged)
    ...
```

Disturbances and the resolver compose cleanly because disturbances operate on the **effective** 12D, not the expanded 50D. A `GriefDisturbance` could coexist with a resolver that also has grief modeling — the disturbance would overlay additional acute stress on top of the resolver's chronic grief trajectory.

#### Path 3: MitoSimulator (zimmerman_bridge.py)

The Zimmerman bridge's `run()` method currently splits a flat 12D dict into intervention + patient and calls `simulate()`. With the resolver:

```python
def run(self, params: dict) -> dict:
    # If params include expanded keys (apoe_genotype, nr_dose, etc.),
    # construct a ParameterResolver
    if self._has_expanded_params(params):
        resolver = ParameterResolver(
            patient_expanded=self._extract_patient(params),
            intervention_expanded=self._extract_intervention(params),
        )
        result = simulate(resolver=resolver)
    else:
        # Legacy 12D path (unchanged)
        intervention, patient = self._split(params)
        result = simulate(intervention=intervention, patient=patient)
```

The `param_spec()` method would need a mode flag to return either 12D or ~50D bounds. Categorical params (diet_type, sex, etc.) would be encoded as integer ordinals for Zimmerman compatibility.

### Downstream Chain (Neuroplasticity, Alzheimer's)

The neuroplasticity chain (MEF2 → HA → synaptic_strength → memory_index) and Alzheimer's pathology (amyloid, tau) are **downstream** — they read core outputs but don't feed back. They are NOT part of the parameter resolver.

Instead, they run as a **post-processing pass** on the core trajectory:

```python
def compute_downstream(
    core_result: dict,           # from simulate()
    patient_expanded: dict,      # for genotype multipliers, engagement level, etc.
    resolver: ParameterResolver, # for time-varying engagement, therapy, etc.
) -> dict:
    """Integrate downstream state variables driven by core trajectory."""
    states = core_result['states']  # (3001, 8)
    times = core_result['time']     # (3001,)
    dt = times[1] - times[0]

    # Initialize downstream state
    mef2 = 0.2
    ha = 0.2
    ss = 1.0
    cr = 0.5
    amyloid = 0.2  # age-dependent
    tau = 0.1

    downstream_history = []

    for i, t in enumerate(times):
        # Read core outputs at this timestep
        atp = states[i, 2]
        ros = states[i, 3]
        nad = states[i, 4]
        sen = states[i, 5]

        # Get time-varying expanded params
        _, patient_t = resolver.resolve(t)
        engagement = patient_t.get('intellectual_engagement', 0)
        # ... other expanded params ...

        # Integrate downstream ODEs (Euler is fine — these are slow dynamics)
        dmef2 = mef2_ode(mef2, engagement, apoe_mult)
        dha = epigenetic_ode(ha, mef2)
        dss = synaptic_ode(ss, ha, engagement)
        dcr = cognitive_reserve_ode(cr, engagement, education, ...)
        damyloid = amyloid_ode(amyloid, inflammation_from_core, age, apoe_clearance)
        dtau = tau_ode(tau, amyloid, inflammation_from_core, mitophagy_from_core)

        mef2 += dmef2 * dt
        ha += dha * dt
        # ... etc ...

        # Derived variables
        mi = memory_index(ss, mef2, cr, amyloid, tau)

        downstream_history.append({
            'MEF2_activity': mef2,
            'histone_acetylation': ha,
            'synaptic_strength': ss,
            'cognitive_reserve': cr,
            'amyloid_burden': amyloid,
            'tau_pathology': tau,
            'memory_index': mi,
        })

    return downstream_history
```

This keeps the core integration loop untouched. The downstream chain reads `core_result['states']` after the fact. Euler integration is acceptable because these variables have slow dynamics (timescales of years, not days).

### Pre-Computation at Construction Time

Since all proposed auxiliary dynamics (grief decay, gut health, alcohol taper) are independent of the core state, the resolver pre-computes all trajectories at `__init__` time:

```python
class ParameterResolver:
    def __init__(self, patient_expanded, intervention_expanded, schedules=None):
        self.patient = patient_expanded
        self.intervention = intervention_expanded

        # Pre-compute time-varying trajectories
        self._grief_curve = self._precompute_grief()
        self._gut_curve = self._precompute_gut_health()
        self._alcohol_curve = self._precompute_alcohol(schedules)
        self._supplement_schedule = self._build_supplement_schedule(schedules)
        # ... etc ...

        # Pre-compute static modifiers (genetics, sex)
        self._genetic_mods = self._compute_genetic_modifiers()
        self._sex_mods = self._compute_sex_modifiers()

    def resolve(self, t, core_state=None):
        # Start from baseline
        intervention = dict(DEFAULT_INTERVENTION)
        patient = self._base_patient_12d.copy()

        # Apply static modifiers
        patient['genetic_vulnerability'] *= self._genetic_mods['vulnerability']
        patient['inflammation_level'] += self._sex_mods['inflammation_delta']

        # Apply time-varying modifiers (interpolate pre-computed curves)
        grief_t = np.interp(t, self._grief_curve.t, self._grief_curve.v)
        patient['inflammation_level'] += grief_t * GRIEF_INFLAMMATION_COEFF

        gut_t = np.interp(t, self._gut_curve.t, self._gut_curve.v)
        alcohol_t = np.interp(t, self._alcohol_curve.t, self._alcohol_curve.v)
        patient['inflammation_level'] += alcohol_t * ALCOHOL_INFLAMMATION * self._genetic_mods['alcohol_sensitivity']

        # Supplements → effective intervention params
        supplement_effects = self._compute_supplements(t, gut_t)
        intervention['nad_supplement'] += supplement_effects['nad_boost']
        intervention['rapamycin_dose'] += supplement_effects['mitophagy_boost']

        # Apply core intervention schedule
        core_sched = self._resolve_core_schedule(t)
        for k, v in core_sched.items():
            intervention[k] = max(intervention[k], v)  # take the higher of supplement-derived and explicit

        # Clamp
        for k in intervention:
            intervention[k] = np.clip(intervention[k], 0.0, 1.0)
        patient['inflammation_level'] = np.clip(patient['inflammation_level'], 0.0, 1.0)
        # ... etc ...

        return intervention, patient
```

### Interaction with the Scenario Framework (Batch 4)

The `Scenario` dataclass from batch 4 maps directly to a `ParameterResolver`:

```python
def scenario_to_resolver(scenario: Scenario) -> ParameterResolver:
    return ParameterResolver(
        patient_expanded=scenario.patient_params,
        intervention_expanded=vars(scenario.interventions),
        schedules=scenario.schedules,  # optional time-varying overrides
    )
```

The scenario runner creates a resolver per scenario, passes it to `simulate()`, then runs the downstream chain on the result.

## File Changes Summary

| File | Change | Scope |
|------|--------|-------|
| `parameter_resolver.py` | **NEW** — ParameterResolver class, modifier chain, pre-computation | ~400-500 lines |
| `downstream_chain.py` | **NEW** — MEF2/HA/SS/CR/amyloid/tau integration + memory_index | ~300-400 lines |
| `simulator.py` | Add optional `resolver` arg to `simulate()`, one `if/else` in loop | ~10 lines changed |
| `disturbances.py` | Add optional `resolver` arg to `simulate_with_disturbances()`, same pattern | ~10 lines changed |
| `zimmerman_bridge.py` | Detect expanded params, construct resolver | ~30 lines added |
| `constants.py` | Add all new constants (grief, genetics, supplements, etc.) | ~200 lines added |
| `scenario_definitions.py` | **NEW** — Scenario/InterventionProfile dataclasses + `scenario_to_resolver()` | ~200 lines |

## What This Design Preserves

1. **Cramer core is untouched.** The 8-state ODE, all book constants, the cliff dynamics, the RK4 integrator, and `derivatives()` are not modified. The resolver only changes what `intervention` and `patient` dicts are passed in — the same dicts the core has always consumed.

2. **All 262 existing tests pass unchanged.** The resolver is opt-in. Without it, `simulate()` and `simulate_with_disturbances()` behave exactly as before.

3. **Disturbances compose with the resolver.** Acute shocks overlay on top of the resolver's chronic modulations, just as they overlay on top of static params today.

4. **GriefDisturbance still works.** The existing grief bridge (which uses the external grief-simulator) can coexist with the resolver's internal grief decay model. They serve different purposes: the bridge uses the full 11D grief ODE for detailed psychobiological modeling; the resolver's grief model is a simplified decay for the expanded precision-medicine scenarios.

5. **Zimmerman protocol compliance.** The bridge can operate in legacy 12D mode or expanded ~50D mode, with the resolver handling the translation.

## Risks and Open Questions

1. **Supplement stacking:** If NR + resveratrol + fasting all boost effective `rapamycin_dose`, the combined value could exceed 1.0. Clamping at 1.0 loses information about the relative contributions. Alternative: use unclamped effective values and let the ODE's internal saturation handle it (e.g., mitophagy rate has diminishing returns).

2. **Gut health feedback loop:** Gut health affects NAD supplement effectiveness, which affects core NAD, which affects... nothing in the auxiliary ODEs (gut health doesn't depend on core NAD). So the pre-computation is valid. But if we later want gut health to depend on core inflammation (plausible biology), pre-computation breaks and gut health must move into the integration loop or the downstream chain.

3. **Grief → engagement → MEF2 → grief reduction feedback.** The handoff specifies that MEF2 activity reduces grief (GRIEF_REDUCTION_FROM_MEF2 = 0.1). But MEF2 is in the downstream chain, and grief is in the resolver. This creates a circular dependency: grief(t) → resolver → core → downstream MEF2(t) → grief(t+1). Options:
   - **Ignore for v1:** Use the simplified grief decay (no MEF2 feedback). The pre-computed trajectory is a reasonable approximation.
   - **Iterate:** Run the system twice — first pass with no MEF2 feedback to get approximate MEF2 trajectory, second pass with MEF2-informed grief curve. Converges in 2-3 iterations.
   - **Move grief into the downstream chain:** Integrate grief alongside MEF2 in the post-processing pass, reading core inflammation as input. This is the cleanest long-term solution but means grief no longer modulates core params (it would only affect the downstream variables).

4. **Categorical params in Zimmerman protocol.** `param_spec()` returns `(low, high)` float tuples. Categoricals (diet_type, sex, coffee_type) need ordinal encoding (0, 1, 2) and the resolver maps back. Sobol sampling will treat them as continuous, which is imperfect but workable for sensitivity analysis.

5. **Performance.** Pre-computing all trajectories at construction time adds ~10ms per resolver creation (negligible). The `resolve()` call adds ~6 `np.interp()` lookups per timestep × 3000 steps = 18,000 interp calls. At ~1μs each, that's ~18ms total — negligible compared to the 3000 RK4 steps (~100ms).
