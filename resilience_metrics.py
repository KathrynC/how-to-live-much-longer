"""Resilience metrics for mitochondrial disturbance response.

Computes ecological resilience indicators from shocked vs baseline
simulation trajectories. Four core metrics:

    1. Resistance    — how much the system deviates under shock
    2. Recovery time — how quickly it returns to pre-shock state
    3. Regime retention — whether it returns to the same attractor
    4. Elasticity    — normalized rate of recovery

Agroecology analogy: a healthy ecosystem resists perturbation (low
peak deviation), recovers quickly, and returns to its original regime
rather than flipping to a degraded state. The heteroplasmy cliff
is the regime boundary — crossing it is an irreversible regime shift.

Usage:
    from simulator import simulate
    from disturbances import simulate_with_disturbances, IonizingRadiation
    from resilience_metrics import compute_resilience

    baseline = simulate()
    shocked = simulate_with_disturbances(
        disturbances=[IonizingRadiation(start_year=10.0, magnitude=0.8)]
    )
    metrics = compute_resilience(shocked, baseline)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# HETEROPLASMY_CLIFF (0.70) is the regime boundary — the mitochondrial analog
# of an ecological tipping point (Scheffer 2009).  Below the cliff, cells
# maintain adequate ATP via remaining healthy mtDNA; above it, energy
# production collapses nonlinearly due to a sigmoid threshold in oxidative
# phosphorylation capacity.  DT (0.01 yr ≈ 3.65 days) sets the temporal
# resolution for "sustained recovery" checks (see compute_recovery_time).
from constants import HETEROPLASMY_CLIFF, DT


# ---------------------------------------------------------------------------
# RESISTANCE — "How much does the ecosystem bend before it breaks?"
#
# In Holling's (1973) ecological resilience framework, resistance measures
# the magnitude of state change a system undergoes in response to
# perturbation.  A resistant prairie keeps its grass cover during drought;
# a resistant mitochondrial network keeps ATP production during oxidative
# stress.  We measure resistance as the peak and mean deviation of the
# shocked trajectory from the unperturbed baseline — larger deviation
# means lower resistance.
# ---------------------------------------------------------------------------

def compute_resistance(
    shocked_states: npt.NDArray[np.float64],
    baseline_states: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    shock_start: float,
    shock_end: float,
    state_idx: int = 2,
) -> dict[str, float]:
    """Measure resistance: peak deviation from baseline during/after shock.

    Args:
        shocked_states: States array (n_steps+1, 7) from disturbed sim.
        baseline_states: States array (n_steps+1, 7) from baseline sim.
        time: Time array.
        shock_start: Shock onset time (years).
        shock_end: Shock end time (years).
        state_idx: Which state variable to analyze (default 2 = ATP).

    Returns:
        Dict with resistance metrics.
    """
    # Observation window extends 5 years past shock end.  This captures the
    # *immediate* post-perturbation transient, which in agroecology corresponds
    # to the "lag phase" before recovery mechanisms kick in.  Five years is
    # long enough to see the worst deviation (which often occurs *after* the
    # shock ends, as downstream cascades propagate through the ODE coupling)
    # but short enough to avoid conflating resistance with recovery dynamics.
    window_end = min(shock_end + 5.0, time[-1])
    mask = (time >= shock_start) & (time <= window_end)

    if not np.any(mask):
        return {"peak_deviation": 0.0, "mean_deviation": 0.0,
                "relative_peak_deviation": 0.0}

    shocked_signal = shocked_states[mask, state_idx]
    baseline_signal = baseline_states[mask, state_idx]
    deviation = np.abs(shocked_signal - baseline_signal)

    baseline_mean = np.mean(baseline_signal)
    peak_dev = float(np.max(deviation))
    mean_dev = float(np.mean(deviation))
    # Relative peak deviation normalizes by baseline magnitude so that a
    # 0.1 MU drop in a healthy cell (baseline ATP ~0.85) is distinguished
    # from the same absolute drop in an already-compromised cell (ATP ~0.3).
    # This parallels "percent biomass loss" in grassland resistance studies.
    rel_peak = float(peak_dev / baseline_mean) if baseline_mean > 1e-12 else 0.0

    return {
        "peak_deviation": peak_dev,
        "mean_deviation": mean_dev,
        "relative_peak_deviation": rel_peak,
    }


# ---------------------------------------------------------------------------
# RECOVERY TIME — "How long before the ecosystem returns to its basin?"
#
# In resilience ecology, recovery time (= "engineering resilience" in
# Pimm 1984) is the time a perturbed system needs to re-enter the
# neighbourhood of its pre-disturbance attractor.  Short recovery =
# strong restoring forces (healthy mitophagy, biogenesis upregulation);
# infinite recovery = the system has been pushed into a new basin
# (post-cliff collapse).
#
# The epsilon parameter defines the "neighbourhood" — how close is close
# enough to count as recovered.  The default of 0.05 (5%) is a pragmatic
# choice: it is tight enough to exclude trajectories that merely *approach*
# baseline without truly converging, yet loose enough to tolerate the
# natural oscillations that arise from ODE coupling (ROS-ATP feedback
# produces damped oscillations during recovery).  In ecology, a 5%
# deviation from pre-disturbance biomass is a common "recovered" threshold
# (e.g., Isbell et al. 2015).
# ---------------------------------------------------------------------------

def compute_recovery_time(
    shocked_states: npt.NDArray[np.float64],
    baseline_states: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    shock_end: float,
    state_idx: int = 2,
    epsilon: float = 0.05,
) -> float:
    """Time for shocked trajectory to return within epsilon of baseline.

    Args:
        shocked_states: States array from disturbed sim.
        baseline_states: States array from baseline sim.
        time: Time array.
        shock_end: End of shock window (recovery starts here).
        state_idx: Which state variable (default 2 = ATP).
        epsilon: Fraction of baseline value defining "recovered".

    Returns:
        Recovery time in years, or np.inf if never recovered.
    """
    post_mask = time >= shock_end
    if not np.any(post_mask):
        return np.inf

    post_time = time[post_mask]
    post_shocked = shocked_states[post_mask, state_idx]
    post_baseline = baseline_states[post_mask, state_idx]

    # Recovery threshold: within epsilon * baseline_value.
    # The multiplicative form means the threshold scales with the baseline
    # magnitude, so ATP recovery in a healthy cell (baseline ~0.85) demands
    # convergence to within ~0.04 MU, while a compromised cell (baseline ~0.3)
    # only needs ~0.015 MU.  This is analogous to using percent-of-reference
    # biomass rather than absolute biomass in ecosystem recovery assessment.
    threshold = epsilon * np.abs(post_baseline)
    # Floor prevents division-by-zero pathology when the baseline itself
    # collapses (e.g., near-cliff patients whose untreated trajectory
    # approaches zero ATP — without the floor, *any* nonzero shocked value
    # would look "unrecovered" relative to a vanishing baseline).
    threshold = np.maximum(threshold, 1e-6)

    recovered = np.abs(post_shocked - post_baseline) < threshold

    if not np.any(recovered):
        return np.inf

    # SUSTAINED RECOVERY CHECK: require the trajectory to stay within
    # the epsilon band for at least 10 consecutive timesteps (~36.5 days
    # at DT=0.01 yr) before declaring recovery.  This filters out transient
    # excursions that briefly touch the baseline during damped oscillations
    # but then diverge again.
    #
    # The biological motivation: mitochondrial homeostasis after stress
    # involves coupled feedback (ROS triggers mitophagy, which lowers ROS,
    # which reduces mitophagy demand — a damped oscillator).  A single
    # zero-crossing of the deviation curve does NOT mean the system has
    # settled; it may be mid-oscillation.  Requiring ~5 weeks of sustained
    # convergence ensures the restoring dynamics have truly damped out.
    #
    # Why 10 steps specifically?  At DT=0.01 years, 10 steps ≈ 36.5 days,
    # which is roughly one mitochondrial turnover cycle (half-life of
    # mammalian mitochondria is ~2-4 weeks, Miwa et al. 2022).  If the
    # system stays converged for one full turnover cycle, the new
    # steady-state population has had time to establish itself.
    for i in range(len(recovered)):
        if recovered[i]:
            check_end = min(i + 10, len(recovered))
            if np.all(recovered[i:check_end]):
                return float(post_time[i] - shock_end)

    return np.inf


# ---------------------------------------------------------------------------
# REGIME RETENTION — "Did the lake stay clear, or flip to turbid?"
#
# This is the most critical metric in Holling's *ecological* (as opposed
# to *engineering*) resilience: does the system return to its original
# basin of attraction, or has it been pushed across a tipping point into
# a qualitatively different regime?
#
# The canonical ecological example is shallow-lake eutrophication (Scheffer
# et al. 2001): below a phosphorus threshold the lake is clear (macrophyte-
# dominated); above it, the lake flips to a turbid algae-dominated state
# that is self-reinforcing and extremely hard to reverse.
#
# In our mitochondrial model, the heteroplasmy cliff at 70% damaged mtDNA
# is precisely this kind of regime boundary.  Below the cliff, oxidative
# phosphorylation capacity degrades gradually and homeostatic mechanisms
# (mitophagy, biogenesis) can compensate.  Above the cliff, ATP production
# collapses via a sigmoid nonlinearity, ROS spikes, and the damaged-
# replication advantage (fix C4: 1.05x) creates a self-reinforcing vicious
# cycle — the mitochondrial analog of the turbid-lake attractor.
#
# Crossing the cliff is functionally irreversible without transplant
# intervention (fix C8), just as restoring a eutrophic lake requires
# massive phosphorus removal, not just halting inputs.
# ---------------------------------------------------------------------------

def compute_regime_retention(
    shocked_het: npt.NDArray[np.float64],
    baseline_het: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    shock_start: float,
) -> dict[str, bool | float]:
    """Determine if the system returns to the same regime after shock.

    The heteroplasmy cliff (0.70) is the regime boundary. If the shocked
    trajectory crosses the cliff when the baseline doesn't, or vice versa,
    that's a regime shift.

    Args:
        shocked_het: Heteroplasmy array from disturbed sim.
        baseline_het: Heteroplasmy array from baseline sim.
        time: Time array.
        shock_start: Shock onset time.

    Returns:
        Dict with regime retention indicators.
    """
    # Determine which "basin" the system was in before the shock arrived.
    # We use the mean pre-shock heteroplasmy rather than an instantaneous
    # value to smooth out any transient oscillations from initial conditions.
    pre_mask = time < shock_start
    if not np.any(pre_mask):
        pre_regime_below_cliff = True
    else:
        pre_het_mean = np.mean(shocked_het[pre_mask])
        pre_regime_below_cliff = pre_het_mean < HETEROPLASMY_CLIFF

    # Assess the final regime by averaging the last 10% of the simulation.
    # Using a tail window rather than the very last timestep avoids
    # sensitivity to end-of-simulation transients and gives a robust
    # indicator of which attractor the trajectory has settled into.
    tail_n = max(int(0.1 * len(time)), 10)
    final_shocked_het = np.mean(shocked_het[-tail_n:])
    final_baseline_het = np.mean(baseline_het[-tail_n:])

    # The regime retention test: are the shocked and baseline trajectories
    # on the SAME side of the heteroplasmy cliff at the end?  If the
    # baseline stays below 0.70 but the shock pushed the system above it
    # (or vice versa), that is an irreversible regime shift — the
    # mitochondrial analog of a lake flipping from clear to turbid.
    shocked_below_cliff = final_shocked_het < HETEROPLASMY_CLIFF
    baseline_below_cliff = final_baseline_het < HETEROPLASMY_CLIFF
    regime_retained = shocked_below_cliff == baseline_below_cliff

    # Track whether the shock *ever* pushed het above the cliff, even if
    # the system ultimately recovered.  This captures "near-miss" events
    # where the system transiently entered the collapse basin but was
    # pulled back — important for clinical risk assessment (a patient who
    # briefly crossed the cliff experienced real ATP crisis even if they
    # eventually recovered).
    post_mask = time >= shock_start
    ever_crossed_cliff = bool(
        np.any(shocked_het[post_mask] >= HETEROPLASMY_CLIFF)
        and pre_regime_below_cliff
    )

    return {
        "regime_retained": regime_retained,
        "final_het_shocked": float(final_shocked_het),
        "final_het_baseline": float(final_baseline_het),
        # het_gap > 0 means the shock left permanent residual damage even
        # if the regime was retained — the mitochondrial analog of a
        # grassland that recovers its species composition but at lower
        # total biomass.
        "het_gap": float(final_shocked_het - final_baseline_het),
        "ever_crossed_cliff": ever_crossed_cliff,
    }


# ---------------------------------------------------------------------------
# ELASTICITY — "How fast does the rubber band snap back?"
#
# Elasticity (Westman 1978, adapted from materials science) measures the
# *rate* at which a system returns toward equilibrium after perturbation,
# complementing recovery *time* (which only tells you the endpoint).
# Two systems can recover in the same total time but with very different
# dynamics: one may snap back immediately (high elasticity, fast initial
# recovery), while the other drifts slowly (low elasticity, sluggish
# repair mechanisms).
#
# In mitochondrial biology, high elasticity corresponds to strong
# compensatory responses: rapid mitophagy upregulation, biogenesis
# activation, NAD+ replenishment.  Low or negative elasticity (divergence)
# indicates that the shock has overwhelmed repair capacity — the ROS
# vicious cycle is amplifying damage faster than homeostasis can correct it.
# ---------------------------------------------------------------------------

def compute_elasticity(
    shocked_states: npt.NDArray[np.float64],
    baseline_states: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    shock_end: float,
    state_idx: int = 2,
) -> float:
    """Elasticity: initial rate of recovery after shock ends.

    Measured as the slope of |shocked - baseline| in the first 2 years
    after shock end. Steeper negative slope = faster recovery = higher
    elasticity.

    Args:
        shocked_states: States array from disturbed sim.
        baseline_states: States array from baseline sim.
        time: Time array.
        shock_end: End of shock window.
        state_idx: Which state variable (default 2 = ATP).

    Returns:
        Elasticity value (positive = recovering, negative = diverging).
    """
    # Measure the initial 2 years post-shock.  This window is chosen to
    # capture the *intrinsic* recovery rate — the strength of the restoring
    # force — before secondary adaptations (senescence clearance, long-term
    # NAD+ rebuilding) dominate.  In ecological terms, this is the "fast
    # variable" response (Carpenter & Brock 2006), analogous to measuring
    # grass regrowth rate in the first season after fire rather than the
    # multi-year succession trajectory.
    window_end = min(shock_end + 2.0, time[-1])
    mask = (time >= shock_end) & (time <= window_end)
    # Need at least 3 points for a meaningful linear fit.
    if np.sum(mask) < 3:
        return 0.0

    t_window = time[mask]
    deviation = np.abs(
        shocked_states[mask, state_idx] - baseline_states[mask, state_idx]
    )

    if np.std(t_window) < 1e-12:
        return 0.0

    # Fit a linear trend to the deviation curve.  A negative slope means
    # deviation is shrinking (recovery); a positive slope means it is
    # growing (ongoing divergence / cascading failure).
    # We subtract t_window[0] to improve numerical conditioning of polyfit.
    coeffs = np.polyfit(t_window - t_window[0], deviation, 1)
    slope = float(coeffs[0])

    # Convention: negate the slope so that POSITIVE elasticity = recovery
    # (deviation shrinking) and NEGATIVE = divergence (deviation growing).
    # This follows the ecological convention where higher resilience is
    # reported as a larger positive number.
    return -slope


# ---------------------------------------------------------------------------
# COMPOSITE RESILIENCE — Bringing the four pillars together
#
# The four metrics above map directly to the four facets of ecological
# resilience identified across Holling (1973), Pimm (1984), and
# Walker et al. (2004):
#
#   Resistance       — amplitude of displacement
#   Recovery time    — return time (engineering resilience)
#   Regime retention — basin-of-attraction fidelity (ecological resilience)
#   Elasticity       — speed of the restoring force
#
# The composite score collapses these into a single 0-1 index for
# screening / ranking, while the component scores remain available for
# deeper analysis.
# ---------------------------------------------------------------------------

def compute_resilience(
    shocked_result: dict,
    baseline_result: dict,
    state_idx: int = 2,
    epsilon: float = 0.05,
) -> dict[str, dict | float]:
    """Compute all resilience metrics for a disturbed simulation.

    Args:
        shocked_result: Dict from simulate_with_disturbances().
        baseline_result: Dict from simulate() (no disturbances).
        state_idx: Primary state variable to analyze (default 2 = ATP).
        epsilon: Recovery threshold fraction.

    Returns:
        Dict with "resistance", "recovery", "regime", "elasticity"
        sub-dicts, plus "summary_score" (0-1 composite).
    """
    time = shocked_result["time"]
    shocked_states = shocked_result["states"]
    baseline_states = baseline_result["states"]
    shocked_het = shocked_result["heteroplasmy"]
    baseline_het = baseline_result["heteroplasmy"]
    shock_times = shocked_result.get("shock_times", [])

    # No disturbance applied — the system is trivially maximally resilient.
    if not shock_times:
        return {
            "resistance": {"peak_deviation": 0.0, "mean_deviation": 0.0,
                           "relative_peak_deviation": 0.0},
            "recovery_time_years": 0.0,
            "regime": {"regime_retained": True, "het_gap": 0.0,
                       "final_het_shocked": 0.0, "final_het_baseline": 0.0,
                       "ever_crossed_cliff": False},
            "elasticity": 0.0,
            "summary_score": 1.0,
        }

    # For multi-shock scenarios, bracket the full disturbance window:
    # earliest onset to latest offset.  Resistance is measured across the
    # whole window; recovery begins only after ALL shocks have ended.
    shock_start = min(s[0] for s in shock_times)
    shock_end = max(s[1] for s in shock_times)

    resistance = compute_resistance(
        shocked_states, baseline_states, time,
        shock_start, shock_end, state_idx)

    recovery_time = compute_recovery_time(
        shocked_states, baseline_states, time,
        shock_end, state_idx, epsilon)

    regime = compute_regime_retention(
        shocked_het, baseline_het, time, shock_start)

    elasticity = compute_elasticity(
        shocked_states, baseline_states, time,
        shock_end, state_idx)

    # ── Composite resilience score (0 = fragile, 1 = maximally resilient) ──
    #
    # Each component is mapped to [0, 1] via a transform chosen to match
    # the natural scale of its raw metric:

    # Resistance score: linear in relative peak deviation, clamped to [0, 1].
    # A relative deviation of 100% (peak_dev = baseline_mean) gives score 0.
    # This is the simplest defensible mapping; nonlinear transforms were
    # tested but added complexity without improving rank ordering.
    resistance_score = max(0.0, 1.0 - resistance["relative_peak_deviation"])

    # Recovery score: exponential decay with time constant 5 years.
    # The 5-year constant means ~37% credit at 5 years, ~14% at 10, ~5% at 15.
    # This reflects clinical reality: a mitochondrial network that takes >5
    # years to recover from a single shock has sustained significant lasting
    # damage, even if it eventually returns to baseline.
    recovery_score = float(np.exp(-recovery_time / 5.0)) if np.isfinite(recovery_time) else 0.0

    # Regime score: binary.  This is deliberately all-or-nothing because
    # crossing the heteroplasmy cliff is a qualitative state change —
    # there is no "partial" regime shift.  Either the cell's energy
    # metabolism returned to the viable basin, or it did not.  This is
    # the most important component: a system that shifts regime has lost
    # ecological resilience regardless of how well it scores on the other
    # three metrics.
    regime_score = 1.0 if regime["regime_retained"] else 0.0

    # Elasticity score: logistic sigmoid centered at elasticity=0, with
    # steepness 10.  The sigmoid maps the unbounded elasticity value to
    # [0, 1]: strong recovery (elasticity >> 0) saturates near 1.0, zero
    # recovery maps to 0.5, and active divergence (elasticity << 0)
    # saturates near 0.0.  The steepness of 10 was chosen so that an
    # elasticity of +-0.3 (typical range in our ODE) spans most of the
    # sigmoid's dynamic range.
    elasticity_score = float(1.0 / (1.0 + np.exp(-10.0 * elasticity)))

    # ── Weighting rationale ──
    #
    # The weights (0.25, 0.30, 0.30, 0.15) reflect a hierarchy of clinical
    # importance:
    #
    #   Regime retention (0.30) — highest, tied with recovery.  A regime
    #       shift (crossing the heteroplasmy cliff) is catastrophic and
    #       often irreversible; this must dominate the score.
    #
    #   Recovery time (0.30) — tied with regime.  Even within the same
    #       regime, prolonged ATP deficit causes real organ damage (Cramer
    #       Ch. IV); recovery speed directly maps to cumulative harm.
    #
    #   Resistance (0.25) — important but secondary.  A large transient
    #       deviation that quickly recovers (high resistance * low recovery
    #       time) is clinically less concerning than a small deviation
    #       that never recovers.
    #
    #   Elasticity (0.15) — supplementary.  It refines the recovery picture
    #       but is partially redundant with recovery_time; its main value
    #       is distinguishing "snap-back" vs "slow-drift" recovery profiles
    #       that have similar total recovery times.
    #
    # These weights were validated via sensitivity analysis: perturbation
    # probing (perturbation_probing.py) confirmed that the summary score
    # rank-orders patient fragility consistently with the 4-pillar
    # analytics from analytics.py.
    summary = 0.25 * resistance_score + 0.30 * recovery_score + \
              0.30 * regime_score + 0.15 * elasticity_score

    return {
        "resistance": resistance,
        "recovery_time_years": float(recovery_time) if np.isfinite(recovery_time) else 999.0,
        "regime": regime,
        "elasticity": elasticity,
        "summary_score": round(summary, 4),
        "component_scores": {
            "resistance": round(resistance_score, 4),
            "recovery": round(recovery_score, 4),
            "regime": round(regime_score, 4),
            "elasticity": round(elasticity_score, 4),
        },
    }


# ---------------------------------------------------------------------------
# MAGNITUDE SWEEP — Probing the resilience landscape
#
# Ecological resilience is not a single number but a function of
# disturbance intensity.  A grassland may be fully resilient to moderate
# grazing but collapse under severe overgrazing.  Sweeping disturbance
# magnitude traces out the "resilience curve" and reveals the critical
# threshold where the system transitions from recoverable to irreversible
# — the mitochondrial analog of a dose-response curve in toxicology.
# ---------------------------------------------------------------------------

def compute_resilience_sweep(
    disturbance_class,
    magnitudes: list[float],
    intervention: dict[str, float] | None = None,
    patient: dict[str, float] | None = None,
    start_year: float = 10.0,
    state_idx: int = 2,
) -> list[dict]:
    """Sweep disturbance magnitude and compute resilience at each level.

    Args:
        disturbance_class: Disturbance subclass to instantiate.
        magnitudes: List of magnitude values to test.
        intervention: Intervention dict (default: no treatment).
        patient: Patient dict (default: typical 70yo).
        start_year: When to apply disturbance.
        state_idx: State variable to analyze.

    Returns:
        List of dicts, each with "magnitude" and resilience metrics.
    """
    from disturbances import simulate_with_disturbances
    from simulator import simulate

    # Single baseline shared across all magnitudes — the unperturbed
    # reference trajectory is independent of disturbance magnitude.
    baseline = simulate(intervention=intervention, patient=patient)
    results = []

    for mag in magnitudes:
        shock = disturbance_class(start_year=start_year, magnitude=mag)
        shocked = simulate_with_disturbances(
            intervention=intervention, patient=patient,
            disturbances=[shock])
        metrics = compute_resilience(shocked, baseline, state_idx=state_idx)
        metrics["magnitude"] = mag
        metrics["disturbance"] = shock.name
        results.append(metrics)

    return results


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulator import simulate
    from disturbances import (
        simulate_with_disturbances, IonizingRadiation,
        ChemotherapyBurst, InflammationBurst,
    )

    print("=" * 70)
    print("Resilience Metrics — Standalone Test")
    print("=" * 70)

    baseline = simulate()

    for DistClass, name in [
        (IonizingRadiation, "Ionizing Radiation"),
        (ChemotherapyBurst, "Chemotherapy"),
        (InflammationBurst, "Inflammation"),
    ]:
        shock = DistClass(start_year=10.0, magnitude=0.8)
        result = simulate_with_disturbances(disturbances=[shock])
        metrics = compute_resilience(result, baseline)

        print(f"\n--- {name} (mag=0.8) ---")
        print(f"  Resistance (peak): {metrics['resistance']['peak_deviation']:.4f}")
        print(f"  Resistance (rel):  {metrics['resistance']['relative_peak_deviation']:.4f}")
        print(f"  Recovery time:     {metrics['recovery_time_years']:.2f} years")
        print(f"  Regime retained:   {metrics['regime']['regime_retained']}")
        print(f"  Het gap:           {metrics['regime']['het_gap']:+.4f}")
        print(f"  Elasticity:        {metrics['elasticity']:.4f}")
        print(f"  Summary score:     {metrics['summary_score']:.4f}")

    # Magnitude sweep
    print("\n--- Magnitude sweep: Radiation 0.1 → 1.0 ---")
    sweep = compute_resilience_sweep(
        IonizingRadiation, [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    for s in sweep:
        print(f"  mag={s['magnitude']:.1f}: "
              f"resist={s['resistance']['relative_peak_deviation']:.3f}  "
              f"recovery={s['recovery_time_years']:.1f}yr  "
              f"regime={'OK' if s['regime']['regime_retained'] else 'SHIFTED'}  "
              f"score={s['summary_score']:.3f}")

    print("\n" + "=" * 70)
    print("Resilience metrics tests completed.")
