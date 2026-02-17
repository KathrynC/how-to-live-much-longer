"""Disturbance framework for mitochondrial resilience analysis.

Models acute biological stressors (radiation, toxins, chemotherapy,
inflammation) that perturb the ODE system, enabling measurement of
cellular resilience — resistance, recovery, and regime retention.

Agroecology-inspired: treats the mitochondrial network as an ecosystem
subject to shocks, with resilience as the key health indicator.

Usage:
    from disturbances import (
        IonizingRadiation, ToxinExposure, ChemotherapyBurst,
        InflammationBurst, simulate_with_disturbances,
    )
    from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT

    shocks = [IonizingRadiation(start_year=5.0, magnitude=0.8)]
    result = simulate_with_disturbances(
        intervention=DEFAULT_INTERVENTION,
        patient=DEFAULT_PATIENT,
        disturbances=shocks,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from constants import (
    N_STATES, SIM_YEARS, DT,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
)
# derivatives(): the 8-variable ODE right-hand side from the core simulator (8D after C11).
# initial_state(): constructs state[0..7] from patient params.
# _resolve_intervention(): handles both plain dicts and InterventionSchedule objects.
# _heteroplasmy_fraction(): legacy N_d / (N_h + N_d), kept for backwards compatibility.
# _total_heteroplasmy(): (N_del + N_pt) / (N_h + N_del + N_pt), total damage metric.
# _deletion_heteroplasmy(): N_del / (N_h + N_del + N_pt), cliff-driving metric.
from simulator import (
    derivatives, initial_state, _resolve_intervention,
    _heteroplasmy_fraction,          # kept for backwards compatibility
    _total_heteroplasmy,             # C11: total het including point mutations
    _deletion_heteroplasmy,          # C11: deletion-only het for cliff logic
)


class Disturbance(ABC):
    """Abstract base for disturbance events.

    Subclasses implement two hooks:
        - modify_state(): impulse change to the state vector at onset
        - modify_params(): parameter modifications during active window

    Attributes:
        name: Human-readable disturbance name.
        start_year: Time (years from sim start) when disturbance begins.
        duration: How long the disturbance is active (years).
        magnitude: Severity on [0, 1] scale.
    """

    def __init__(self, name: str, start_year: float, duration: float,
                 magnitude: float) -> None:
        self.name = name
        self.start_year = start_year
        self.duration = duration
        # Magnitude is clamped to [0, 1] — this is a severity fraction, not a
        # physical dose. All subclass coefficients scale linearly off this value,
        # so 0.0 = no perturbation and 1.0 = maximum biological plausibility.
        self.magnitude = np.clip(magnitude, 0.0, 1.0)
        # _applied_impulse distinguishes the two disturbance mechanisms:
        #   - Impulse (modify_state): instantaneous, one-shot damage at onset.
        #     Represents acute physical/chemical insult (DNA strand breaks,
        #     membrane depolarization, ROS burst). Applied once, then latched.
        #   - Ongoing (modify_params): sustained parameter perturbation for the
        #     active window. Represents the biological aftermath (inflammation
        #     cascade, elevated vulnerability, metabolic stress of detox/repair).
        # This flag ensures the impulse fires exactly once per simulation even
        # if the disturbance object is reused.
        self._applied_impulse = False

    def is_active(self, t: float) -> bool:
        """True if the disturbance is active at time t."""
        return self.start_year <= t < self.start_year + self.duration

    @abstractmethod
    def modify_state(self, state: npt.NDArray[np.float64],
                     t: float) -> npt.NDArray[np.float64]:
        """Apply one-time impulse modification to state vector at onset.

        Called once when t first enters the active window.

        Args:
            state: Current state vector (8,) — 8D after C11 mutation split.
            t: Current time in years.

        Returns:
            Modified state vector (8,).
        """
        ...

    @abstractmethod
    def modify_params(
        self,
        intervention: dict[str, float],
        patient: dict[str, float],
        t: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Modify intervention/patient params during active window.

        Called at every timestep while disturbance is active.

        Args:
            intervention: Current intervention dict (will be copied).
            patient: Current patient dict (will be copied).
            t: Current time in years.

        Returns:
            (modified_intervention, modified_patient) tuple.
        """
        ...


# ── Concrete disturbance types ───────────────────────────────────────────────
#
# Each disturbance implements two complementary perturbation channels:
#
#   1. modify_state() — IMPULSE: instantaneous state vector change at onset.
#      Models the acute physical insult (e.g., ionizing radiation breaking
#      mtDNA strands). Applied once via the _applied_impulse latch.
#
#   2. modify_params() — ONGOING: patient/intervention parameter override
#      at every timestep within the active window. Models the biological
#      aftermath that persists (e.g., chronic inflammation, elevated
#      vulnerability to further damage). These modified params feed into
#      the ODE derivatives, so the effect compounds through the dynamics.
#
# The state vector indices (matching simulator.py, 8D after C11 split):
#   [0] N_healthy          — healthy mtDNA copy number (normalized to ~1.0)
#   [1] N_deletion         — deletion-mutated mtDNA (drives cliff at ~70%)
#   [2] ATP                — energy production (MU/day)
#   [3] ROS                — reactive oxygen species level
#   [4] NAD                — NAD+ cofactor availability
#   [5] Senescent_fraction — fraction of cells in senescence (0..1)
#   [6] Membrane_potential — mitochondrial inner membrane ΔΨ (normalized)
#   [7] N_point            — point-mutated mtDNA (linear growth, C11)


class IonizingRadiation(Disturbance):
    """Ionizing radiation burst: damages mtDNA directly, produces ROS.

    At onset: converts a fraction of healthy mtDNA to damaged.
    During active window: increases genetic vulnerability (higher
    deletion rate) and elevates ROS via increased inflammation.

    Biological basis: Cramer Ch. II.H p.14 — ROS-damage vicious cycle;
    radiation directly causes double-strand breaks in mtDNA.
    """

    def __init__(self, start_year: float = 5.0, duration: float = 1.0,
                 magnitude: float = 0.5) -> None:
        # Default 1-year duration models a prolonged exposure scenario (e.g.,
        # repeated medical imaging, occupational exposure, or the chronic
        # aftermath of a single acute dose as DNA repair plays out).
        super().__init__("Ionizing Radiation", start_year, duration, magnitude)

    def modify_state(self, state, t):
        state = state.copy()
        # IMPULSE: Direct mtDNA strand breaks from ionizing radiation.
        # mtDNA is particularly vulnerable because it lacks histones and has
        # limited repair machinery compared to nuclear DNA (Cramer Ch. II.H p.14).
        # 5% max conversion rate is conservative — a severe acute dose could
        # damage more, but we cap here to keep the system recoverable at
        # moderate magnitudes.
        #
        # MASS CONSERVATION: The transfer is mass-conserving — total copy number
        # (N_h + N_del + N_pt) is unchanged. We subtract `transfer` from N_healthy
        # and add it back split across N_deletion and N_point. This models the
        # physical reality that radiation doesn't destroy mtDNA molecules; it
        # converts healthy copies into dysfunctional mutant copies.
        #
        # C11 70/30 SPLIT: Ionizing radiation produces two lesion types in mtDNA:
        #   - 70% DELETIONS: High-energy radiation causes double-strand breaks
        #     (DSBs). When the mtDNA ring is linearized and misrepaired, large
        #     segments (>3kbp) are lost, producing deletion mutants. These are
        #     the cliff-driving mutations with replication advantage.
        #   - 30% POINT MUTATIONS: Radiation also produces oxidative base damage
        #     (8-oxo-guanine, thymine glycol) via water radiolysis → hydroxyl
        #     radicals. These cause single-base substitutions when replicated.
        #     No replication advantage; linear accumulation.
        # The 70/30 ratio is based on radiobiology literature showing that
        # ionizing radiation's primary mtDNA lesion is strand breakage, with
        # oxidative base damage as a secondary effect. For nuclear DNA the
        # ratio would differ, but mtDNA's circular topology and lack of
        # chromatin packaging make it especially susceptible to DSBs.
        damage_fraction = 0.05 * self.magnitude  # up to 5% of healthy pool
        transfer = damage_fraction * state[0]
        state[0] -= transfer                     # healthy copies lost
        state[1] += transfer * 0.7               # 70% → deletions (DSBs)
        state[7] += transfer * 0.3               # 30% → point mutations (base damage)
        # IMPULSE: Acute ROS burst from water radiolysis and damaged ETC.
        # Ionizing radiation splits water molecules into hydroxyl radicals
        # (immediate ROS), and newly damaged mitochondria with defective
        # electron transport chains leak additional superoxide (Cramer Ch. II.H
        # p.14 — the ROS-damage vicious cycle). 0.15 is moderate relative to
        # the baseline ROS of ~0.1, representing a transient ~150% spike.
        state[3] += 0.15 * self.magnitude
        return state

    def modify_params(self, intervention, patient, t):
        if self.is_active(t):
            patient = dict(patient)
            # ONGOING: Radiation-damaged DNA repair machinery makes the cell
            # more susceptible to further mtDNA deletions during the exposure
            # window. genetic_vulnerability feeds into the deletion rate in the
            # ODE (simulator.py:_deletion_rate), so a 50% increase at max
            # magnitude substantially accelerates the damage accumulation.
            patient["genetic_vulnerability"] *= (1.0 + 0.5 * self.magnitude)
            # ONGOING: Radiation triggers NF-kB inflammatory signaling and
            # activates innate immune responses. The 0.2 additive increase
            # (at max magnitude) represents moderate systemic inflammation —
            # enough to accelerate senescence via SASP but not catastrophic.
            patient["inflammation_level"] = min(
                patient["inflammation_level"] + 0.2 * self.magnitude, 1.0)
        return intervention, patient


class ToxinExposure(Disturbance):
    """Environmental toxin exposure: damages membrane potential, boosts ROS.

    At onset: drops membrane potential and NAD.
    During active window: increases inflammation and metabolic demand
    (detoxification costs energy).

    NOTE (C11): ToxinExposure does NOT directly modify mtDNA copy counts
    (N_healthy, N_deletion, N_point). Environmental toxins primarily damage
    the electron transport chain proteins and lipid membranes rather than
    breaking or mutating DNA strands. The mtDNA damage from toxins occurs
    INDIRECTLY: ETC disruption → elevated ROS → oxidative DNA damage, which
    is captured by the ODE dynamics (ROS_POINT_COEFF drives point mutations,
    ROS-mediated strand breaks drive deletions). No 70/30 split is needed
    here because the toxin doesn't directly create DNA lesions.

    Biological basis: Cramer Ch. IV pp.46-47 (membrane potential),
    Ch. VI.B p.75 (low ΔΨ triggers mitophagy).
    """

    def __init__(self, start_year: float = 5.0, duration: float = 2.0,
                 magnitude: float = 0.5) -> None:
        # Default 2-year duration — longer than radiation because toxin
        # clearance depends on hepatic/renal processing, and many environmental
        # toxins (heavy metals, persistent organics) bioaccumulate.
        super().__init__("Toxin Exposure", start_year, duration, magnitude)

    def modify_state(self, state, t):
        state = state.copy()
        # IMPULSE: Membrane potential (ΔΨ) drop from direct ETC poisoning.
        # Many toxins (rotenone, cyanide, antimycin A) inhibit specific
        # complexes of the electron transport chain, collapsing the proton
        # gradient (Cramer Ch. IV pp.46-47). A 20% drop at max magnitude
        # is significant — PINK1 starts accumulating on depolarized
        # mitochondria at ~60-70% of normal ΔΨ (Cramer Ch. VI.B p.75),
        # triggering mitophagy. Multiplicative, not additive, because ΔΨ
        # is a voltage that scales with existing gradient.
        state[6] *= (1.0 - 0.2 * self.magnitude)
        # IMPULSE: NAD+ depletion via PARP activation. Toxin-induced DNA
        # damage triggers poly(ADP-ribose) polymerase (PARP), which consumes
        # NAD+ as substrate for DNA repair. At high toxin loads, PARP
        # hyperactivation can deplete the NAD+ pool (Cramer Ch. VI.A.3 pp.72-73).
        # 10% max depletion is moderate — acute PARP storms can be worse,
        # but we model the partial consumption typical of sublethal exposure.
        state[4] *= (1.0 - 0.1 * self.magnitude)
        # IMPULSE: ROS from ETC disruption. Blocked electron flow at
        # complexes I/III diverts electrons to O2, forming superoxide.
        # 0.1 additive on baseline ~0.1 = ~100% transient spike.
        state[3] += 0.1 * self.magnitude
        return state

    def modify_params(self, intervention, patient, t):
        if self.is_active(t):
            patient = dict(patient)
            # ONGOING: Toxin-induced inflammatory response. The immune system
            # mounts a sustained inflammatory response to cellular damage, and
            # many toxins directly activate inflammasomes. 0.3 is stronger than
            # radiation's 0.2 because chemical toxins often trigger broader
            # tissue damage and immune activation.
            patient["inflammation_level"] = min(
                patient["inflammation_level"] + 0.3 * self.magnitude, 1.0)
            # ONGOING: Detoxification is energetically expensive. Phase I
            # (cytochrome P450) and Phase II (conjugation) detox pathways
            # consume ATP, NADPH, and glutathione. This extra metabolic demand
            # competes with normal cellular function, effectively starving
            # mitochondria of resources. Capped at 2.0 to prevent unphysical
            # demand levels.
            patient["metabolic_demand"] = min(
                patient["metabolic_demand"] + 0.2 * self.magnitude, 2.0)
        return intervention, patient


class ChemotherapyBurst(Disturbance):
    """Chemotherapy treatment: massive ROS, NAD depletion, mtDNA damage.

    The most severe disturbance type. Models cytotoxic chemotherapy
    which collaterally damages mitochondria along with cancer cells.

    At onset: large state perturbation across multiple variables.
    During active window: sustained elevation of damage mechanisms.

    Biological basis: Cramer Ch. VII (cellular damage mechanisms);
    clinical scenario "post_chemo_55" in constants.py.
    """

    def __init__(self, start_year: float = 5.0, duration: float = 0.5,
                 magnitude: float = 0.8) -> None:
        # Default 0.5-year (6-month) duration models a typical chemo cycle.
        # Shorter than toxin exposure because treatment is episodic, but the
        # default magnitude is higher (0.8) because chemo agents are designed
        # to be cytotoxic — collateral mitochondrial damage is severe.
        super().__init__("Chemotherapy", start_year, duration, magnitude)

    def modify_state(self, state, t):
        state = state.copy()
        # IMPULSE: Direct mtDNA damage — 2x the radiation coefficient (0.10
        # vs 0.05) because chemo agents (cisplatin, doxorubicin) are designed
        # to intercalate/crosslink DNA. mtDNA is especially vulnerable due to
        # proximity to ETC ROS production and lack of protective histones.
        # Cisplatin forms platinum-DNA adducts; doxorubicin intercalates and
        # inhibits topoisomerase II. At max magnitude, 10% of healthy copies
        # are instantly rendered dysfunctional.
        #
        # MASS CONSERVATION: Same as IonizingRadiation — transfer is from
        # N_healthy into N_deletion + N_point; total copy number unchanged.
        #
        # C11 70/30 SPLIT: Chemotherapy agents produce two lesion types:
        #   - 70% DELETIONS: Cisplatin creates interstrand crosslinks that stall
        #     replication forks; attempted bypass leads to double-strand breaks
        #     and large deletions. Doxorubicin inhibits topoisomerase II,
        #     trapping cleavage complexes that produce DSBs. Both mechanisms
        #     generate the large-deletion mutants that drive cliff dynamics.
        #   - 30% POINT MUTATIONS: Doxorubicin's redox cycling generates
        #     superoxide → 8-oxo-guanine point mutations. Cisplatin also forms
        #     monoadducts (single-base platinum lesions) that cause miscoding
        #     during replication. These impair ETC function without conferring
        #     replication advantage.
        # The 70/30 ratio matches radiation because both insults are dominated
        # by strand-break chemistry (crosslinks/intercalation for chemo,
        # direct ionization/radiolysis for radiation), with oxidative base
        # damage as the secondary pathway.
        damage_fraction = 0.1 * self.magnitude
        transfer = damage_fraction * state[0]
        state[0] -= transfer                     # healthy copies lost
        state[1] += transfer * 0.7               # 70% → deletions (crosslinks, DSBs)
        state[7] += transfer * 0.3               # 30% → point mutations (adducts, oxidative)
        # IMPULSE: Massive ROS burst. Doxorubicin generates superoxide via
        # redox cycling, and cisplatin disrupts ETC complex activity. 0.3 on
        # baseline ~0.1 = 300% spike — the most severe of all disturbance
        # types, reflecting the designed cytotoxicity of these agents.
        state[3] += 0.3 * self.magnitude
        # IMPULSE: NAD+ crash from dual PARP activation (DNA damage response)
        # and direct mitochondrial Complex I inhibition. 25% max depletion is
        # 2.5x the toxin coefficient, reflecting the more severe DNA damage
        # load that triggers sustained PARP hyperactivation.
        state[4] *= (1.0 - 0.25 * self.magnitude)
        # IMPULSE: Membrane potential collapse from ETC disruption across
        # multiple complexes simultaneously. Less severe than the NAD crash
        # (15% vs 25%) because ΔΨ partially recovers through remaining
        # functional complexes, while NAD depletion is stoichiometric.
        state[6] *= (1.0 - 0.15 * self.magnitude)
        return state

    def modify_params(self, intervention, patient, t):
        if self.is_active(t):
            patient = dict(patient)
            # ONGOING: Chemo-induced genomic instability. DNA damage response
            # is overwhelmed, leaving residual unrepaired lesions that make the
            # genome fragile. 0.8 multiplier (vs radiation's 0.5) reflects that
            # chemo agents specifically target DNA, creating far more lesions
            # per unit time than background radiation.
            patient["genetic_vulnerability"] *= (1.0 + 0.8 * self.magnitude)
            # ONGOING: Chemo triggers severe systemic inflammation — tumor
            # lysis, neutrophil activation, cytokine storm from dying cells.
            # 0.4 is the highest inflammation coefficient across all disturbance
            # types (radiation=0.2, toxin=0.3, inflammation_burst=0.5 but that
            # IS inflammation, so it should be highest).
            patient["inflammation_level"] = min(
                patient["inflammation_level"] + 0.4 * self.magnitude, 1.0)
            # ONGOING: Massive metabolic overhead from DNA repair, protein
            # resynthesis, immune activation, and clearance of dead cells.
            # 0.3 is the highest metabolic_demand coefficient, reflecting
            # that the body is simultaneously fighting cancer, repairing
            # collateral damage, and mounting an immune response.
            patient["metabolic_demand"] = min(
                patient["metabolic_demand"] + 0.3 * self.magnitude, 2.0)
        return intervention, patient


class InflammationBurst(Disturbance):
    """Acute systemic inflammation (infection, fever, autoimmune flare).

    At onset: immediate ROS and senescence pressure increase.
    During active window: elevated inflammation and metabolic demand.

    NOTE (C11): InflammationBurst does NOT directly modify mtDNA copy counts
    (N_healthy, N_deletion, N_point). Inflammation damages cells through
    cytokine signaling (TNF-alpha, IL-6, IL-1beta) and immune cell ROS
    bursts, not through direct DNA strand breaks or base modifications.
    The mtDNA damage from inflammation occurs INDIRECTLY via two ODE paths:
      1. Elevated ROS (from immune cell respiratory burst) → point mutations
         and deletions through normal ROS-damage coupling.
      2. SASP-driven senescence → reduced mitophagy capacity → accumulation
         of both mutation types via impaired quality control.
    No 70/30 split is needed because inflammation acts through systemic
    signaling cascades, not direct DNA lesion chemistry.

    Biological basis: Cramer Ch. VII.A pp.89-90 (SASP),
    Ch. VIII.F p.103 (senescent cells use ~2x energy).
    """

    def __init__(self, start_year: float = 5.0, duration: float = 0.5,
                 magnitude: float = 0.5) -> None:
        # Default 0.5-year duration models a severe infection or autoimmune
        # flare (e.g., COVID, sepsis, lupus flare). Shorter than toxin/radiation
        # because the immune system resolves acute inflammation faster than the
        # body clears chemical insults — but the inflammatory damage accrues.
        super().__init__("Inflammation Burst", start_year, duration, magnitude)

    def modify_state(self, state, t):
        state = state.copy()
        # IMPULSE: ROS elevation from activated neutrophils and macrophages.
        # Immune cells deliberately produce ROS (respiratory burst) to kill
        # pathogens, but this also damages bystander mitochondria. 0.1 is the
        # same as ToxinExposure — moderate, because inflammation-driven ROS is
        # less concentrated than direct ETC poisoning or radiation radiolysis.
        state[3] += 0.1 * self.magnitude
        # IMPULSE: Direct senescence induction. Acute inflammation triggers
        # stress-induced premature senescence (SIPS) in affected cells.
        # Senescent cells then emit SASP (Cramer Ch. VII.A pp.89-90), creating
        # a positive feedback loop — this is the "inflammaging" cascade.
        # 0.02 additive increase (at max magnitude) on a baseline ~0.05 is a
        # ~40% jump, representing a significant but sublethal senescence wave.
        # Capped at 1.0 because senescent_fraction is a true fraction.
        state[5] = min(state[5] + 0.02 * self.magnitude, 1.0)
        return state

    def modify_params(self, intervention, patient, t):
        if self.is_active(t):
            patient = dict(patient)
            # ONGOING: Sustained inflammation is this disturbance's primary
            # mechanism — it IS an inflammation event, so the coefficient (0.5)
            # is the highest of all disturbance types. This feeds directly
            # into the ODE's senescence and ROS terms, creating the SASP
            # positive feedback loop described in Cramer Ch. VII.A.
            patient["inflammation_level"] = min(
                patient["inflammation_level"] + 0.5 * self.magnitude, 1.0)
            # ONGOING: Fever and immune activation increase basal metabolic
            # rate. Senescent cells consume ~2x normal energy (Cramer Ch. VIII.F
            # p.103), so even a modest senescence wave strains ATP supply.
            # 0.15 is the lowest metabolic_demand coefficient — inflammation
            # is less metabolically costly than detoxification (toxin: 0.2) or
            # full chemo recovery (chemo: 0.3) because the immune response
            # itself is the energy consumer, not a repair process.
            patient["metabolic_demand"] = min(
                patient["metabolic_demand"] + 0.15 * self.magnitude, 2.0)
        return intervention, patient


# ── Simulation with disturbances ─────────────────────────────────────────────


def simulate_with_disturbances(
    intervention: dict[str, float] | None = None,
    patient: dict[str, float] | None = None,
    disturbances: list[Disturbance] | None = None,
    sim_years: float | None = None,
    dt: float | None = None,
) -> dict:
    """Run mitochondrial aging simulation with disturbance events.

    Wraps the ODE integration loop from simulator.py, injecting
    disturbance effects (state impulses + parameter modifications)
    at the appropriate times.

    Args:
        intervention: Dict of 6 intervention params (defaults to no treatment).
        patient: Dict of 6 patient params (defaults to typical 70yo).
        disturbances: List of Disturbance objects to apply during simulation.
        sim_years: Override simulation horizon (default: constants.SIM_YEARS).
        dt: Override timestep (default: constants.DT).

    Returns:
        Dict with:
            "time": np.array of time points
            "states": np.array of shape (n_steps+1, 8) — 8D after C11 split
            "heteroplasmy": np.array of total heteroplasmy at each step
            "deletion_heteroplasmy": np.array of deletion-only het (drives cliff)
            "intervention": intervention dict used
            "patient": patient dict used
            "disturbances": list of disturbance event dicts
            "shock_times": list of (start, end) tuples for each disturbance
    """
    # Default to no-treatment baseline and typical 70-year-old patient.
    # These are copied (dict()) to avoid mutating the module-level defaults.
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    if patient is None:
        patient = dict(DEFAULT_PATIENT)
    if disturbances is None:
        disturbances = []
    if sim_years is None:
        sim_years = SIM_YEARS
    if dt is None:
        dt = DT

    # The integration mirrors simulator.simulate() but cannot reuse it directly
    # because disturbances need to inject state modifications and parameter
    # overrides *inside* the integration loop, between the constraint enforcement
    # and the RK4 step. The core simulator doesn't expose these injection points.
    n_steps = int(sim_years / dt)
    # initial_state() constructs state[0..7] from patient params — sets N_h, N_del,
    # N_pt from baseline_heteroplasmy (90/10 del/pt split), ATP from cliff factor,
    # ROS from damage level, NAD from baseline_nad_level, senescence from age,
    # ΔΨ from energy state.
    state = initial_state(patient)

    # Pre-allocate contiguous arrays for the full trajectory.
    # N_STATES = 8 (the ODE system dimension, 8D after C11 mutation split).
    time_arr = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, N_STATES))
    het_arr = np.zeros(n_steps + 1)
    del_het_arr = np.zeros(n_steps + 1)  # C11: deletion-only het (drives cliff)

    # Record initial conditions at t=0.
    states[0] = state
    het_arr[0] = _total_heteroplasmy(state[0], state[1], state[7])
    del_het_arr[0] = _deletion_heteroplasmy(state[0], state[1], state[7])

    # Reset impulse flags so disturbance objects can be reused across
    # multiple simulate_with_disturbances() calls without stale state.
    for d in disturbances:
        d._applied_impulse = False

    for i in range(n_steps):
        t = i * dt

        # --- Phase 1: Impulse injection (once per disturbance) ---
        # Check each disturbance for first entry into its active window.
        # The impulse models instantaneous physical/chemical damage at onset.
        # Constraint enforcement after the impulse prevents the state from
        # going negative (e.g., ROS burst can't make N_healthy negative if
        # damage_fraction is well-calibrated, but clamp anyway for safety).
        for d in disturbances:
            if d.is_active(t) and not d._applied_impulse:
                state = d.modify_state(state, t)
                # All state variables must be non-negative (biological quantities).
                state = np.maximum(state, 0.0)
                # Senescent fraction is a true probability — hard cap at 1.0.
                state[5] = min(state[5], 1.0)
                d._applied_impulse = True

        # --- Phase 2: Parameter modification (every active timestep) ---
        # _resolve_intervention handles both plain dicts and InterventionSchedule
        # objects, returning the intervention dict active at time t.
        current_intervention = _resolve_intervention(intervention, t)
        # Copy the base patient params so disturbances overlay without
        # permanently mutating the originals. Each disturbance chains its
        # modifications — if multiple disturbances are active simultaneously,
        # their effects stack (e.g., radiation + inflammation = both elevated
        # vulnerability AND elevated inflammation).
        current_patient = dict(patient)
        for d in disturbances:
            current_intervention, current_patient = d.modify_params(
                current_intervention, current_patient, t)

        # --- Phase 3: RK4 integration step ---
        # Classical 4th-order Runge-Kutta with the (possibly perturbed)
        # parameters. The derivatives() function computes all 8 state
        # derivatives from the current state, time, intervention, and patient
        # params. RK4 evaluates derivatives at 4 points per step for O(dt^5)
        # local error — critical for capturing the nonlinear cliff dynamics
        # where Euler methods would overshoot. Note: the perturbed parameters
        # are held constant across all 4 RK4 sub-evaluations within a single
        # timestep, which is valid because dt (~3.65 days) is much shorter
        # than any disturbance duration (months to years).
        k1 = derivatives(state, t, current_intervention, current_patient)
        k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt,
                         current_intervention, current_patient)
        k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt,
                         current_intervention, current_patient)
        k4 = derivatives(state + dt * k3, t + dt,
                         current_intervention, current_patient)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # --- Phase 4: Post-step constraint enforcement ---
        # Same constraints as the core simulator: biological quantities
        # cannot go negative, and senescent fraction cannot exceed 1.0.
        # Without these, numerical overshoot near the cliff (where derivatives
        # are steep) could produce unphysical states that destabilize the ODE.
        state = np.maximum(state, 0.0)
        state[5] = min(state[5], 1.0)

        # Record this step's results.
        time_arr[i + 1] = (i + 1) * dt
        states[i + 1] = state
        het_arr[i + 1] = _total_heteroplasmy(state[0], state[1], state[7])
        del_het_arr[i + 1] = _deletion_heteroplasmy(state[0], state[1], state[7])

    # Package disturbance metadata for downstream analysis and plotting.
    # shock_times enables shaded regions on trajectory plots; disturbance_info
    # preserves the full parameterization for reproducibility.
    shock_times = [(d.start_year, d.start_year + d.duration) for d in disturbances]
    disturbance_info = [
        {"name": d.name, "start_year": d.start_year,
         "duration": d.duration, "magnitude": d.magnitude}
        for d in disturbances
    ]

    return {
        "time": time_arr,
        "states": states,
        "heteroplasmy": het_arr,
        "deletion_heteroplasmy": del_het_arr,
        "intervention": intervention,
        "patient": patient,
        "disturbances": disturbance_info,
        "shock_times": shock_times,
    }


# ── Standalone test ──────────────────────────────────────────────────────────
#
# Smoke test: runs each disturbance type at year 10 and compares ATP/het
# trajectories against undisturbed baseline. Verifies that:
#   1. Each shock causes a measurable ATP drop at onset (impulse works).
#   2. Final heteroplasmy is higher than baseline (damage accumulates).
#   3. Multiple shocks compound (multi-shock test).

if __name__ == "__main__":
    print("=" * 70)
    print("Disturbance Framework — Standalone Test")
    print("=" * 70)

    from simulator import simulate

    # Baseline: no-treatment, default 70yo patient, no disturbances.
    # This is the reference trajectory for measuring disturbance impact.
    baseline = simulate()

    # Test each disturbance type individually at year 10 (after the system
    # has settled from initial transients but before age-related decline
    # dominates). Magnitudes chosen to produce clearly visible but non-
    # catastrophic perturbations for visual inspection.
    for DistClass, kwargs in [
        (IonizingRadiation, {"start_year": 10.0, "magnitude": 0.8}),
        (ToxinExposure, {"start_year": 10.0, "magnitude": 0.6}),
        (ChemotherapyBurst, {"start_year": 10.0, "magnitude": 0.8}),
        (InflammationBurst, {"start_year": 10.0, "magnitude": 0.7}),
    ]:
        shock = DistClass(**kwargs)
        result = simulate_with_disturbances(disturbances=[shock])

        # Compare ATP 1 step before and 10 steps after shock onset.
        # The 10-step offset (~36 days) allows the impulse + initial ODE
        # response to manifest — checking immediately at onset would only
        # show the impulse, not the dynamical cascade.
        shock_idx = int(shock.start_year / DT)
        pre_atp = result["states"][shock_idx - 1, 2]
        post_atp = result["states"][shock_idx + 10, 2]

        print(f"\n--- {shock.name} (mag={shock.magnitude}) ---")
        print(f"  Pre-shock ATP:  {pre_atp:.4f}")
        print(f"  Post-shock ATP: {post_atp:.4f}  (delta={post_atp - pre_atp:+.4f})")
        print(f"  Final het:      {result['heteroplasmy'][-1]:.4f}  "
              f"(baseline: {baseline['heteroplasmy'][-1]:.4f})")
        print(f"  Final ATP:      {result['states'][-1, 2]:.4f}  "
              f"(baseline: {baseline['states'][-1, 2]:.4f})")

    # Multi-shock scenario: tests that disturbance stacking works correctly.
    # Radiation at year 5 weakens the system; chemo at year 15 hits a
    # pre-damaged mitochondrial network. The key question is whether the
    # combined effect is worse than either alone (expected: yes, because the
    # first shock shifts the system closer to the heteroplasmy cliff,
    # making it more vulnerable to the second).
    print("\n--- Multi-shock: radiation at year 5, chemo at year 15 ---")
    shocks = [
        IonizingRadiation(start_year=5.0, magnitude=0.6),
        ChemotherapyBurst(start_year=15.0, magnitude=0.7),
    ]
    result_multi = simulate_with_disturbances(disturbances=shocks)
    print(f"  Final het: {result_multi['heteroplasmy'][-1]:.4f}")
    print(f"  Final ATP: {result_multi['states'][-1, 2]:.4f}")

    print("\n" + "=" * 70)
    print("Disturbance tests completed.")
